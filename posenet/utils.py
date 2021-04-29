import cv2
import numpy as np
import math
import posenet.constants


def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, target_height, target_width)
    return input_img, source_img, scale


def read_cap(cap, scale_factor=1.0, output_stride=16):
    res, img = cap.read()
    if not res:
        raise IOError("webcam failure")
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
    return _process_input(img, scale_factor, output_stride)


def read_imgfile(path, scale_factor=1.0, output_stride=16):
    img = cv2.imread(path)
    return _process_input(img, scale_factor, output_stride)


def draw_keypoints(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results


def draw_skeleton(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img


def draw_skel_and_kp(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.5, min_part_score=0.5):

    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

    if cv_keypoints:
        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img



def draw_box_and_center(img, centroids, left_up_points, right_bottom_points, isFailed):
    out_img = img
    cv_keypoints = []
    box_lines = []
    for i,(center, left_up, right_bottom) in enumerate(zip(centroids, left_up_points, right_bottom_points)):
        center = center.astype(np.int32)
        left_up = tuple(left_up.astype(np.int32)[::-1])
        right_bottom = tuple(right_bottom.astype(np.int32)[::-1])

        cv_keypoints.append(cv2.KeyPoint(center[1],center[0],10))
        out_img = cv2.rectangle(out_img, left_up, right_bottom, color=(255, 255, 0))
        if isFailed[i]:
            out_img = cv2.putText(out_img, 'Falled', left_up, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
    if cv_keypoints:
        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return out_img

def get_potential_information(pose_scores, keypoint_coords, min_pose_score=0.5):#(y,x)
    centroids = []
    tilt_angles = []
    heights = []
    widths = []
    left_up_points = []
    right_bottom_points = []
    for i, score in enumerate(pose_scores):
        if score < min_pose_score:
            continue

        centroid = (keypoint_coords[i][11] + keypoint_coords[i][12])/2
        ankle_center = (keypoint_coords[i][-1] + keypoint_coords[i][-2])/2
        left_up = np.min(keypoint_coords[i],0)
        right_bottom = np.max(keypoint_coords[i],0)
        width = right_bottom[1] - left_up[1]
        height = right_bottom[0] - left_up[0]
        
        centroids.append(centroid)
        tilt_angles.append(get_tilt_angle(ankle_center[1],ankle_center[0],centroid[1],centroid[0]))
        widths.append(width)
        heights.append(height)
        left_up_points.append(left_up)
        right_bottom_points.append(right_bottom)
    return centroids, tilt_angles, widths, heights, left_up_points, right_bottom_points


def get_tilt_angle(x1,y1,x2,y2):
    angle = 0
    if y2<y1:
        angle = math.atan2(y1-y2, math.fabs(x2-x1))/math.pi*180
    return angle 


def get_isFailed(pre_centroids, centroids, tilt_angles, widths, heights, dt):
    isFailed = [False]*len(centroids)
    if len(pre_centroids)==0:
        return isFailed
    for i,(centroid, tilt_angle, width, height) in enumerate(zip(centroids, tilt_angles, widths, heights)):
        for pre_centroid in pre_centroids:
            d = math.sqrt((centroid[0]-pre_centroid[0])*(centroid[0]-pre_centroid[0])
            +(centroid[1]-pre_centroid[1])*(centroid[1]-pre_centroid[1]))
            if d/dt < 666.66666666:
                f1 = math.fabs(centroid[0]-pre_centroid[0])
                f2 = tilt_angle
                f3 = height / width
                if f1 > 4 and f2 < 89 and f3<1.05:
                    isFailed[i] = True
                break
    return isFailed