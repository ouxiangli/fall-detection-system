import torch
import cv2
import time
import argparse

import posenet

model_size = 101
cam_id = 0
cam_width = 1280
cam_height = 720
scale_factor = 1.0
max_pose_detections = 10
min_pose_score = 0.25
min_part_score = 0.25


def main():
    model = posenet.load_model(model_size)
    model = model.cuda()
    output_stride = model.output_stride

    cap = cv2.VideoCapture(cam_id)
    cap.set(3, cam_width)
    cap.set(4, cam_height)

    start = time.time()
    frame_count = 0
    pre_centroids = []
    pre_time = start
    cnt = 0
    while True:
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=scale_factor, output_stride=output_stride)
        now = time.time()
        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()
            
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=min_pose_score)

        keypoint_coords *= output_scale

        centroids, tilt_angles, widths, heights, left_up_points, right_bottom_points = posenet.get_potential_information(pose_scores, keypoint_coords, min_pose_score=min_pose_score)
        isFailed = posenet.get_isFailed(pre_centroids, centroids, tilt_angles, widths, heights, now-pre_time)
        pre_centroids = centroids
        pre_time = now
        print(pre_time)
        overlay_image = posenet.draw_box_and_center(display_image, centroids, left_up_points, right_bottom_points, isFailed) 
        if isFailed.count(True):
            cv2.imwrite('output_webcam/'+str(cnt)+'.png', overlay_image)
            cnt = cnt + 1

        cv2.imshow('Fall detection system', overlay_image)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()