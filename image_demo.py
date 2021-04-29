import cv2
import time
import argparse
import os
import torch

import posenet

model_size = 101
scale_factor = 1.0
max_pose_detections = 10
min_pose_score = 0.25
min_part_score = 0.25

def main(path):
    model = posenet.load_model(model_size)
    model = model.cuda()
    output_stride = model.output_stride

    filenames = [
        f.path for f in os.scandir(path) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

    start = time.time()
    pre_centroids = []
    cnt = 0
    for f in filenames:
        time.sleep(0.03)
        input_image, draw_image, output_scale = posenet.read_imgfile(
            f, scale_factor=scale_factor, output_stride=output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=max_pose_detections,
                min_pose_score=min_pose_score)

        keypoint_coords *= output_scale

        centroids, tilt_angles, widths, heights, left_up_points, right_bottom_points = posenet.get_potential_information(pose_scores, keypoint_coords, min_pose_score=min_pose_score)
        isFailed = posenet.get_isFailed(pre_centroids, centroids, tilt_angles, widths, heights, 0.03)
        pre_centroids = centroids
        draw_image = posenet.draw_box_and_center(draw_image, centroids, left_up_points, right_bottom_points, isFailed) 
        if isFailed.count(True):
            cv2.imwrite('output_image/'+str(cnt)+'.png', draw_image)
            cnt = cnt + 1
        cv2.imshow('Fall detection system', draw_image)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main('images')
    
