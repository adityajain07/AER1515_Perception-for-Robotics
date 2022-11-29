import os
import sys
from PIL import Image

import cv2
import numpy as np
import kitti_dataHandler
from kitti_dataHandler import read_frame_calib, get_stereo_calibration


def depth_from_disparity(disp_map, B, f):
    """
    Builds depth map from a disparity map, given baseline and focal length information of stereo camera
    
    Args:
        disp_map  : grayscale disparity map
        B         : baseline of the stereo camera (in meters)
        f         : focal length of the left camera (in pixels)

    Returns:
        depth map image
    """

    depth_img_array = np.zeros(disp_map.shape)
    
    for i in range(disp_map.shape[0]):
        for j in range(disp_map.shape[1]):
            if disp_map[i, j]!=0:
                Z = (B*f)/disp_map[i, j]
                if Z<=80 and Z>=0.1:
                    depth_img_array[i, j] = Z

    depth_img_array = (((depth_img_array - depth_img_array.min())/(depth_img_array.max() - depth_img_array.min()))*255.9).astype(np.uint8)
    depth_img       = Image.fromarray(depth_img_array)
    
    return depth_img

def main():

    ################
    # Options
    ################
    # Input dir and output dir for training
    # disp_dir = './data/train/disparity/'
    # output_dir = './data/train/est_depth/'
    # calib_dir = 'data/train/calib/'
    # sample_list = ['000001', '000002', '000003', '000004', '000005',
    #                 '000006', '000007', '000008', '000009', '000010']
    

    # Input dir and output dir for testing
    disp_dir = 'data/test/disparity/'
    output_dir = 'data/test/est_depth/'
    calib_dir = 'data/test/calib/'
    sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    for sample_name in (sample_list):
        # Read disparity map
        disp_img  = cv2.imread(disp_dir + sample_name + '.png', 0)

        # Read calibration info
        calib_file  = calib_dir + sample_name + '.txt'
        frame_calib = read_frame_calib(calib_file)
        left_cam_mat, right_cam_mat = frame_calib.p2, frame_calib.p3
        stereo_calib   = get_stereo_calibration(left_cam_mat, right_cam_mat)
        B, f = stereo_calib.baseline, stereo_calib.f

        # Save depth map
        depth_map = depth_from_disparity(disp_img, B, f)
        depth_map.save(output_dir + '/' + sample_name + '.png')
        


if __name__ == '__main__':
    main()
