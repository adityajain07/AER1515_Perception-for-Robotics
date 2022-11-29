import os
import sys
import json
from PIL import Image

import cv2
import numpy as np
import kitti_dataHandler


def segmentation_from_depth(depth_map, bbox_list, threshold, 
                            crop_depth_dir, sample_name):
    """
    Builds segmentation mask from the depth image, 
    given 2D object detection results and a segmentation threshold
    
    Args:
        depth_map      : depth map of the given image
        bbox_list      : 2D bounding box detection data
        threshold      : segmentation threshold
        crop_depth_dir : directory to save the cropped images [for test purposes]
        sample_name    : original name
    Returns:
        segmentation mask image
    """
    seg_mask = 255*np.ones(depth_map.shape)
    cnt = 1

    for key in bbox_list.keys():
        row_min, row_max = bbox_list[key]['1'], bbox_list[key]['1']+bbox_list[key]['3']
        col_min, col_max = bbox_list[key]['0'], bbox_list[key]['0']+bbox_list[key]['2']
        det_object       = depth_map[row_min:row_max+1, col_min:col_max+1]

        ## saving the cropped images for testing
        # det_object_img = Image.fromarray(det_object)
        # det_object_img.save(crop_depth_dir + '/' + sample_name + '_' + str(cnt) + '.png')
        # cnt += 1

        # average pixel value calculation 
        non_zero_idx  = np.nonzero(det_object)
        non_zero_elem = det_object[non_zero_idx]
        if len(non_zero_elem)!=0:
            avg_depth     = np.sum(non_zero_elem)/len(non_zero_elem)

            # converting pixels near the average to zero 
            det_object_mask = 255*np.ones(det_object.shape)
            idx_object = np.where(np.logical_and(det_object>=avg_depth-threshold, det_object<=avg_depth+threshold))
            det_object_mask[idx_object] = 0

            # map to original image
            seg_mask[row_min:row_max+1, col_min:col_max+1] = det_object_mask        
            
    seg_image = Image.fromarray(seg_mask)
    seg_image = seg_image.convert('L')
    return seg_image

def main():

    ################
    # Options
    ################
    # Input dir and output dir
    seg_thresh = 7
    depth_dir = './data/test/est_depth/'
    bbox_dir  = './data/test/est_bboxes/'
    crop_dep_dir = './data/test/crop_depth/'
    label_dir = './data/test/gt_labels'
    output_dir = './data/test/est_segmentation/'
    sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    for sample_name in sample_list:
    	# Read depth map
        depth_map = cv2.imread(depth_dir + sample_name + '.png', 0)

        # Read 2d bbox
        bbox_data = json.load(open(bbox_dir + sample_name + '.json'))

        # For each bbox
            # Estimate the average depth of the objects

            # Find the pixels within a certain distance from the centroid

        # Save the segmentation mask
        seg_mask = segmentation_from_depth(depth_map, bbox_data, seg_thresh, 
                                            crop_dep_dir, sample_name)
        seg_mask.save(output_dir + sample_name + '.png')
        # break


if __name__ == '__main__':
    main()
