from random import sample
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import csv
import os
from PIL import Image

class FrameCalib:
    """Frame Calibration

    Fields:
        p0-p3: (3, 4) Camera P matrices. Contains extrinsic and intrinsic parameters.
        r0_rect: (3, 3) Rectification matrix
        velo_to_cam: (3, 4) Transformation matrix from velodyne to cam coordinate
            Point_Camera = P_cam * R0_rect * Tr_velo_to_cam * Point_Velodyne
        """

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.r0_rect = []
        self.velo_to_cam = []


def read_frame_calib(calib_file_path):
    """Reads the calibration file for a sample

    Args:
        calib_file_path: calibration file path

    Returns:
        frame_calib: FrameCalib frame calibration
    """

    data_file = open(calib_file_path, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    frame_calib = FrameCalib()
    frame_calib.p0 = p_all[0]
    frame_calib.p1 = p_all[1]
    frame_calib.p2 = p_all[2]
    frame_calib.p3 = p_all[3]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calib.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calib.velo_to_cam = np.reshape(tr_v2c, (3, 4))

    return frame_calib


class StereoCalib:
    """Stereo Calibration

    Fields:
        baseline: distance between the two camera centers
        f: focal length
        k: (3, 3) intrinsic calibration matrix
        p: (3, 4) camera projection matrix
        center_u: camera origin u coordinate
        center_v: camera origin v coordinate
        """

    def __init__(self):
        self.baseline = 0.0
        self.f = 0.0
        self.k = []
        self.center_u = 0.0
        self.center_v = 0.0


def krt_from_p(p, fsign=1):
    """Factorize the projection matrix P as P=K*[R;t]
    and enforce the sign of the focal length to be fsign.


    Keyword Arguments:
    ------------------
    p : 3x4 list
        Camera Matrix.

    fsign : int
            Sign of the focal length.


    Returns:
    --------
    k : 3x3 list
        Intrinsic calibration matrix.

    r : 3x3 list
        Extrinsic rotation matrix.

    t : 1x3 list
        Extrinsic translation.
    """
    s = p[0:3, 3]
    q = np.linalg.inv(p[0:3, 0:3])
    u, b = np.linalg.qr(q)
    sgn = np.sign(b[2, 2])
    b = b * sgn
    s = s * sgn

    # If the focal length has wrong sign, change it
    # and change rotation matrix accordingly.
    if fsign * b[0, 0] < 0:
        e = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    if fsign * b[2, 2] < 0:
        e = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    # If u is not a rotation matrix, fix it by flipping the sign.
    if np.linalg.det(u) < 0:
        u = -u
        s = -s

    r = np.matrix.transpose(u)
    t = np.matmul(b, s)
    k = np.linalg.inv(b)
    k = k / k[2, 2]

    # Sanity checks to ensure factorization is correct
    if np.linalg.det(r) < 0:
        print('Warning: R is not a rotation matrix.')

    if k[2, 2] < 0:
        print('Warning: K has a wrong sign.')

    return k, r, t


def get_stereo_calibration(left_cam_mat, right_cam_mat):
    """Extract parameters required to transform disparity image to 3D point
    cloud.

    Keyword Arguments:
    ------------------
    left_cam_mat : 3x4 list
                   Left Camera Matrix.

    right_cam_mat : 3x4 list
                   Right Camera Matrix.


    Returns:
    --------
    stereo_calibration_info : Instance of StereoCalibrationData class
                              Placeholder for stereo calibration parameters.
    """

    stereo_calib = StereoCalib()
    k_left, r_left, t_left = krt_from_p(left_cam_mat)
    _, _, t_right = krt_from_p(right_cam_mat)

    stereo_calib.baseline = abs(t_left[0] - t_right[0])
    stereo_calib.f = k_left[0, 0]
    stereo_calib.k = k_left
    stereo_calib.center_u = k_left[0, 2]
    stereo_calib.center_v = k_left[1, 2]

    return stereo_calib


## Input for training dataset
left_image_dir_train  = os.path.abspath('./training/left')
right_image_dir_train = os.path.abspath('./training/right')
calib_dir_train       = os.path.abspath('./training/calib')
pred_depth_map_train  = os.path.abspath('./training/pred_depth_map')
sample_list_train     = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010']


## Input for test dataset
left_image_dir_test  = os.path.abspath('./test/left')
right_image_dir_test = os.path.abspath('./test/right')
calib_dir_test       = os.path.abspath('./test/calib')
pred_depth_map_test  = os.path.abspath('./test/pred_depth_map')
sample_list_test     = ['000011', '000012', '000013', '000014','000015']

## Output
output_file = open("P3_result.txt", "a")
output_file.truncate(0)


def feature_detection(left_image_dir, right_image_dir, sample_list):
    """
    Feature detection for given left and right images from a stereo camera
    
    Args:
        left_image_dir  : directory containing the left images of the stereo camera
        right_image_dir : directory containing the right images of the stereo camera
        sample_list     : list of image names, without extension

    Returns:
        None
    """

    for sample_name in sample_list:
        left_image_path = left_image_dir +'/' + sample_name + '.png'
        right_image_path = right_image_dir +'/' + sample_name + '.png'

        img_left_orig  = cv.imread(left_image_path)
        img_left_gray  = cv.cvtColor(img_left_orig, cv.COLOR_BGR2GRAY)
        img_right_orig = cv.imread(right_image_path)
        img_right_gray = cv.cvtColor(img_right_orig, cv.COLOR_BGR2GRAY)

        # TODO: Initialize a feature detector
        sift    = cv.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img_left_gray, None)

        # Randomly choose 1000 keypoints from the list
        random_kp_idx = np.random.randint(0, len(kp)-1, 1000)
        kp            = [kp[i] for i in random_kp_idx]

        # Plot keypoints on the image
        cv.drawKeypoints(img_left_orig, kp, img_left_orig)
        img_save_path = os.path.abspath('./feature_detection/sift_keypoints_')
        cv.imwrite(img_save_path + sample_name + '.png', img_left_orig)


def feature_matching(left_image_dir, right_image_dir, sample_list, n=25):
    """
    Feature matching for given left and right images from a stereo camera
    
    Args:
        left_image_dir  : directory containing the left images of the stereo camera
        right_image_dir : directory containing the right images of the stereo camera
        sample_list     : list of image names, without extension
        n               : number of top matches to draw

    Returns:
        None
    """
    sift    = cv.xfeatures2d.SIFT_create()

    for sample_name in sample_list:
        left_image_path = left_image_dir +'/' + sample_name + '.png'
        right_image_path = right_image_dir +'/' + sample_name + '.png'

        # read images
        img_left_orig  = cv.imread(left_image_path)
        img_left_gray  = cv.cvtColor(img_left_orig, cv.COLOR_BGR2GRAY)
        img_right_orig = cv.imread(right_image_path)
        img_right_gray = cv.cvtColor(img_right_orig, cv.COLOR_BGR2GRAY)

        # detect features and their descriptors
        kp_left, des_left   = sift.detectAndCompute(img_left_gray, None)
        kp_right, des_right = sift.detectAndCompute(img_right_gray, None)

        # create BFMatcher object
        bf = cv.BFMatcher(crossCheck=True)

        # match descriptors.
        matches = bf.match(des_left, des_right)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # Draw top n matches.
        matched_img = cv.drawMatches(img_left_orig, kp_left, 
            img_right_orig, kp_right, matches[:n], 
            None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        img_save_path = os.path.abspath('./feature_matching/matched_keypoints_')
        cv.imwrite(img_save_path + sample_name + '.png', matched_img)


def save_depth_image(img_shape, px_u_list, px_v_list, depth_list):
    """
    Given depth values, saves the grayscale image
    
    Args:
        img_shape  : the shape of the image
        px_u_list  : x/u coordinates of the feature (along the columns)
        px_v_list  : y/v coordinates of the feature (along the rows)
        depth_list : corresponding depth values

    Returns:
        None uint8 depth image
    """
    depth_img_array = np.zeros(img_shape)
    
    for i in range(len(depth_list)):
        depth_img_array[px_v_list[i], px_u_list[i]] = depth_list[i]

    depth_img_array = (((depth_img_array - depth_img_array.min())/(depth_img_array.max() - depth_img_array.min()))*255.9).astype(np.uint8)
    depth_img       = Image.fromarray(depth_img_array)

    return depth_img


def outlier_rejection(matches, kp_left, kp_right):
    """
    Given a set of matches, removes the outlier and returns inlier matches
    
    Args:
        matches: BFMatcher object containing the matches
        kp_left: detected keypoints of the left image
        kp_right: detected keypoints on the right image

    Returns
        inlier matches
    """
    # store matched points
    src_pts = np.float32([ kp_left[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp_right[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    # calculate homography
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 2.0)
    matchesMask = mask.ravel().tolist()
    
    # select inlier matches
    inlier_matches = []
    for i in range(len(matchesMask)):
        if matchesMask[i] == 1:
            inlier_matches.append(matches[i])

    return inlier_matches

def depth_from_stereo(left_image_dir, right_image_dir, sample_list, calib_dir, depth_img_dir):
    """
    Depth calculation using left and right images from a stereo camera
    
    Args:
        left_image_dir  : directory containing the left images of the stereo camera
        right_image_dir : directory containing the right images of the stereo camera
        sample_list     : list of image names, without extension
        calib_dir       : directory containing the calibration data
        depth_img_dir   : directory to save predicted depth images

    Returns:
        Nonex
    """
    sift    = cv.xfeatures2d.SIFT_create()

    for sample_name in sample_list:
        calib_filename = calib_dir + '/' + sample_name + '.txt'
        frame_calib    = read_frame_calib(calib_filename)
        left_cam_mat, right_cam_mat = frame_calib.p2, frame_calib.p3

        # stereo calibration
        stereo_calib   = get_stereo_calibration(left_cam_mat, right_cam_mat)
        
        # get baseline and focal length
        B, f = stereo_calib.baseline, stereo_calib.f

        # left and right image paths
        left_image_path = left_image_dir +'/' + sample_name + '.png'
        right_image_path = right_image_dir +'/' + sample_name + '.png'

        # read images
        img_left_orig  = cv.imread(left_image_path)
        img_left_gray  = cv.cvtColor(img_left_orig, cv.COLOR_BGR2GRAY)
        img_right_orig = cv.imread(right_image_path)
        img_right_gray = cv.cvtColor(img_right_orig, cv.COLOR_BGR2GRAY)

        # detect features and their descriptors
        kp_left, des_left   = sift.detectAndCompute(img_left_gray, None)
        kp_right, des_right = sift.detectAndCompute(img_right_gray, None)

        # create BFMatcher object
        bf = cv.BFMatcher(crossCheck=True)

        # match descriptors
        matches = bf.match(des_left, des_right)

        # outlier rejection using RANSAC
        print(f'Total number of matches before outlier rejection: {len(matches)} for image {sample_name}')
        inlier_matches = outlier_rejection(matches, kp_left, kp_right)
        print(f'Total number of matches after outlier rejection: {len(inlier_matches)} for image {sample_name}')

        # variable to store
        pixel_u_list = []     # x pixel on left image
        pixel_v_list = []     # y pixel on left image
        disparity_list = []
        depth_list = []

        for i, match in enumerate(inlier_matches):
            l_idx, r_idx = match.queryIdx, match.trainIdx
            u_l, v_l = kp_left[l_idx].pt[0], kp_left[l_idx].pt[1]
            u_r, v_r = kp_right[r_idx].pt[0], kp_right[r_idx].pt[1]
            disparity = int(abs(u_r - u_l))
            depth     = B*f/disparity

            # variables to be saved
            if depth <= 80:                
                pixel_u_list.append(int(u_l))
                pixel_v_list.append(int(v_l))
                disparity_list.append(disparity)
                depth_list.append(depth)            

        print(f'Total number of points with depth values: {len(depth_list)} for image {sample_name}')
        depth_image = save_depth_image(np.shape(img_left_gray), pixel_u_list, pixel_v_list, depth_list)
        depth_image.save(depth_img_dir + '/' + sample_name + '.png')

        # output
        for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
            line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
            output_file.write(line + '\n')

        # draw matches and save image
        matched_img = cv.drawMatches(img_left_orig, kp_left, 
            img_right_orig, kp_right, inlier_matches, 
            None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img_save_path = os.path.abspath('./outlier_removal_matches/matched_keypoints_outlier_r_')
        cv.imwrite(img_save_path + sample_name + '.png', matched_img)

    output_file.close()
## Main
# for sample_name in sample_list:
#     left_image_path = left_image_dir +'/' + sample_name + '.png'
#     right_image_path = right_image_dir +'/' + sample_name + '.png'

#     img_left = cv.imread(left_image_path, 0)
#     img_right = cv.imread(right_image_path, 0)

#     # TODO: Initialize a feature detector
#     sift = cv.SIFT_create()
#     kp, des = sift.detectAndCompute(img_left, None)

#     # TODO: Perform feature matching

#     # TODO: Perform outlier rejection

#     # Read calibration
#     frame_calib = None
#     stereo_calib = None

#     # Find disparity and depth
#     pixel_u_list = [] # x pixel on left image
#     pixel_v_list = [] # y pixel on left image
#     disparity_list = []
#     depth_list = []
#     for i, match in enumerate(matches):
#       	pass

#     # Output
#     for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
#         line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
#         output_file.write(line + '\n')

#     # Draw matches
#     img = cv.drawMatches(img_left, kp_left, img_right, kp_right, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     plt.imshow(img)
#     plt.show()

# output_file.close()

if __name__=='__main__':
    # feature_detection(left_image_dir_train, right_image_dir_train, sample_list_train)
    # feature_detection(left_image_dir_test, right_image_dir_test, sample_list_test)
    # feature_matching(left_image_dir_train, right_image_dir_train, sample_list_train)
    # feature_matching(left_image_dir_test, right_image_dir_test, sample_list_test)
    # depth_from_stereo(left_image_dir_train, right_image_dir_train, sample_list_train, calib_dir_train, pred_depth_map_train)
    depth_from_stereo(left_image_dir_test, right_image_dir_test, sample_list_test, calib_dir_test, pred_depth_map_test)
