from random import sample
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import csv
import os

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
sample_list_train     = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010']


## Input for test dataset
left_image_dir_test  = os.path.abspath('./test/left')
right_image_dir_test = os.path.abspath('./test/right')
calib_dir_test       = os.path.abspath('./test/calib')
sample_list_test     = ['000011', '000012', '000013', '000014','000015']

## Output
output_file = open("P3_result.txt", "a")
output_file.truncate(0)


def feature_detection(left_image_dir, right_image_dir, sample_list):
    """
    feature detection for given left and right images from a stereo camera
    
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
        img_right_orig = cv.imread(left_image_path)
        img_right_gray = cv.imread(right_image_path, 0)

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
    feature_detection(left_image_dir_test, right_image_dir_test, sample_list_test)
