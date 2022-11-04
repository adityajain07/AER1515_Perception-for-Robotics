from cProfile import label
from tkinter import N
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

def load_point_cloud(path):
    # Load the point cloud data (do NOT change this function!)
    data = pd.read_csv(path, header=None)
    point_cloud = data.to_numpy()
    return point_cloud


def nearest_search(pcd_source, pcd_target):
    # TODO: Implement the nearest neighbour search
    # TODO: Compute the mean nearest euclidean distance between the source and target point cloud
    corr_target = []
    corr_source = []
    ec_dist_mean = 0
    ec_dist_list = []

    for pt_source in pcd_source:
        corr_source.append(pt_source)
        min_ec_dist = 1e6
        min_pt_target = None
        for pt_target in pcd_target:
            ec_dist = np.linalg.norm(pt_target - pt_source)
            if ec_dist < min_ec_dist:
                min_ec_dist   = ec_dist
                min_pt_target = pt_target
        ec_dist_list.append(min_ec_dist)
        corr_target.append(min_pt_target)

    ec_dist_mean = sum(ec_dist_list)/len(ec_dist_list)

    return np.array(corr_source), np.array(corr_target), ec_dist_mean


def estimate_pose(corr_source, corr_target):
    # TODO: Compute the 6D pose (4x4 transform matrix)
    # TODO: Get the 3D translation (3x1 vector)
    pose = np.identity(4)
    translation_x = 0
    translation_y = 0
    translation_z = 0

    corr_source = np.array(corr_source)
    corr_target = np.array(corr_target)
    N           = len(corr_source)

    # get centroids for source and target cluster
    p_s = np.mean(corr_source, axis=0)     # centroid of source
    p_t = np.mean(corr_target, axis=0)     # centroid of target

    # compute outer product matrix
    W_ts = 0
    for i in range(N):
        W_ts += ((corr_target[i] - p_t).reshape(3,1))@((corr_source[i] - p_s).reshape(1,3))
    W_ts = W_ts/N

    # compute SVD
    V, _, U_T = np.linalg.svd(W_ts)
    det_V = np.linalg.det(V)
    det_U = np.linalg.det(U_T.T)
    temp  = np.array([[1, 0, 0], 
                      [0, 1, 0], 
                      [0, 0, det_U*det_V]])
    C_ts  = V@temp@U_T
    r_s_ts = -(C_ts.T)@p_t + p_s

    # final translation
    r = -C_ts@r_s_ts
    translation_x = r[0]
    translation_y = r[1]
    translation_z = r[2]

    # final pose
    temp = np.hstack((C_ts, r.reshape(3,1)))
    pose = np.vstack((temp, np.array([0, 0, 0, 1])))

    return pose, translation_x, translation_y, translation_z


def plot_icp_loss(icp_loss_list, pcd_name, save_path):
    """plots icp loss"""

    print(f'Plotting ICP loss for {pcd_name}')
    plt.figure()
    plt.plot(icp_loss_list)
    plt.ylabel("ICP Loss (Mean Euclidean Distance)")
    plt.xlabel("Number of Iterations")
    # plt.title("ICP Loss Plot for " + pcd_name)
    plt.savefig(save_path + '/' 'icp_loss_' + pcd_name + '.png')


def plot_3d_translation(trans_3d_list, pcd_name, save_path):
    """plots 3d translation values"""

    trans_3d_list = np.array(trans_3d_list)
    print(f'Plotting 3d translation for {pcd_name}')
    plt.figure()
    plt.plot(trans_3d_list[:, 0], label='Translation X')
    plt.plot(trans_3d_list[:, 1], label='Translation Y')
    plt.plot(trans_3d_list[:, 2], label='Translation Z')
    plt.ylabel("3D Translation (mm)")
    plt.xlabel("Number of Iterations")
    plt.legend()
    # plt.title("3D Translation Plot for  " + pcd_name)
    plt.savefig(save_path + '/' '3d_trans_' + pcd_name + '.png')

def icp(pcd_source, pcd_target, pcd_name, plot_save_path, iter=30):
    """
    main ICP function

    Args:
        pcd_source: source point cloud
        pcd_target: target point cloud
        pcd_name: the name of the point cloud model
        plot_save_path: directory to save the plots

    Returns:
        Pose, 4x4 transformation matrix
    """
    # TODO: Put all together, implement the ICP algorithm
    # TODO: Use your implemented functions "nearest_search" and "estimate_pose"
    # TODO: Run 30 iterations
    # TODO: Show the plot of mean euclidean distance (from function "nearest_search") for each iteration
    # TODO: Show the plot of pose translation (from function "estimate_pose") for each iteration
    pose              = np.identity(4)
    cur_source        = pcd_source
    ec_dist_mean_list = []
    trans_3d_list     = []
    
    for i in range(iter):
        corr_source, corr_target, ec_dist_mean = nearest_search(cur_source, pcd_target)        
        pose_new, translation_x, translation_y, translation_z = estimate_pose(corr_source, corr_target)
        pose = pose_new@pose

        # transform the source points according to the calculated pose
        pts              = np.vstack([np.transpose(corr_source), np.ones(len(corr_source))])
        cloud_registered = np.matmul(pose_new, pts)
        cloud_registered = np.transpose(cloud_registered[0:3, :])
        cur_source       = cloud_registered

        print(f'Mean euclidean distance for iteration {i+1} is {ec_dist_mean}')
        ec_dist_mean_list.append(ec_dist_mean)
        trans_3d_list.append([translation_x, translation_y, translation_z])

    plot_icp_loss(ec_dist_mean_list, pcd_name, plot_save_path)
    plot_3d_translation(trans_3d_list, pcd_name, plot_save_path)

    return pose




def main():
    # Dataset and ground truth poses
    #########################################################################################
    # Training and test data (3 pairs in total)
    train_file = ['bunny', 'dragon']
    test_file = ['armadillo']
    plot_save_path = os.path.abspath('./plots')

    # Ground truth pose (from training data only, used for validating your implementation)
    GT_poses = []
    gt_pose = [0.8738,-0.1128,-0.4731,24.7571,
            0.1099,0.9934,-0.0339,4.5644,
            0.4738,-0.0224,0.8804,10.8654,
            0.0,0.0,0.0,1.0]
    gt_pose = np.array(gt_pose).reshape([4,4])
    GT_poses.append(gt_pose)
    gt_pose = [0.7095,-0.3180,0.6289,46.3636,
               0.3194,0.9406,0.1153,3.3165,
               -0.6282,0.1191,0.7689,-6.4642,
               0.0,0.0,0.0,1.0]
    gt_pose = np.array(gt_pose).reshape([4,4])
    GT_poses.append(gt_pose)
    #########################################################################################



    # Training (validate your algorithm)
    ##########################################################################################################
    for i in range(2):
        # Load data
        print('Plotting for: ', train_file[i])
        path_source = './training/' + train_file[i] + '_source.csv'
        path_target = './training/' + train_file[i] + '_target.csv'
        pcd_source = load_point_cloud(path_source)
        pcd_target = load_point_cloud(path_target)
        gt_pose_i = GT_poses[i]

        # Visualize the point clouds before the registration
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(pcd_source[:,0], pcd_source[:,1], pcd_source[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Source Point Cloud" , "Target Point Cloud"])
        ax.set_title('Point Clouds Before Registration')
        plt.show()


        # TODO: Use your implemented ICP algorithm to get the estimated 6D pose (from source to target point cloud)
        pose = icp(pcd_source, pcd_target, train_file[i], plot_save_path)

        # Transform the point cloud
        # TODO: Replace the ground truth pose with your computed pose and transform the source point cloud
        pts = np.vstack([np.transpose(pcd_source), np.ones(len(pcd_source))])
        # cloud_registered = np.matmul(gt_pose_i, pts)
        cloud_registered = np.matmul(pose, pts)
        cloud_registered = np.transpose(cloud_registered[0:3, :])

        # TODO: Evaluate the rotation and translation error of your estimated 6D pose with the ground truth pose
        print(f'6D pose error between ground truth and estimation for {train_file[i]} is {gt_pose_i-pose}')

        # Visualize the point clouds after the registration
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(cloud_registered[:,0], cloud_registered[:,1], cloud_registered[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Transformed Source Point Cloud", "Target Point Cloud"])
        ax.set_title('Point Clouds After Registration')
        plt.show()

    ##########################################################################################################



    # Test
    ####################################################################################
    for i in range(1):
        # Load data
        path_source = './test/' + test_file[i] + '_source.csv'
        path_target = './test/' + test_file[i] + '_target.csv'
        pcd_source = load_point_cloud(path_source)
        pcd_target = load_point_cloud(path_target)

        # Visualize the point clouds before the registration
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(pcd_source[:,0], pcd_source[:,1], pcd_source[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Source Point Cloud" , "Target Point Cloud"])
        ax.set_title('Point Clouds Before Registration')
        plt.show()

        # TODO: Use your implemented ICP algorithm to get the estimated 6D pose (from source to target point cloud)
        pose = icp(pcd_source, pcd_target, test_file[i], plot_save_path)

        # TODO: Show your outputs in the report
        # TODO: 1. Show your estimated 6D pose (4x4 transformation matrix)
        print(f'The estimated 6D pose for the test point cloud is {pose}')
        # TODO: 2. Visualize the registered point cloud and the target point cloud

        # Transform the point cloud
        pts = np.vstack([np.transpose(pcd_source), np.ones(len(pcd_source))])
        cloud_registered = np.matmul(pose, pts)
        cloud_registered = np.transpose(cloud_registered[0:3, :])

        # Visualize the point clouds after the registration
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(cloud_registered[:,0], cloud_registered[:,1], cloud_registered[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Transformed Source Point Cloud", "Target Point Cloud"])
        ax.set_title('Point Clouds After Registration')
        plt.show()
        plt.pause(1)
        plt.close()


if __name__ == '__main__':
    main()