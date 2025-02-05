import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as plt


def plot_imu(imu_data, title="IMU data", path=None):
    """
    Plot the IMU data.
    IMU data should be a Numpy array of shape (7, N) where N is the number of samples.
    Each row of the IMU data is: [timestamp, AccX, AccY, AccZ, GyrX, GyrY, GyrZ]
    """
    timestamps = imu_data[0, :] - imu_data[0, 0]

    fig, axs = plt.subplots(3, 2, figsize=(18, 12), dpi=400)
    fig.suptitle(title, fontsize=20)

    # change the font size of all text in the image and the font type except the title
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

    axs[0, 0].set_title("Accelerometer")

    axs[0, 0].plot(timestamps, imu_data[1, :], label="AccX")
    # axs[0, 0].set_xlabel("Time [sec]")
    axs[0, 0].set_ylabel("Acc_X (g)")

    axs[1, 0].plot(timestamps, imu_data[2, :], label="AccY")
    # axs[1, 0].set_xlabel("Time [sec]")
    axs[1, 0].set_ylabel("Acc_Y (g)")

    axs[2, 0].plot(timestamps, imu_data[3, :], label="AccZ")
    axs[2, 0].set_xlabel("Time [sec]")
    axs[2, 0].set_ylabel("Acc_Z (g)")

    axs[0, 1].set_title("Gyroscope")

    axs[0, 1].plot(timestamps, imu_data[4, :], label="GyrX")
    # axs[0, 1].set_xlabel("Time [sec]")
    axs[0, 1].set_ylabel("Gyr_X")

    axs[1, 1].plot(timestamps, imu_data[5, :], label="GyrY")
    # axs[1, 1].set_xlabel("Time [sec]")
    axs[1, 1].set_ylabel("Gyr_Y")

    axs[2, 1].plot(timestamps, imu_data[6, :], label="GyrZ")
    axs[2, 1].set_xlabel("Time [sec]")
    axs[2, 1].set_ylabel("Gyr_Z")

    if path is not None:
        plt.savefig(path)

    plt.tight_layout()
    plt.show()

def roll_pitch_yaw_plot(estimated, ground_truth, title=None, path=None, axes=None):
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    estimated = estimated.copy()
    estimated[1:, :] = np.degrees(estimated[1:, :])
    estimated[0, :] -= estimated[0, 0]

    if ground_truth is not None:
        ground_truth = ground_truth.copy()
        ground_truth[1:, :] = np.degrees(ground_truth[1:, :])
        ground_truth[0, :] -= ground_truth[0, 0]

    if axes is None:
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    else:
        ax = axes
    ax[0].plot(estimated[0], estimated[1], label="Estimated")
    if ground_truth is not None:
        ax[0].plot(ground_truth[0], ground_truth[1], label="Ground truth")
    ax[0].set_ylabel("Roll (deg)")
    if title is not None:
        ax[0].set_title(title)
    ax[0].legend()
    ax[1].plot(estimated[0], estimated[2], label="Estimated")
    if ground_truth is not None:
        ax[1].plot(ground_truth[0], ground_truth[2], label="Ground truth")
    ax[1].set_ylabel("Pitch (deg)")
    ax[1].legend()
    ax[2].plot(estimated[0], estimated[3], label="Estimated")
    if ground_truth is not None:
        ax[2].plot(ground_truth[0], ground_truth[3], label="Ground truth")
    ax[2].set_ylabel("Yaw (deg)")
    ax[2].legend()
    ax[2].set_xlabel("Time [sec]")
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    if axes is None:
        plt.show()

def acc_plot(estimated, ground_truth, title=None, path=None, axes=None):
    estimated = estimated.copy()
    ground_truth = ground_truth.copy()
    estimated[0, :] -= estimated[0, 0]
    ground_truth[0, :] -= ground_truth[0, 0]
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    if axes is None:
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    else:
        ax = axes
    ax[0].plot(estimated[0], estimated[1], label="Estimated")
    ax[0].plot(ground_truth[0], ground_truth[1], label="Acc Readings")
    ax[0].set_ylabel("AccX (g)")
    if title is not None:
        ax[0].set_title(title)
    ax[0].legend()
    ax[1].plot(estimated[0], estimated[2], label="Estimated")
    ax[1].plot(ground_truth[0], ground_truth[2], label="Acc Readings")
    ax[1].set_ylabel("AccY (g)")
    ax[1].legend()
    ax[2].plot(estimated[0], estimated[3], label="Estimated")
    ax[2].plot(ground_truth[0], ground_truth[3], label="Acc Readings")
    ax[2].set_ylabel("AccZ (g)")
    ax[2].legend()
    ax[2].set_xlabel("Time [sec]")
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    if axes is None:
        plt.show()

def acc_rpy_plot(rpy_estimated, acc_estimated, rpy_ground_truth, acc_ground_truth, title=None, path=None):
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, ax = plt.subplots(3, 2, figsize=(10, 8))
    acc_plot(acc_estimated, acc_ground_truth, title=title+"Acceleration", axes=ax[:, 0])
    roll_pitch_yaw_plot(rpy_estimated, rpy_ground_truth, title=title+"Roll, Pitch, Yaw", axes=ax[:, 1])
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    plt.show()

def vicon_roll_pitch_yaw(vicon_data):
    """
    Extract the roll, pitch, yaw angles from the vicon data.
    vicon_data : dict with keys {'rots', 'ts'} where
                 'rots' is a numpy array of shape (3, 3, n) where n is the number of time steps,
                 'ts' is a numpy array of shape (n,) of timestamps.
    return: a numpy array of shape (4, n) where n is the number of time steps,
            the first row is the timestamp,
            and the last three rows are the roll pitch yaw angles. 
    """
    n = vicon_data['rots'].shape[2]
    yaw, pitch, roll = np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n))
    for i in range(n):
        yaw[0, i], pitch[0, i], roll[0, i] = t3d.euler.mat2euler(vicon_data['rots'][:, :, i])
    return np.vstack((vicon_data['ts'], roll, pitch, yaw))

def loss_plot(loss_list, title="", path=None):
    """
    Plot the loss over iterations.
    """
    loss_list = np.array(loss_list)
    plt.plot(loss_list[:,0], label='motion error')
    plt.plot(loss_list[:,1], label='observation error')
    plt.plot(loss_list[:,0] + loss_list[:,1], label='total error')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title + ' Loss over Iterations')
    if path is not None:
        plt.savefig(path)
    plt.show()

def plot_pc(pc, imgs, title="Point Cloud"):
    """
    Plot the point cloud in the world frame with the image pixel colors.
    This takes a huge amount of time to plot. (not suggested for large point clouds)
    """
    pc = pc.reshape(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=imgs.reshape(-1, 3)/255)
    ax.set_title(title)
    plt.show()