import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def process_cam(camd, rot, start_time):
    """
    Process the camera data to match the timestamps with the rotation data.

    Parameters:
        camd: a dictionary with the camera data.
        rot: a dictionary with the rotation data (VICON ground truth),
            or a numpy array with the rotation data (estimated quaternions).
        start_time: the start time of the experiment (IMU sensor).

    Returns:
        camd: a dictionary with the camera data and the matched rotation data.
    """

    if isinstance(rot, dict):
        rot_ts = np.copy(rot["ts"])
        R = rot["rots"].transpose(2, 0, 1)
    elif isinstance(rot, np.ndarray):
        rot_ts = np.copy(rot[0, :][None, :])
        rot = np.copy(rot[1:, :])
        R = []
        for i in range(rot.shape[1]):
            R.append(t3d.quaternions.quat2mat(rot[:, i]))
        R = np.array(R)
    else:
        raise ValueError("Invalid pose data type")
    rot_ts = rot_ts - start_time
    camd["ts"] = camd["ts"] - start_time
    indices = np.argmin(np.abs(camd["ts"] - rot_ts.T), axis=0)
    camd["rots"] = R[indices]
    return camd

def img2world(image, fov, rot_mats, shift=np.array([0, 0, 0.1])):
    """
    Calculate the world coordinates of the image points.

    Parameters:
        image (np.ndarray): Image points in the camera frame. shape = (240, 320, 3)
        fov (float): Field of view of the camera.
        rot_mats (np.ndarray): Rotation matrices of the body frame. shape = (3, 3)
        shift (np.ndarray): Shift of the camera frame in the body frame. shape = (3,)

    Returns:
        np.ndarray: World coordinates of the image points. shape = (240, 320, 3)
    """
    h_fov, v_fov = np.radians(fov)
    img_h, img_w, _ = image.shape

    x_idx, y_idx = np.meshgrid(np.arange(img_w), np.arange(img_h))

    x_normalized = (x_idx - (img_w / 2)) / (img_w / 2)
    y_normalized = (y_idx - (img_h / 2)) / (img_h / 2)

    # lambda_ = x_normalized * (h_fov / 2)
    # phi_ = y_normalized * (v_fov / 2)
    lambda_ = np.arctan(x_normalized * np.tan(h_fov / 2))
    phi_ = np.arctan(y_normalized * np.tan(v_fov / 2))

    cart_coord = np.stack([np.cos(phi_) * np.sin(lambda_), 
                          np.sin(phi_), 
                          np.cos(phi_) * np.cos(lambda_)], axis=-1).reshape(-1, 3)
    
    cam2rob = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]])
    cart_coord = np.dot(cam2rob, cart_coord.T).T + shift[None, :]

    rotation = R.from_matrix(rot_mats)
    return rotation.apply(cart_coord).reshape(img_h, img_w, 3)

def images2pc(imgs, fov, rots, shift=np.array([0, 0, 0.1])):
    """
    Calculate the world coordinates of the image points.

    Parameters:
        imgs (np.ndarray): Image points in the camera frame. shape = (240, 320, 3, n)
        fov (float): Field of view of the camera.
        rots (np.ndarray): Rotation matrices of the body frame. shape = (n, 3, 3)
        shift (np.ndarray): Shift of the camera frame in the body frame. shape = (3,)

    Returns:
        np.ndarray: World coordinates of the image points. shape = (n, 240, 320, 3)
        np.ndarray: Image pixel colors. shape = (n, 240, 320, 3)
    """
    n = imgs.shape[-1]
    pc = []
    for i in range(n):
        pc.append(img2world(imgs[:, :, :, i], fov, rots[i, :, :], shift))
    return np.array(pc), imgs.transpose(3, 0, 1, 2)

def cart2cylinder(cart_coords):
    """
    Convert cartesian coordinates to cylindrical coordinates.

    Parameters:
        cart_coords (np.ndarray): Cartesian coordinates. shape = (n, 3)

    Returns:
        tuple of np.ndarray: Cylindrical coordinates. shapes = (n,), (n,)
    """
    phi = np.arctan2(cart_coords[:, 1], cart_coords[:, 0])
    phi = np.mod(phi, 2 * np.pi)
    x = phi / (2 * np.pi)
    y = cart_coords[:, 2] / 2 + 0.5
    return x, y

def plot_cylinder(pc, imgs, res=(3000, 1000), title="Cylindrical Projection", path=None):
    """
    Plot the cylindrical projection of the point cloud.

    Parameters:
        pc (np.ndarray): Point cloud. shape = (n, w, h, 3)
        imgs (np.ndarray): Image pixel colors. shape = (n, w, h, 3)
        res (tuple): Resolution of the cylindrical projection. Default is (3000, 1000).
        title (str): Title of the plot. Default is "Cylindrical Projection".
        path (str): Path to save the plot. Default is None.
    
    """

    x, y = cart2cylinder(pc.reshape(-1, 3))
    x = np.clip((x * res[0]).astype(int), 0, res[0] - 1)
    y = np.clip((y * res[1]).astype(int), 0, res[1] - 1)
    cylinder_image = np.zeros((res[1], res[0], 3), dtype=np.uint8)
    cylinder_image[y, x] = imgs.reshape(-1, 3)
    plt.figure(figsize=(30, 10))
    plt.imshow(cylinder_image)
    plt.title(title)
    if path is not None:
        plt.savefig(path)
    plt.show()
