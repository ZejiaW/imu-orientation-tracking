import numpy as np
import jax.numpy as jnp
import transforms3d as t3d
from quaternion_jax import qexp_jax, qlog_jax, qmul_jax, qinv_jax

def q2euler(q):
    """
    Convert a quaternion to Euler angles.
    q : a numpy array of shape (5, n), where n is the number of time steps,
        the first row is the timestamp,
        and the last four rows are the quaternions. 
    return : a numpy array of shape (4, n) where n is the number of time steps,
    """
    roll, pitch, yaw = np.zeros(q.shape[1]), np.zeros(q.shape[1]), np.zeros(q.shape[1])
    yaw[0], pitch[0], roll[0] = t3d.euler.quat2euler(q[1:, 0], axes='sxyz')
    for i in range(q.shape[1]):
        yaw[i], pitch[i], roll[i] = t3d.euler.quat2euler(q[1:, i], axes='sxyz')
    return np.vstack((q[0], roll, pitch, yaw))

def rot_integrate(imu_data):
    """
    Integrate the angular velocity to get the orientation.
    imu_data : a numpy array of shape (7, n) where n is the number of time steps.
    Each row of the IMU data is: [timestamp, AccX, AccY, AccZ, GyrX, GyrY, GyrZ]

    q_{t+1} = q_t quaternion_multiplication exp([0, w_t*dt/2])

    return: a numpy array of shape (4, n) where n is the number of time steps,
            the first row is the timestamp,
            and the last three rows are the roll pitch yaw angles. 
    """
    q = [t3d.quaternions.np.array([1, 0, 0, 0])]
    yaw, pitch, roll = np.zeros(imu_data.shape[1]), np.zeros(imu_data.shape[1]), np.zeros(imu_data.shape[1])
    yaw[0], pitch[0], roll[0] = t3d.euler.quat2euler(q[0], axes='sxyz')
    for i in range(1, imu_data.shape[1]):
        dt = imu_data[0, i] - imu_data[0, i-1]
        q_tau = t3d.quaternions.np.array([0, imu_data[4, i], imu_data[5, i], imu_data[6, i]]) * dt / 2
        q.append(t3d.quaternions.qmult(q[-1], t3d.quaternions.qexp(q_tau)))
        yaw[i], pitch[i], roll[i] = t3d.euler.quat2euler(q[-1], axes='sxyz')
    return np.vstack((imu_data[0], roll, pitch, yaw)), np.vstack((imu_data[0], np.array(q).T))

def motion_loss(q, gyro):
    """
    A function to compute the motion loss for pure rotations.
    
    c = 0.5 * sum_{t=0}^{t-1} ||2log(q_{t+1}^{-1} qmult (q_{t} qmul exp([0, tau*omega_{t}/2])))||^{2}_{2}
    
    Parameters:
      q: jnp.array of shape (5, n), where n is the number of time steps,
         the first row is the timestamp,
         and the last four rows are the quaternions.
        
      gyro: jnp.array of shape (3, n) with the angular velocity
    
    Returns:
      A scalar representing the motion loss.
    """
    ts = q[0, 1:] - q[0, :-1]
    q_t = q[1:, :-1]
    q_t_plus_1 = q[1:, 1:]
    q_omega_tau = jnp.stack([jnp.zeros_like(ts), 
                             gyro[0, :-1] * ts / 2, 
                             gyro[1, :-1] * ts / 2, 
                             gyro[2, :-1] * ts / 2], axis=0)
    q_t1_inv_q_t = qmul_jax(qinv_jax(q_t_plus_1), qmul_jax(q_t, qexp_jax(q_omega_tau)))
    q_t1_inv_q_t = jnp.round(q_t1_inv_q_t, 8) # to avoid numerical issues
    return jnp.sum(jnp.linalg.norm(2 * qlog_jax(q_t1_inv_q_t), axis=0) ** 2) / 2.0

    
