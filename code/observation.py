import numpy as np
import jax.numpy as jnp
import transforms3d as t3d
from quaternion_jax import qexp_jax, qlog_jax, qmul_jax, qinv_jax

def acc_body_frame(q, init=[0, 0, 1]):
    """
    Compute the sequence of acceleration vectors in the body frame by integration, 
        for comparison with the IMU data.
    q : a numpy array of shape (5, n), where n is the number of time steps,
        the first row is the timestamp,
        and the last four rows are the quaternions.
    return : a numpy array of shape (4, n) where n is the number of time steps,
             the first row is the timestamp,
             and the last three rows are the acceleration vectors in the body frame.
    """
    acc = np.array(init)[:, None]
    for i in range(1, q.shape[1]):
        acc = np.hstack((acc, t3d.quaternions.rotate_vector(acc[:, 0], t3d.quaternions.qinverse(q[1:, i]), is_normalized=True)[:, None]))
    return np.vstack((q[0], acc))

def observation_loss(q, acc):
    """
    A function to compute the observation loss for the accelerometer.
    
    c = 0.5 * sum_{t=0}^{T} ||[0, a_t] - q_t^{-1} qmul [0, 0, 0, 1] qmul q_t||^{2}_{2}

    Parameters:
      q: jnp.array of shape (5, n), where n is the number of time steps,
         the first row is the timestamp,
         and the last four rows are the quaternions.
        
      acc: jnp.array of shape (3, n) with the acceleration vectors in the body frame.

    Returns:
        A scalar representing the observation loss.
    """
    g = jnp.array([0, 0, 0, 1]).repeat(acc.shape[1]-1).reshape(4, acc.shape[1]-1)
    obs = qmul_jax(qinv_jax(q[1:, 1:]), qmul_jax(g, q[1:, 1:]))
    return jnp.sum(jnp.linalg.norm(acc[:, 1:] - obs[1:, :], axis=0) ** 2) / 2.0
