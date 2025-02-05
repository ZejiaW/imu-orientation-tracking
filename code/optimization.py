from jax import vmap, grad, jit
import jax.numpy as jnp
from motion import motion_loss
from observation import observation_loss
from tqdm import tqdm

def proj2tan(grd, q):
    """
    Project the gradient to the tangent space of the quaternion manifold
    """
    return grd - q * jnp.dot(q, grd)

def optimize(q0, imu_processed, loss_fn_q, n_iter=100, lr=1e-2):
    """
    Optimize the quaternion using the projected gradient descent algorithm.

    Parameters:
        q0: jnp.array of shape (5, n), where n is the number of time steps,
            the first row is the timestamp,
            and the last four rows are the quaternions.
        imu_processed: jnp.array of shape (7, n) with the IMU data.
        loss_fn_q: a function that computes the loss given the quaternions.
        n_iter: number of iterations.
        lr: learning rate.
    
    Returns:
        q: the optimized quaternions.
        loss_list: a list of tuples with the motion and observation losses.
    """
    grad_loss_fn_q = jit(grad(loss_fn_q))
    loss_list = []
    q = q0
    progress_bar = tqdm(range(n_iter), desc="Optimizing")
    for i in progress_bar:
        grd = grad_loss_fn_q(q)
        tan_grad = vmap(proj2tan, in_axes=1, out_axes=1)(grd[1:, :], q[1:, :])
        q[1:, :] = q[1:, :] - lr * tan_grad
        q[1:, :] = q[1:, :] / jnp.linalg.norm(q[1:, :], axis=0, keepdims=True)
        mot_loss = motion_loss(q, imu_processed[4:])
        obs_loss = observation_loss(q, imu_processed[1:4])
        loss_list.append((mot_loss, obs_loss))
        progress_bar.set_postfix({"loss": mot_loss+obs_loss, "motion loss": mot_loss, "observation loss": obs_loss})
        # print(f"iter: {i}, loss: {mot_loss + obs_loss}, motion loss: {mot_loss}, observation loss: {obs_loss}")
    return q, loss_list