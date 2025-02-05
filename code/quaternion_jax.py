import jax
import jax.numpy as jnp

def qexp_jax(q, eps=1e-12, threshold=1e-6):
    """
    The exponential of a batch of quaternion, implemented in JAX.

    Parameters:
        q : a jnp.array of shape (4, n) with the quaternion, where n is the number of quaternions.
        eps : a small number added inside square roots to avoid division by zero.
        threshold : if the norm of the vector part is below this, we use the limit expansion.

    Returns:
        A jnp.array of shape (4, n) with the exponential of the quaternion.
    """
    theta = jnp.sqrt(jnp.sum(q[1:]**2, axis=0) + eps)
    sin_term = jnp.where(theta < threshold, 1-theta**2/6, jnp.sin(theta) / theta) 
    q_exp = jnp.zeros_like(q)
    q_exp = q_exp.at[0].set(jnp.cos(theta))
    q_exp = q_exp.at[1:].set(sin_term * q[1:])
    return q_exp * jnp.exp(q[0])

def qlog_single(q, eps=1e-6, threshold=1e-6):
    """
    Compute the quaternion logarithm for a single quaternion.
    
    Parameters:
      q: jnp.array with shape (4,). q[0] is the scalar part, q[1:] is the vector part.
      eps: a small number added inside square roots to avoid division by zero.
      threshold: if the norm of the vector part is below this, we set the vector part to zero.
      
    Returns:
      A jnp.array of shape (4,) representing log(q).
    """
    q0 = q[0]
    v = q[1:]
    norm_q = jnp.sqrt(q0**2 + jnp.sum(v**2) + eps)
    ratio = jnp.clip(q0 / norm_q, -1.0, 1.0)
    theta = jnp.arccos(ratio)
    norm_v = jnp.sqrt(jnp.sum(v**2) + eps)
    
    def nonzero_branch(_):
        return v * (theta / norm_v)
    
    def zero_branch(_):
        return jnp.zeros_like(v)

    vector_part = jax.lax.cond(norm_v < threshold, zero_branch, nonzero_branch, operand=None)
    scalar_part = jnp.log(norm_q)
    
    return jnp.concatenate([jnp.array([scalar_part]), vector_part], axis=0)

def qlog_jax(q, eps=1e-6, threshold=1e-6):
    """
    Compute the logarithm of a batch of quaternions in JAX.
    
    Parameters:
      q: jnp.array with shape (4, n) representing n quaternions.
      eps: a small regularization constant to avoid division by zero.
      threshold: a small threshold for the vector norm to decide when to use the safe branch.
      
    Returns:
      A jnp.array of shape (4, n) containing the quaternion logarithms.
    """
    # Vectorize the single quaternion logarithm along the batch dimension (axis 1).
    return jax.vmap(lambda qq: qlog_single(qq, eps=eps, threshold=threshold), in_axes=1, out_axes=1)(q)

def qmul_jax(q1, q2):
    """
    Batched quaternion multiplication in JAX.
    
    Given two sets of quaternions q1 and q2 (each of shape (4, n)),
    the quaternion product is defined for each column as:
    
        q1 * q2 = [ s1*s2 - dot(v1, v2),
                    s1*v2 + s2*v1 + cross(v1, v2) ]
                    
    where q = [s, v] with s being the scalar part and v the 3-vector part.
    
    Parameters:
      q1: jnp.array of shape (4, n)
      q2: jnp.array of shape (4, n)
    
    Returns:
      A jnp.array of shape (4, n) containing the product quaternions.
    """
    s1 = q1[0]
    s2 = q2[0]

    v1 = q1[1:]
    v2 = q2[1:]
    
    scalar = s1 * s2 - jnp.sum(v1 * v2, axis=0)
    
    cross = jnp.cross(v1.T, v2.T).T  # shape (3, n)
    vector = s1 * v2 + s2 * v1 + cross
    return jnp.vstack([scalar, vector])

def qinv_jax(q, eps=1e-12):
    """
    Batched quaternion inverse in JAX.
    
    For a quaternion q = [s, v] (with s the scalar part and v the 3-vector part),
    the inverse is defined as:
    
          q^{-1} = q* / ||q||^{2},
    
    where the conjugate is q* = [s, -v] and ||q||^{2} = s^{2} + ||v||^{2}.
    A small epsilon is added to the denominator for numerical safety.
    
    Parameters:
      q: jnp.array of shape (4, n)
      eps: a small regularization constant to avoid division by zero.
    
    Returns:
      A jnp.array of shape (4, n) containing the inverse quaternions.
    """
    s = q[0]
    v = q[1:]
    
    norm_sq = s**2 + jnp.sum(v**2, axis=0) + eps
    q_conj = jnp.vstack([s, -v])
    return q_conj / norm_sq



if __name__ == "__main__":
    pass