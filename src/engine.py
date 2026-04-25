import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple
import hashlib

# Constants
D_PERIOD = 8
N0 = D_PERIOD**3
J_CONST = 1.0
K_CONST = 0.6
D_CONST = 0.3

class ModelParams(NamedTuple):
    p: float
    h: float
    T: float
    h_dir: jnp.ndarray

def get_q_vectors():
    Q = 2 * jnp.pi / D_PERIOD
    return jnp.stack([
        jnp.array([Q, 0, 0]), jnp.array([0, Q, 0]), jnp.array([0, 0, Q]),
        jnp.array([Q, -Q, -Q]), jnp.array([-Q, Q, -Q]), jnp.array([-Q, -Q, Q]), jnp.array([Q, Q, Q])
    ])

def get_lattice_sites():
    x, y, z = jnp.meshgrid(jnp.arange(D_PERIOD), jnp.arange(D_PERIOD), jnp.arange(D_PERIOD), indexing='ij')
    return jnp.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)

_Q_VECS = get_q_vectors()
_SITES = get_lattice_sites()
Q_DOT_R = jnp.dot(_SITES, _Q_VECS.T)

def l_function(v):
    """Langevin function L(v) = coth(v) - 1/v"""
    return 1.0 / jnp.tanh(v) - 1.0 / v

def stable_log_sinh_v(v):
    """Numerically stable computation of ln(sinh(v)/v)"""
    # Prevent NaNs in the unused branch during autodiff overflow
    safe_v = jnp.where(v < 20.0, v, 1.0)
    return jnp.where(
        v < 20.0,
        jnp.log(jnp.sinh(safe_v) / safe_v),
        v - jnp.log(2.0) - jnp.log(v)
    )

def spins_from_params(params):
    theta, phi = params
    return jnp.stack([jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)], axis=-1)

def compute_S_Q(S_r0):
    cos_qr = jnp.cos(Q_DOT_R).T
    sin_qr = jnp.sin(Q_DOT_R).T
    return jnp.dot(cos_qr, S_r0) / jnp.sqrt(N0), jnp.dot(-sin_qr, S_r0) / jnp.sqrt(N0)

def compute_g_functional(params, model_params: ModelParams):
    theta, phi, v_raw = params

    # 1. Enforce v > 0 (1e-4 prevents float32 div-by-zero in Langevin function)
    v = jax.nn.softplus(v_raw) + 1e-4

    # 2. Thermal spin magnitude and vectors
    m = l_function(v)
    S_r0 = spins_from_params((theta, phi)) * m[:, None]

    # 3. Fourier components and intensive normalization
    S_Q_real, S_Q_imag = compute_S_Q(S_r0)
    S_Q_sq = jnp.sum(S_Q_real**2 + S_Q_imag**2, axis=-1)

    # Normalize to intensive m_eta^2 to match the Hamiltonian scaling
    m_sq = S_Q_sq / N0

    # 4. Interaction Parameters
    p = model_params.p
    J_eta = jnp.where(jnp.arange(7) < 3, J_CONST * (1-p), J_CONST * p)
    K_eta = jnp.where(jnp.arange(7) < 3, K_CONST * (1-p), K_CONST * p)
    D_eta = jnp.where(jnp.arange(7) < 3, D_CONST * (1-p), D_CONST * p)

    # 5. Hamiltonian H/N (per spin) [Properly Scaled]
    term_j = -2.0 * jnp.sum(J_eta * m_sq)
    term_k = 2.0 * jnp.sum(K_eta * (m_sq**2))

    cross_prod = jnp.cross(S_Q_real, S_Q_imag)
    q_norm = jnp.linalg.norm(_Q_VECS, axis=-1)
    q_unit = _Q_VECS / q_norm[:, None]

    # DM term intensive scaling
    term_dm = -4.0 * jnp.sum(D_eta * jnp.sum(cross_prod * q_unit, axis=-1)) / N0

    zeeman = -jnp.mean(jnp.dot(S_r0, model_params.h_dir) * model_params.h)
    H_per_spin = term_j + term_k + term_dm + zeeman

    # 6. Variational Free Energy Functional G
    beta = 1.0 / model_params.T
    # Added log(4*pi) to match the exact partition function offset
    entropy_term = jnp.mean(jnp.log(4.0 * jnp.pi) + stable_log_sinh_v(v) - v * m)

    return -beta * H_per_spin + entropy_term

def init_params(key):
    k1, k2, k3 = jax.random.split(key, 3)
    # Random spherical angles and a positive initial spin magnitude parameter
    theta = jax.random.uniform(k1, (N0,), minval=0.0, maxval=jnp.pi)
    phi = jax.random.uniform(k2, (N0,), minval=0.0, maxval=2*jnp.pi)
    v_raw = jax.random.uniform(k3, (N0,), minval=0.5, maxval=2.0)
    return theta, phi, v_raw

def optimize_g(model_params: ModelParams, seeds=64, anneal_steps=15, steps_per_T=250, prev_params=None):
    
    # --- Guard: Ensure seeds are perfectly divisible by the number of GPUs ---
    num_devices = jax.local_device_count()
    seeds = max(seeds, num_devices)
    seeds = (seeds // num_devices) * num_devices

    # --- Deterministic seed based on p, h, and T for exact reproducibility ---
    hash_str = f"{float(model_params.p):.4f}_{float(model_params.h):.6f}_{float(model_params.T):.6f}".encode()
    seed_val = int(hashlib.md5(hash_str).hexdigest(), 16) % 1000000
    keys = jax.random.split(jax.random.PRNGKey(seed_val), seeds)
    batched_params = jax.vmap(init_params)(keys)

    theta_batch, phi_batch, v_batch = batched_params
    
    # =================================================================
    # --- INJECT PREVIOUS STATE (Seed 0) ---
    if prev_params is not None:
        theta_batch = theta_batch.at[0].set(prev_params[0])
        phi_batch = phi_batch.at[0].set(prev_params[1])
        v_batch = v_batch.at[0].set(prev_params[2])

    # --- INJECT FIELD-AWARE 1Q STATE (Seed 1) ---
    q_projections = jnp.abs(jnp.dot(_Q_VECS[:3], model_params.h_dir))
    best_q_idx = int(jnp.argmax(q_projections))  # convert to Python int for static indexing
    perfect_1Q_phi = Q_DOT_R[:, best_q_idx]

    perfect_1Q_theta = jnp.full((N0,), jnp.pi / 2.0)
    perfect_1Q_v = jnp.ones((N0,)) * 1.5

    theta_batch = theta_batch.at[1].set(perfect_1Q_theta)
    phi_batch = phi_batch.at[1].set(perfect_1Q_phi)
    v_batch = v_batch.at[1].set(perfect_1Q_v)


    # --- INJECT TRUE 3Q SKYRMION STATE (Seed 2) ---
    def get_rot_mat(a, b):
        """Rodrigues' rotation formula to align vector 'a' to vector 'b'"""
        a = a / (jnp.linalg.norm(a) + 1e-8)
        b = b / (jnp.linalg.norm(b) + 1e-8)
        v = jnp.cross(a, b)
        c = jnp.dot(a, b)
        vx = jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        
        # Safe denominator to prevent NaN propagation in JAX if vectors are antiparallel
        safe_denom = jnp.where(1.0 + c > 1e-8, 1.0 + c, 1.0)
        R = jnp.eye(3) + vx + jnp.dot(vx, vx) * (1.0 / safe_denom)
        
        return jnp.where(c < -0.999, -jnp.eye(3), R)

    # 1. Construct pristine orthogonal proper screws
    ph0 = Q_DOT_R[:, 0]
    S1 = jnp.stack([jnp.zeros_like(ph0), jnp.cos(ph0), jnp.sin(ph0)], axis=-1)
    
    ph1 = Q_DOT_R[:, 1]
    S2 = jnp.stack([jnp.sin(ph1), jnp.zeros_like(ph1), jnp.cos(ph1)], axis=-1)
    
    ph2 = Q_DOT_R[:, 2]
    S3 = jnp.stack([jnp.cos(ph2), jnp.sin(ph2), jnp.zeros_like(ph2)], axis=-1)
    
    S_3Q_raw = S1 + S2 + S3
    
    # 2. The natural magnetization axis of this pristine 3Q state is [1,1,1]
    nat_axis = jnp.array([1.0, 1.0, 1.0])
    target_axis = jnp.where(jnp.linalg.norm(model_params.h_dir) > 1e-8, 
                            model_params.h_dir, nat_axis)
    
    # 3. Rotate the pristine state to perfectly align with the applied field
    R_mat = get_rot_mat(nat_axis, target_axis)
    S_3Q_rot = jnp.dot(S_3Q_raw, R_mat.T)
    
    S_3Q_norm = S_3Q_rot / (jnp.linalg.norm(S_3Q_rot, axis=-1, keepdims=True) + 1e-8)

    perfect_3Q_theta = jnp.arccos(jnp.clip(S_3Q_norm[:, 2], -1.0, 1.0))
    perfect_3Q_phi = jnp.arctan2(S_3Q_norm[:, 1], S_3Q_norm[:, 0])
    perfect_3Q_v = jnp.ones((N0,)) * 1.5

    theta_batch = theta_batch.at[2].set(perfect_3Q_theta)
    phi_batch = phi_batch.at[2].set(perfect_3Q_phi)
    v_batch = v_batch.at[2].set(perfect_3Q_v)

    # --- RE-PACK FOR PMAP ---
    batched_params = (theta_batch, phi_batch, v_batch)
    # =================================================================

    # --- THE FIX: Define the reshaping function for 2 GPUs ---
    def reshape_for_pmap(x):
        return x.reshape((num_devices, -1) + x.shape[1:])

    batched_params = jax.tree_util.tree_map(reshape_for_pmap, batched_params)

    # 3. Setup optimizer
    total_steps = (anneal_steps * steps_per_T) + 1500
    schedule = optax.exponential_decay(init_value=0.05, transition_steps=total_steps, decay_rate=0.1)
    optimizer = optax.adam(learning_rate=schedule)

    # Initialize opt_state matching the [num_devices, seeds_per_device, ...] shape
    opt_state = jax.pmap(optimizer.init)(batched_params)

    def single_loss_fn(params, current_T):
        temp_params = ModelParams(
            p=model_params.p,
            h=model_params.h,
            T=current_T,
            h_dir=model_params.h_dir
        )
        return -compute_g_functional(params, temp_params)

    # vmap the gradient over the 32 seeds on a single device
    vmap_grad_fn = jax.vmap(jax.grad(single_loss_fn, argnums=0), in_axes=(0, None))

    # 4. Define the raw step function first (NO @jax.pmap decorator here)
    def raw_step(params, opt_state, current_T):
        grads = vmap_grad_fn(params, current_T)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # Explicitly wrap the function in jax.pmap
    pmap_step = jax.pmap(raw_step, in_axes=(0, 0, None))

    # 5. Simulated Annealing Schedule
    T_start = jnp.maximum(1.0, model_params.T + 0.5)
    T_schedule = jnp.linspace(T_start, model_params.T, anneal_steps)

    # Compile the optimizer state reset once
    pmap_init = jax.pmap(optimizer.init)

    for T_val in T_schedule:
        # Reset optimizer momentum cleanly using the compiled function
        opt_state = pmap_init(batched_params)
        for _ in range(steps_per_T):
            batched_params, opt_state = pmap_step(batched_params, opt_state, T_val)

    # Reset one last time before the final deep optimization
    opt_state = pmap_init(batched_params)
    for _ in range(1500):
        batched_params, opt_state = pmap_step(batched_params, opt_state, model_params.T)

    # 6. Re-flatten the parameters back to [seeds, ...] to evaluate the winner
    def flatten_pmap(x):
        return x.reshape((seeds,) + x.shape[2:])

    flat_params = jax.tree_util.tree_map(flatten_pmap, batched_params)

    # Evaluate final Free Energy across all flattened seeds
    flat_g = jax.vmap(lambda p: compute_g_functional(p, model_params))(flat_params)

    best_idx = jnp.argmax(flat_g)
    best_g = flat_g[best_idx]
    best_params = jax.tree_util.tree_map(lambda x: x[best_idx], flat_params)

    return best_params, best_g
