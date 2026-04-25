import jax
import jax.numpy as jnp
from src.engine import compute_S_Q, spins_from_params, D_PERIOD, N0

def compute_order_parameters(params):
    theta, phi, v_raw = params
    
    # 1. Recover the thermal magnitude exactly as done in engine.py
    v = jax.nn.softplus(v_raw) + 1e-4
    m_mag = 1.0 / jnp.tanh(v) - 1.0 / v  # Langevin function L(v)
    
    # 2. Multiply the unit vectors by the thermal magnitude
    S_r0 = spins_from_params((theta, phi)) * m_mag[:, None]
    
    # 3. Compute Fourier transform as usual
    S_Q_real, S_Q_imag = compute_S_Q(S_r0)

    # N0 division inside the square root to match Eq. 21
    return jnp.sqrt(jnp.sum(S_Q_real**2 + S_Q_imag**2, axis=-1) / N0)

def compute_monopole_charge(params):
    theta, phi, _ = params
    # Monopoles use unit vectors, no thermal scaling needed here
    S_r0 = spins_from_params((theta, phi))
    S = S_r0.reshape((D_PERIOD, D_PERIOD, D_PERIOD, 3))

    # Vectorized shift operator to fetch all corners of all 512 cubes simultaneously
    def g(dx, dy, dz):
        return jnp.roll(S, shift=(-dx, -dy, -dz), axis=(0, 1, 2))

    v0=g(0,0,0); v1=g(1,0,0); v2=g(0,1,0); v3=g(1,1,0)
    v4=g(0,0,1); v5=g(1,0,1); v6=g(0,1,1); v7=g(1,1,1)

    # Vectorized Oosterom-Strackee solid angle formula
    def solid_angle_tri(s1, s2, s3):
        num = jnp.sum(s1 * jnp.cross(s2, s3, axis=-1), axis=-1)
        den = (1.0 + jnp.sum(s1*s2, axis=-1) 
                   + jnp.sum(s2*s3, axis=-1) 
                   + jnp.sum(s3*s1, axis=-1))
        return 2.0 * jnp.arctan2(num, den)

    # The verified, strictly OUTWARD CCW triangle list
    triangles = [
        (v0, v2, v3), (v0, v3, v1),   # 1. Bottom face (-z)
        (v4, v5, v7), (v4, v7, v6),   # 2. Top face (+z)
        (v0, v4, v6), (v0, v6, v2),   # 3. Left face (-x)
        (v1, v3, v7), (v1, v7, v5),   # 4. Right face (+x)
        (v0, v1, v5), (v0, v5, v4),   # 5. Front face (-y)
        (v2, v6, v7), (v2, v7, v3)    # 6. Back face (+y)
    ]

    # Sum solid angles over the 12 faces for all 512 cubes simultaneously
    cube_charges = sum(solid_angle_tri(s1, s2, s3) for s1, s2, s3 in triangles)
    
    # Absolute value on the net flux of each cube to get local monopole density
    total_charge = jnp.sum(jnp.abs(cube_charges / (4 * jnp.pi)))
    
    return jnp.round(total_charge)

def compute_scalar_chirality(params, h_dir):
    theta, phi, v_raw = params

    # 1. Recover the per-site thermal magnitude (Langevin function)
    v = jax.nn.softplus(v_raw) + 1e-4
    m_mag = 1.0 / jnp.tanh(v) - 1.0 / v

    # 2. Scale the spins by their thermal magnitude (Matches PRB Eq. 20)
    S_r0 = spins_from_params((theta, phi)) * m_mag[:, None]
    S = S_r0.reshape((D_PERIOD, D_PERIOD, D_PERIOD, 3))

    def shift(tensor, dx, dy, dz):
        return jnp.roll(tensor, shift=(-dx, -dy, -dz), axis=(0, 1, 2))

    chi_vec = jnp.zeros(3)
    perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

    for gamma, alpha, beta in perms:
        chi_gamma = 0.0
        for nu_a in [-1, 1]:
            for nu_b in [-1, 1]:
                da = [0, 0, 0]; da[alpha] = nu_a
                db = [0, 0, 0]; db[beta] = nu_b

                # Calculate the triple product on the thermally scaled spins
                cross_ab = jnp.cross(shift(S, *da), shift(S, *db), axis=-1)
                chi_gamma += nu_a * nu_b * jnp.sum(jnp.sum(S * cross_ab, axis=-1))

        chi_vec = chi_vec.at[gamma].set(chi_gamma)

    # Restored: Double jnp.where to prevent NaN propagation in XLA
    h_norm = jnp.linalg.norm(h_dir)
    safe_norm = jnp.where(h_norm > 1e-8, h_norm, 1.0)
    h_unit = jnp.where(h_norm > 1e-8, h_dir / safe_norm, jnp.array([0.0, 0.0, 1.0]))
    
    chi_val = jnp.dot(chi_vec, h_unit) / N0
    
    # The paper explicitly plots the absolute magnitude |\chi_sc|
    return jnp.abs(chi_val)

def compute_magnetization(params, h_dir):
    import jax
    import jax.numpy as jnp
    from src.engine import spins_from_params

    theta, phi, v_raw = params
    v = jax.nn.softplus(v_raw) + 1e-4
    m_mag = 1.0 / jnp.tanh(v) - 1.0 / v
    S_r0 = spins_from_params((theta, phi)) * m_mag[:, None]
    return jnp.mean(jnp.dot(S_r0, h_dir))

def compute_energy_quenched(params, p, h, T_old, T_new, h_dir):
    import jax
    import jax.numpy as jnp
    from src.engine import l_function, compute_S_Q, _Q_VECS, J_CONST, K_CONST, D_CONST, N0, spins_from_params

    theta, phi, v_raw = params
    v_old = jax.nn.softplus(v_raw) + 1e-4

    # QUENCHED APPROXIMATION: Assume the local exchange field is momentarily frozen.
    v_new = v_old * (T_old / T_new)

    m = l_function(v_new)
    S_r0 = spins_from_params((theta, phi)) * m[:, None]

    S_Q_real, S_Q_imag = compute_S_Q(S_r0)
    m_sq = jnp.sum(S_Q_real**2 + S_Q_imag**2, axis=-1) / N0

    J_eta = jnp.where(jnp.arange(7) < 3, J_CONST * (1-p), J_CONST * p)
    K_eta = jnp.where(jnp.arange(7) < 3, K_CONST * (1-p), K_CONST * p)
    D_eta = jnp.where(jnp.arange(7) < 3, D_CONST * (1-p), D_CONST * p)

    term_j = -2.0 * jnp.sum(J_eta * m_sq)
    term_k = 2.0 * jnp.sum(K_eta * (m_sq**2))

    cross_prod = jnp.cross(S_Q_real, S_Q_imag)
    q_unit = _Q_VECS / jnp.linalg.norm(_Q_VECS, axis=-1)[:, None]
    term_dm = -4.0 * jnp.sum(D_eta * jnp.sum(cross_prod * q_unit, axis=-1)) / N0
    zeeman = -jnp.mean(jnp.dot(S_r0, h_dir) * h)

    # Return pure per-spin energy, matching the main Hamiltonian exactly
    return term_j + term_k + term_dm + zeeman

def compute_energy(params, p, h, h_dir):
    import jax
    import jax.numpy as jnp
    from src.engine import l_function, compute_S_Q, _Q_VECS, J_CONST, K_CONST, D_CONST, N0, spins_from_params

    theta, phi, v_raw = params
    v = jax.nn.softplus(v_raw) + 1e-4
    m = l_function(v)
    S_r0 = spins_from_params((theta, phi)) * m[:, None]

    S_Q_real, S_Q_imag = compute_S_Q(S_r0)
    m_sq = jnp.sum(S_Q_real**2 + S_Q_imag**2, axis=-1) / N0

    J_eta = jnp.where(jnp.arange(7) < 3, J_CONST * (1-p), J_CONST * p)
    K_eta = jnp.where(jnp.arange(7) < 3, K_CONST * (1-p), K_CONST * p)
    D_eta = jnp.where(jnp.arange(7) < 3, D_CONST * (1-p), D_CONST * p)

    term_j = -2.0 * jnp.sum(J_eta * m_sq)
    term_k = 2.0 * jnp.sum(K_eta * (m_sq**2))

    cross_prod = jnp.cross(S_Q_real, S_Q_imag)
    q_unit = _Q_VECS / jnp.linalg.norm(_Q_VECS, axis=-1)[:, None]
    term_dm = -4.0 * jnp.sum(D_eta * jnp.sum(cross_prod * q_unit, axis=-1)) / N0
    zeeman = -jnp.mean(jnp.dot(S_r0, h_dir) * h)

    return term_j + term_k + term_dm + zeeman
