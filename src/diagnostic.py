import numpy as np
import jax.numpy as jnp
from src.engine import ModelParams, optimize_g
from src.analysis import (compute_energy, compute_monopole_charge, 
                          compute_magnetization)

def raw_monopole_charge(params):
    """Bypasses the rounding to expose fractional topological defects."""
    import jax
    from src.engine import spins_from_params, D_PERIOD
    theta, phi, v_raw = params
    v = jax.nn.softplus(v_raw) + 1e-4
    m_mag = 1.0 / jnp.tanh(v) - 1.0 / v
    S_r0 = spins_from_params((theta, phi)) * m_mag[:, None]
    S = S_r0.reshape((D_PERIOD, D_PERIOD, D_PERIOD, 3))
    
    def g(dx, dy, dz): return jnp.roll(S, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
    v0=g(0,0,0); v1=g(1,0,0); v2=g(0,1,0); v3=g(1,1,0)
    v4=g(0,0,1); v5=g(1,0,1); v6=g(0,1,1); v7=g(1,1,1)

    def sa(s1, s2, s3):
        num = jnp.sum(s1 * jnp.cross(s2, s3, axis=-1), axis=-1)
        den = 1.0 + jnp.sum(s1*s2, axis=-1) + jnp.sum(s2*s3, axis=-1) + jnp.sum(s3*s1, axis=-1)
        return 2.0 * jnp.arctan2(num, den)

    triangles = [(v0,v2,v3), (v0,v3,v1), (v4,v5,v7), (v4,v7,v6), (v0,v4,v6), (v0,v6,v2),
                 (v1,v3,v7), (v1,v7,v5), (v0,v1,v5), (v0,v5,v4), (v2,v6,v7), (v2,v7,v3)]
    
    charges = sum(sa(s1, s2, s3) for s1, s2, s3 in triangles)
    return jnp.sum(charges) / (4 * jnp.pi)

def run_diagnostics():
    h_dir = jnp.array([1.0, 1.0, 0.0]) / jnp.sqrt(2.0)

    print("=== DIAGNOSTIC 1: LOW-FIELD SYMMETRY TRAP ===")
    for h_test in [0.000, 0.003, 0.005]:
        mp = ModelParams(p=0.4, h=h_test, T=0.45, h_dir=h_dir)
        best_params, best_g = optimize_g(mp, seeds=64, prev_params=None)
        nm = compute_monopole_charge(best_params)
        print(f"h={h_test:.3f} | Nm={nm:02.0f} | G={best_g:.6f}")

    print("\n=== DIAGNOSTIC 2: UNHEALED TOPOLOGY (h=0.175) ===")
    mp = ModelParams(p=0.4, h=0.175, T=0.45, h_dir=h_dir)
    best_params, best_g = optimize_g(mp, seeds=64, prev_params=None)
    raw_nm = raw_monopole_charge(best_params)
    rounded_nm = compute_monopole_charge(best_params)
    print(f"h=0.175 | Rounded Nm={rounded_nm} | RAW Nm={float(raw_nm):.5f} | G={best_g:.6f}")

    print("\n=== DIAGNOSTIC 3: HIGH-FIELD C COLLAPSE ===")
    for h_test in [0.585, 0.600, 0.620, 0.635]:
        mp = ModelParams(p=0.4, h=h_test, T=0.45, h_dir=h_dir)
        best_params, best_g = optimize_g(mp, seeds=64, prev_params=None)
        nm = compute_monopole_charge(best_params)
        e = compute_energy(best_params, 0.4, h_test, h_dir)
        mag = compute_magnetization(best_params, h_dir)
        print(f"h={h_test:.3f} | Nm={nm:02.0f} | G={best_g:.6f} | E={e:.6f} | mag={mag:.4f}")

if __name__ == "__main__":
    run_diagnostics()
