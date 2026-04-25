import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
from src.engine import optimize_g, ModelParams, D_PERIOD, N0
from src.analysis import compute_order_parameters, compute_monopole_charge, compute_scalar_chirality, compute_magnetization, compute_energy_quenched
import argparse
import optax
from src.engine import compute_g_functional

print(f"JAX sees {jax.device_count()} global devices and {jax.local_device_count()} local GPUs.")

def run_p_sweep(T=0.1, h=0.0, h_dir=jnp.array([1.0, 0.0, 0.0]), p_range=(0, 1), num_points=21):
    """Sweep mixing ratio p. Defaults to 0-1 range."""
    ps = np.linspace(p_range[0], p_range[1], num_points)
    results = []

    print(f"Starting p-sweep at T={T}, h={h} (range {p_range})...")
    for p in ps:
        model_params = ModelParams(p=p, h=h, T=T, h_dir=h_dir)
        best_params, best_g = optimize_g(model_params, seeds=10)

        # Calculate observables
        m_eta = compute_order_parameters(best_params)
        nm = compute_monopole_charge(best_params)
        chi = compute_scalar_chirality(best_params, h_dir)

        res = {
            "p": p,
            "G": float(best_g),
            "Nm": float(nm),
            "chi": float(chi),
        }
        # Add order parameters
        for i in range(7):
            res[f"m{i+1}"] = float(m_eta[i])

        results.append(res)
        print(f"p={p:.3f} | G={best_g:.4f} | Nm={nm:.1f} | m1={m_eta[0]:.4f}")

    return pd.DataFrame(results)

def run_h_sweep(p=0.5, T=0.1, h_max=2.0, h_dir=jnp.array([1.0, 0.0, 0.0])):
    """Sweep magnetic field h from 0 to h_max."""
    hs = np.linspace(0, h_max, 21)
    results = []

    print(f"Starting h-sweep at p={p}, T={T}...")
    for h in hs:
        model_params = ModelParams(p=p, h=h, T=T, h_dir=h_dir)
        best_params, best_g = optimize_g(model_params, seeds=10)

        m_eta = compute_order_parameters(best_params)
        nm = compute_monopole_charge(best_params)
        chi = compute_scalar_chirality(best_params, h_dir)

        res = {
            "h": h,
            "G": float(best_g),
            "Nm": float(nm),
            "chi": float(chi),
        }
        for i in range(7):
            res[f"m{i+1}"] = float(m_eta[i])

        results.append(res)
        print(f"h={h:.2f} | G={best_g:.4f} | Nm={nm:.1f} | m1={m_eta[0]:.4f}")

    return pd.DataFrame(results)

def run_ht_grid_sweep(p_val, h_max=1.5, T_max=0.6, h_step=0.05, T_step=0.02, h_dir=jnp.array([1.0, 0.0, 0.0])):
    hs = np.arange(0.0, h_max + h_step/2, h_step)
    Ts = np.arange(0.01, T_max + T_step/2, T_step)
    
    results = []
    total_runs = len(hs) * len(Ts)
    print(f"Starting 2D h-T sweep for p={p_val}. Total grid points: {total_runs}")
    
    run_count = 0
    
    # SWAPPED LOOPS: Hold T constant, slowly ramp h outward
    for T in Ts:
        prev_params = None  # Reset memory for each new temperature curve
        for h in hs:
            run_count += 1
            model_params = ModelParams(p=p_val, h=float(h), T=float(T), h_dir=h_dir)
            
            # Pass the previous state into the optimizer
            best_params, best_g = optimize_g(model_params, seeds=64, prev_params=prev_params)
            
            # Save the winning parameters to inject into the next slightly higher h
            prev_params = best_params 

            m_eta = compute_order_parameters(best_params)
            nm = compute_monopole_charge(best_params)
            chi = compute_scalar_chirality(best_params, h_dir)

            res = {
                "h": float(h), "T": float(T),
                "G": float(best_g), "Nm": float(nm), "chi": float(chi)
            }
            for i in range(7):
                res[f"m{i+1}"] = float(m_eta[i])
                
            results.append(res)
            print(f"[{run_count}/{total_runs}] h={h:.2f}, T={T:.2f} | Nm={nm:02.0f} | m1={m_eta[0]:.4f}")

    return pd.DataFrame(results)

def run_ph_grid_sweep(T_val=0.01, p_step=0.05, h_step=0.1, h_max=3.0, h_dir=jnp.array([1.0, 0.0, 0.0]), out_file="results/ph_grid_checkpoint.csv"):
    """
    Sweeps mixing ratio p and magnetic field h to reproduce PRB Figure 4.
    Runs at a very low T to simulate the ground state.
    """
    # --- Guard: Normalize field direction ---
    h_dir = h_dir / jnp.linalg.norm(h_dir)
    
    # --- Guard: Ensure output directory exists before writing ---
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    
    ps = np.arange(0.0, 1.0 + p_step/2, p_step)
    hs = np.arange(0.0, h_max + h_step/2, h_step)
    
    results = []
    total_runs = len(ps) * len(hs)
    print(f"Starting 2D p-h sweep at T={T_val}. Total grid points: {total_runs}")
    
    run_count = 0
    
    for p in ps:
        prev_params = None  # Reset memory for each new p-column
        for h in hs:
            run_count += 1
            model_params = ModelParams(p=float(p), h=float(h), T=float(T_val), h_dir=h_dir)
            
            # Pass the previous state into the optimizer
            best_params, best_g = optimize_g(model_params, seeds=64, prev_params=prev_params)
            prev_params = best_params 

            m_eta = compute_order_parameters(best_params)
            nm = compute_monopole_charge(best_params)
            chi = compute_scalar_chirality(best_params, h_dir)

            res = {
                "p": float(p), "h": float(h),
                "G": float(best_g), "Nm": float(nm), "chi": float(chi)
            }
            for i in range(7):
                res[f"m{i+1}"] = float(m_eta[i])
                
            results.append(res)
            print(f"[{run_count}/{total_runs}] p={p:.2f}, h={h:.2f} | Nm={nm:02.0f} | m1={m_eta[0]:.4f}")

            # --- Safety: Incremental Checkpoint ---
            if run_count % 10 == 0:
                pd.DataFrame(results).to_csv(out_file, index=False)

    # Final save
    final_df = pd.DataFrame(results)
    final_df.to_csv(out_file, index=False)
    return final_df

def run_fig6_pipeline(out_file="results/fig6_sweep.csv", snap_dir="results/snapshots"):
    import os
    import numpy as np
    import pandas as pd
    import jax.numpy as jnp
    from src.engine import ModelParams, optimize_g
    from src.analysis import (compute_energy, compute_monopole_charge, 
                              compute_order_parameters, compute_scalar_chirality, 
                              compute_magnetization)
    
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    os.makedirs(snap_dir, exist_ok=True)
    
    h_dir = jnp.array([1.0, 1.0, 0.0]) / jnp.sqrt(2.0)
    hs = np.linspace(0.0, 0.65, 131)
    target_hs = [0.300, 0.305, 0.350, 0.400]
    
    results = []
    prev_params = None
    print("Starting Unified Fig 6 Pipeline (Airtight Dual Search)...")
    
    for h in hs:
        # --- ACTION 1: STRATEGIC RESEEDING ---
        if h < 0.001 or h > 0.615:
            prev_params = None
            
        # --- ACTION 4: DYNAMIC SEEDING ---
        current_seeds = 128 if (0.170 <= h <= 0.250) else 64

        # 1. DUAL-SEARCH BASELINE OPTIMIZATION
        mp = ModelParams(p=0.4, h=float(h), T=0.45, h_dir=h_dir)
        
        # Search A: Global Random Search
        p_glob, g_glob = optimize_g(mp, seeds=current_seeds, prev_params=None)
        
        # Search B: Local Phase Tracking
        if prev_params is not None and h > 0.001:
            p_loc, g_loc = optimize_g(mp, seeds=1, prev_params=prev_params)
            
            # --- EXPLICIT TOLERANCE LOGIC ---
            if 0.580 <= h <= 0.615:
                # High-field anomaly: Local tracker wins to preserve adiabatic orientation
                # as long as it is within 1e-4 of the global search.
                if g_loc <= g_glob + 1e-4:
                    best_params, best_g = p_loc, g_loc
                else:
                    best_params, best_g = p_glob, g_glob
            else:
                # Standard regime: Global search wins by default! Local only wins if strictly better.
                if g_loc < g_glob - 1e-6:
                    best_params, best_g = p_loc, g_loc
                else:
                    best_params, best_g = p_glob, g_glob
        else:
            best_params, best_g = p_glob, g_glob
            
        prev_params = best_params

        m_eta = compute_order_parameters(best_params)
        nm_base = compute_monopole_charge(best_params)
        chi = compute_scalar_chirality(best_params, h_dir)
        mag = compute_magnetization(best_params, h_dir)
        e_base = compute_energy(best_params, 0.4, float(h), h_dir)

        # --- ACTION 3: SHARPER SPECIFIC HEAT RESOLUTION ---
        dT = 0.002
        
        mp_up = ModelParams(p=0.4, h=float(h), T=0.45 + dT, h_dir=h_dir)
        p_up, _ = optimize_g(mp_up, seeds=1, prev_params=best_params)
        e_up = compute_energy(p_up, 0.4, float(h), h_dir)
        nm_up = compute_monopole_charge(p_up)

        mp_dn = ModelParams(p=0.4, h=float(h), T=0.45 - dT, h_dir=h_dir)
        p_dn, _ = optimize_g(mp_dn, seeds=1, prev_params=best_params)
        e_dn = compute_energy(p_dn, 0.4, float(h), h_dir)
        nm_dn = compute_monopole_charge(p_dn)

        # Strict Basin-Safe Derivative Logic
        if nm_up == nm_base and nm_dn == nm_base:
            spec_heat = (e_up - e_dn) / (2 * dT)
            jump_flag = ""
        elif nm_up != nm_base and nm_dn == nm_base:
            spec_heat = (e_base - e_dn) / dT
            jump_flag = " [FORWARD JUMP AVERTED]"
        elif nm_up == nm_base and nm_dn != nm_base:
            spec_heat = (e_up - e_base) / dT
            jump_flag = " [BACKWARD JUMP AVERTED]"
        else:
            spec_heat = float('nan')
            jump_flag = " [CRITICAL: BOTH JUMPED]"

        res = {"h": float(h), "G": float(best_g), "Nm": float(nm_base), 
               "chi": float(chi), "mag": float(mag), "C": float(spec_heat)}
        for i in range(7): res[f"m{i+1}"] = float(m_eta[i])
        results.append(res)
        
        print(f"h={h:.3f} | Nm={nm_base:02.0f} | C={spec_heat:.5f} | G={best_g:.5f}{jump_flag}")

        # Save Snapshots
        if any(np.isclose(h, t, atol=1e-4) for t in target_hs):
            theta, phi, v_raw = best_params
            np.savez(os.path.join(snap_dir, f"spins_h{h:.3f}.npz"), 
                     theta=np.array(theta), phi=np.array(phi), 
                     v_raw=np.array(v_raw), m_eta=np.array(m_eta))

    pd.DataFrame(results).to_csv(out_file, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["fig45", "fig6"], help="Which figures to compute")
    parser.add_argument("--dir", type=str, choices=["h100", "h110", "h111"], help="Direction for Fig 4/5")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.task == "fig45":
        if not args.dir:
            raise ValueError("Must provide --dir for fig45 task")

        direction_vectors = {
            "h100": jnp.array([1.0, 0.0, 0.0]),
            "h110": jnp.array([1.0, 1.0, 0.0]) / jnp.sqrt(2.0),
            "h111": jnp.array([1.0, 1.0, 1.0]) / jnp.sqrt(3.0),
        }
        h_dir = direction_vectors[args.dir]

        print(f"--- Running Fig 4 & 5 Sweeps for {args.dir} ---")
        df_grid = run_ht_grid_sweep(p_val=0.4, h_max=1.5, T_max=0.6, h_dir=h_dir)
        df_grid.to_csv(f"results/ht_grid_p0.4_{args.dir}.csv", index=False)
        run_ph_grid_sweep(h_dir=h_dir, out_file=f"results/ph_grid_{args.dir}.csv")

    elif args.task == "fig6":
        print("--- Running Fig 6 Pipeline ---")
        run_fig6_pipeline()
