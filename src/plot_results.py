import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import warnings
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from itertools import combinations, product

warnings.filterwarnings("ignore") # Suppress contour warnings

# Set APS-style publication plot parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "mathtext.fontset": "stix",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 1.0,
})

D_PERIOD = 8  # Required for Fig 6 real-space reconstructions

# ==========================================
# Phase Classification & Colormaps
# ==========================================
def classify_magnetic_phase(row, threshold=0.03): 
    m_prim = np.array([row['m1'], row['m2'], row['m3']])
    m_sec = np.array([row['m4'], row['m5'], row['m6'], row['m7']])
    
    n_p = np.sum(m_prim > threshold)
    n_s = np.sum(m_sec > threshold)
    
    if n_p == 0 and n_s == 0: return 0    # PM/FFM
    elif n_p == 1 and n_s == 0: return 1  # 1Q
    elif n_p == 2 and n_s == 0: return 2  # 2Q
    elif n_p == 3: return 3               # 3Q
    elif n_p == 0 and n_s >= 3: return 4  # 4Q
    elif n_p == 1 and n_s >= 3: return 5  # 5Q
    elif n_p == 0 and n_s == 1: return 6  # 1Q'
    elif n_p == 0 and n_s == 2: return 7  # 2Q'
    elif n_p == 1 and n_s == 2: return 8  # 3Q'
    else: return 0

# Standardized PRB Colors
phase_colors = [
    '#ffffff', '#5c8ab9', '#5c995c', '#e6dda6', '#b58b6b', 
    '#9b8db3', '#4a7298', '#4a7b4a', '#b8b085'  
]
cmap_phase = mcolors.ListedColormap(phase_colors)
bounds_phase = np.arange(-0.5, 9.5, 1)
norm_phase = mcolors.BoundaryNorm(bounds_phase, cmap_phase.N)

# Custom Chirality Colormap
chi_cmap = mcolors.LinearSegmentedColormap.from_list(
    "chi_paper",
    [(0.00, "#ffffff"), (0.20, "#a5d0f0"), (0.40, "#4a94d0"), 
     (0.60, "#1c4ba0"), (0.80, "#190c65"), (1.00, "#0a0026")]
)


# ==========================================
# Helpers for Figure 6 (Real Space Rendering)
# ==========================================
def reconstruct_spins(theta, phi):
    S = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    return S.reshape((D_PERIOD, D_PERIOD, D_PERIOD, 3))

def get_monopoles_np(S):
    def g(dx, dy, dz): return np.roll(S, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
    v0=g(0,0,0); v1=g(1,0,0); v2=g(0,1,0); v3=g(1,1,0)
    v4=g(0,0,1); v5=g(1,0,1); v6=g(0,1,1); v7=g(1,1,1)
    def sa(s1, s2, s3):
        num = np.sum(s1 * np.cross(s2, s3, axis=-1), axis=-1)
        den = 1.0 + np.sum(s1*s2, axis=-1) + np.sum(s2*s3, axis=-1) + np.sum(s3*s1, axis=-1)
        return 2.0 * np.arctan2(num, den)
    triangles = [(v0,v2,v3), (v0,v3,v1), (v4,v5,v7), (v4,v7,v6), (v0,v4,v6), (v0,v6,v2),
                 (v1,v3,v7), (v1,v7,v5), (v0,v1,v5), (v0,v5,v4), (v2,v6,v7), (v2,v7,v3)]
    return np.round(sum(sa(s1, s2, s3) for s1, s2, s3 in triangles) / (4 * np.pi))

def add_meta_inset(ax, m_eta, loc='lower right'):
    axins = inset_axes(ax, width="35%", height="30%", loc=loc, borderpad=1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    axins.bar(range(1, 8), m_eta, color=colors)
    axins.set_xticks([1, 4, 7])
    axins.set_ylim(0, 0.3)
    axins.tick_params(labelsize=8)
    axins.set_ylabel(r'$m_\eta$', fontsize=9, labelpad=1)


# ==========================================
# Plotting Routines
# ==========================================
def plot_fig4_replica(base_dir="results"):
    """Reproduces PRB Figure 4 (Ground State p-h phase diagram)"""
    directions = ["h100", "h110", "h111"]
    titles = [r"$\mathbf{h} \parallel [100]$", r"$\mathbf{h} \parallel [110]$", r"$\mathbf{h} \parallel [111]$"]
    letters = ["(a)", "(b)", "(c)"]
    
    fig, axes = plt.subplots(3, 1, figsize=(7, 12), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    for i, (dir_str, title) in enumerate(zip(directions, titles)):
        file_path = os.path.join(base_dir, f"ph_grid_{dir_str}.csv")
        ax = axes[i]
        
        if not os.path.exists(file_path):
            ax.text(0.5, 0.5, f"Missing: {file_path}", ha='center', va='center')
            continue

        df = pd.read_csv(file_path)
        df['Phase_ID'] = df.apply(classify_magnetic_phase, axis=1)
        pivot_phase = df.pivot(index='h', columns='p', values='Phase_ID')
        X, Y = np.meshgrid(pivot_phase.columns, pivot_phase.index)
        
        ax.pcolormesh(X, Y, pivot_phase.values, cmap=cmap_phase, norm=norm_phase, shading='nearest')
        ax.set_ylabel(r'$h$', fontsize=13)
        ax.set_ylim([0, 3.0])
        ax.set_xlim([0, 1.0])
        
        ax.text(0.02, 0.95, letters[i], transform=ax.transAxes, fontsize=14, va='top')
        ax.text(0.5, 0.95, title, transform=ax.transAxes, fontsize=14, va='top', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

        for (phase, nm), group in df.groupby(['Phase_ID', 'Nm']):
            if phase == 0 or len(group) < 15: continue
            ax.text(group['p'].mean(), group['h'].mean(), f'({int(nm)})', 
                    fontsize=10, ha='center', va='center', color='black')

    axes[2].set_xlabel(r'$p$', fontsize=13)
    plt.savefig(os.path.join(base_dir, "PRB_Fig4_Replica.png"), dpi=300, bbox_inches='tight')
    print("Figure 4 Replica Saved!")
    plt.close()


def plot_fig5_combined(base_dir="results"):
    """Reproduces PRB Figure 5 (h-T phase diagrams and chirality)"""
    directions = ["h100", "h110", "h111"]
    titles = [r"$\mathbf{h} \parallel [100]$", r"$\mathbf{h} \parallel [110]$", r"$\mathbf{h} \parallel [111]$"]
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.08, hspace=0.08)

    for i, (dir_str, title) in enumerate(zip(directions, titles)):
        file_path = os.path.join(base_dir, f"ht_grid_p0.4_{dir_str}.csv")
        ax_phase, ax_chi = axes[i, 0], axes[i, 1]
        
        if not os.path.exists(file_path):
            ax_phase.text(0.5, 0.5, "Missing Data", ha='center')
            continue
            
        df = pd.read_csv(file_path)
        df['Phase_ID'] = df.apply(classify_magnetic_phase, axis=1)
        
        pivot_phase = df.pivot(index='h', columns='T', values='Phase_ID')
        pivot_chi = df.pivot(index='h', columns='T', values='chi')
        X, Y = np.meshgrid(pivot_phase.columns, pivot_phase.index)
        
        ax_phase.pcolormesh(X, Y, pivot_phase.values, cmap=cmap_phase, norm=norm_phase, shading='nearest')
        ax_phase.set_ylabel(r'Magnetic Field $h$', fontsize=12)
        ax_phase.set_xlim([0, 0.55])
        ax_phase.set_ylim([0, 1.5])
        
        for (phase, nm), group in df.groupby(['Phase_ID', 'Nm']):
            if phase == 0 or len(group) < 10: continue 
            ax_phase.text(group['T'].mean(), group['h'].mean(), f'({int(nm)})', 
                          fontsize=11, ha='center', va='center', color='black')
            
        ax_phase.text(0.04, 0.95, title, transform=ax_phase.transAxes, fontsize=12, va='top', ha='left',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

        mesh_chi = ax_chi.pcolormesh(X, Y, np.abs(pivot_chi.values), cmap=chi_cmap, shading='nearest', vmin=0, vmax=0.25)
        ax_chi.contour(X, Y, pivot_phase.values, levels=bounds_phase, colors='gray', linestyles='dashed', linewidths=0.8, alpha=0.6)

        if i == 0:
            ax_phase.set_title('Phase Diagram', fontsize=13)
            ax_chi.set_title(r'Scalar Chirality $|\chi_{sc}|$', fontsize=13)
        if i == 2:
            ax_phase.set_xlabel('Temperature $T$', fontsize=12)
            ax_chi.set_xlabel('Temperature $T$', fontsize=12)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(mesh_chi, cax=cbar_ax)
    cbar.set_label(r'$|\chi_{sc}|$', fontsize=13)

    plt.savefig(os.path.join(base_dir, "PRB_Fig5_Replica.png"), dpi=400, bbox_inches='tight')
    print("Figure 5 Replica Saved!")
    plt.close()


def plot_fig6_replica(base_dir="results"):
    sweep_file = os.path.join(base_dir, "fig6_sweep.csv")
    if not os.path.exists(sweep_file): return

    df = pd.read_csv(sweep_file)
    fig = plt.figure(figsize=(10, 16))
    gs = GridSpec(5, 3, height_ratios=[1, 1, 1, 1.4, 1.4], hspace=0.25, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :]); ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, :], sharex=ax1)

    # --- (a) Order Parameters ---
    styles = ['-o', '-^', '-v', '--s', '--D', '--P', '--X']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for i in range(7):
        y = df[f'm{i+1}'] * (10 if i >= 3 else 1)
        label = rf'$m_{i+1}$' + (r' ($\times 10$)' if i >= 3 else '')
        ax1.plot(df['h'], y, styles[i], color=colors[i], markersize=4, fillstyle='none', label=label)
    ax1.set_ylabel(r'$m_\eta$', fontsize=14)
    ax1.legend(loc='center right', fontsize=9, ncol=2)
    ax1.text(0.01, 0.90, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), visible=False)

    # --- (b) dm/dh (Left Axis) & Specific Heat C (Right Axis) ---
    dm_dh = np.gradient(df['mag'], df['h'])
    
    ax2.plot(df['h'], dm_dh, '-^', color='purple', markersize=4, fillstyle='none')
    ax2.set_ylabel(r'$dm/dh$', color='purple', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.text(0.01, 0.90, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['h'], df['C'], '-ro', markersize=4, fillstyle='none')
    ax2_twin.set_ylabel(r'Specific Heat $C$', color='red', fontsize=14)
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2_twin.set_ylim([1.38, 1.58]) 
    plt.setp(ax2.get_xticklabels(), visible=False)

    # --- (c) Topology ---
    ax3.plot(df['h'], df['Nm'], '-ko', markersize=5, drawstyle='steps-mid')
    ax3.set_ylabel(r'$N_m$', fontsize=14)
    ax3.text(0.01, 0.90, '(c)', transform=ax3.transAxes, fontsize=14, fontweight='bold')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df['h'], df['chi'] * 1e4, '-o', color='#1f77b4', markersize=5, fillstyle='none')
    ax3_twin.set_ylabel(r'$|\chi_{sc}|$ ($\times 10^4$)', color='#1f77b4', fontsize=14)
    ax3_twin.tick_params(axis='y', labelcolor='#1f77b4')
    ax3.set_xlabel(r'Magnetic Field $h \ (\mathbf{h} \parallel [110])$', fontsize=14)
    ax3.set_xlim([0, 0.65])

    # --- (d) Panel: h=0.300 ---
    snap_path_d = os.path.join(base_dir, "snapshots", "spins_h0.300.npz")
    if os.path.exists(snap_path_d):
        d_data = np.load(snap_path_d)
        spins_d = reconstruct_spins(d_data['theta'], d_data['phi'])
        charges_d = get_monopoles_np(spins_d)
        
        ax_d1 = fig.add_subplot(gs[3, 0], projection='3d') 
        ax_d2 = fig.add_subplot(gs[3, 1], projection='3d') 
        ax_d3 = fig.add_subplot(gs[3, 2])                  
        
        ax_d1.text2D(-0.1, 1.05, "(d)  h = 0.300", transform=ax_d1.transAxes, fontsize=14, fontweight='bold')

        # d1: 3D Full Volume 
        from itertools import combinations, product
        
        # CHANGED: np.arange(4) -> np.arange(D_PERIOD) to fill the entire box
        x, y, z = np.meshgrid(np.arange(D_PERIOD), np.arange(D_PERIOD), np.arange(D_PERIOD), indexing='ij')
        
        # CHANGED: Removed the [:4] slicing so it takes the full 8x8x8 array
        S_110 = (spins_d[...,0] + spins_d[...,1]) / np.sqrt(2.0)
        colors_d = plt.cm.coolwarm(np.clip((S_110 + 1.0) / 2.0, 0, 1)).reshape(-1, 4)
        
        # Reduced alpha to 0.6 so you can still see inside the dense 512-arrow lattice!
        ax_d1.quiver(x, y, z, spins_d[...,0], spins_d[...,1], spins_d[...,2], 
                     color=colors_d, length=0.8, normalize=True, linewidth=0.5, arrow_length_ratio=0.3, alpha=0.6)
        
        r = [0, 8]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == 8:
                ax_d1.plot3D(*zip(s, e), color="k", linewidth=0.5, alpha=0.3)
                
        ax_d1.view_init(elev=20, azim=-45)
        ax_d1.set_axis_off()

        # d2: Monopole box
        px, py, pz = np.where(charges_d > 0.5); nx, ny, nz = np.where(charges_d < -0.5)
        
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == 8:
                ax_d2.plot3D(*zip(s, e), color="k", linewidth=0.5, alpha=0.3)

        ax_d2.scatter(px, py, pz, c='magenta', s=80, edgecolors='k', depthshade=False, zorder=10)
        ax_d2.scatter(nx, ny, nz, c='cyan', s=80, edgecolors='k', depthshade=False, zorder=10)
        
        for mx, my, mz in zip(np.concatenate([px, nx]), np.concatenate([py, ny]), np.concatenate([pz, nz])):
            ax_d2.plot([mx, mx], [my, my], [0, mz], 'k--', alpha=0.5, linewidth=0.8)
            ax_d2.scatter([mx], [my], [0], c='gray', s=60, alpha=0.3)

        ax_d2.set_xlim(0, 8); ax_d2.set_ylim(0, 8); ax_d2.set_zlim(0, 8)
        ax_d2.view_init(elev=20, azim=-45)
        ax_d2.set_axis_off()

        # d3: 2D Projection
        ax_d3.scatter(px, py, c='magenta', s=100, edgecolors='k', zorder=5)
        ax_d3.scatter(nx, ny, c='cyan', s=100, edgecolors='k', zorder=5)
        ax_d3.set_xlim(-0.5, 7.5); ax_d3.set_ylim(-0.5, 7.5) # Tight frame
        ax_d3.set_aspect('equal')
        ax_d3.set_xticks(np.arange(0, 9, 2)); ax_d3.set_yticks(np.arange(0, 9, 2))
        ax_d3.grid(True, linestyle='--', alpha=0.5)
        add_meta_inset(ax_d3, d_data['m_eta'], loc='lower right')

    # --- (e, f, g) 2D Slices ---
    from matplotlib.patches import Rectangle
    axes_2d = [fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 1]), fig.add_subplot(gs[4, 2])]
    targets = [0.305, 0.350, 0.400]; letters = ['(e)', '(f)', '(g)']
    
    for ax, h, letter in zip(axes_2d, targets, letters):
        snap_path = os.path.join(base_dir, "snapshots", f"spins_h{h:.3f}.npz")
        if not os.path.exists(snap_path): continue
            
        data = np.load(snap_path)
        spins = reconstruct_spins(data['theta'], data['phi'])
        S_slice = spins[:, :, 0, :]
        
        X, Y = np.meshgrid(np.arange(D_PERIOD), np.arange(D_PERIOD), indexing='ij')
        U = S_slice[..., 0]; V = S_slice[..., 1]
        
        # Color by OUT-OF-PLANE component (Sz) to match standard 2D slice literature
        C = S_slice[..., 2] 
        
        # Plot Arrows (Refined thickness and head sizes)
        ax.quiver(X, Y, U, V, C, cmap='coolwarm', pivot='mid', scale=8, width=0.012, 
                  headwidth=4, headlength=4, norm=mcolors.Normalize(vmin=-1, vmax=1), zorder=5)
        
        S2 = S_slice[..., :2]
        Sx1 = np.roll(S2, -1, axis=0); Sy1 = np.roll(S2, -1, axis=1); Sxy1 = np.roll(Sx1, -1, axis=1)
        def ang(v1, v2): return np.arctan2(v1[...,0]*v2[...,1] - v1[...,1]*v2[...,0], v1[...,0]*v2[...,0] + v1[...,1]*v2[...,1])
        vort = np.round((ang(S2, Sx1) + ang(Sx1, Sxy1) + ang(Sxy1, Sy1) + ang(Sy1, S2)) / (2*np.pi))
        
        vx, vy = np.where(vort > 0.5)
        avx, avy = np.where(vort < -0.5)
        
        # Physically accurate Rectangles instead of floating scatter points!
        # This draws exact 1x1 boxes from the bottom-left coordinate of the plaquette
        for x, y in zip(vx, vy):
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor='#f4a582', edgecolor='k', lw=1.0, zorder=0)) # Orange
        for x, y in zip(avx, avy):
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor='#92c5de', edgecolor='k', lw=1.0, zorder=0)) # Blue
        
        ax.set_xlim(-0.5, 7.5); ax.set_ylim(-0.5, 7.5) # Tight frame to the grid
        
        ax.set_title(f"h = {h:.3f}"); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
        ax.text(0.05, 0.95, letter, transform=ax.transAxes, fontsize=14, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        add_meta_inset(ax, data['m_eta'], loc='lower left')

    plt.savefig(os.path.join(base_dir, "PRB_Fig6_Replica.png"), dpi=400, bbox_inches='tight')
    print("Figure 6 Replica Saved!")
    plt.close()

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    print("Generating Figure 4...")
    plot_fig4_replica()
    
    print("Generating Figure 5...")
    plot_fig5_combined()

    print("Generating Figure 6...")
    plot_fig6_replica()
