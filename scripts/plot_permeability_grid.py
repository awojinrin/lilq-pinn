"""
The permeability figure of Section 6.8 (``figures/perm_grid_shared_cbar.png``)
==============================================================================

Plots log K of the four Darcy fields (S1, S2, S3 and SPE10 layer 3) from
``data/spe10/perm_field_*.txt`` on a shared colour bar. It comes from the
pre-v2 local codebase (``SPE10/plot_permeability_grid.py``). Three changes:

- the panel titles carry the letters (a)-(d) of the manuscript's figure;
- the outputs go to ``--out`` instead of the data folder;
- a missing field file is an error, not a dummy field.

At the release (Package 2, gate G1) it reproduces the manuscript's PNG pixel
for pixel. The second figure, with one colour bar per field, is not in the
manuscript.

Usage::

    python scripts/plot_permeability_grid.py --out <folder>
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'spe10'

# =========================================================================
# DOMAIN PARAMETERS
# =========================================================================
LX = 1200.0       # Physical width [ft]
LY = 2200.0       # Physical height [ft]
NX_CELLS = 60     # Number of cells in X
NY_CELLS = 220    # Number of cells in Y
K_MIN_CLIP = 0.01 # Minimum permeability clip [mD]

# =========================================================================
# PUBLICATION AESTHETICS
# =========================================================================
def setup_publication_params():
    FONT_SIZE = 8
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'axes.linewidth': 0.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

def style_axis(ax):
    ax.grid(False)
    ax.tick_params(which='major', direction='out')
    ax.minorticks_off()

def load_permeability(filepath):
    """Load and process permeability matching PINN's approach."""
    K_raw = np.loadtxt(filepath)
    K = K_raw.reshape((NX_CELLS, NY_CELLS))
    K = np.maximum(K, K_MIN_CLIP)
    return K

def generate_permeability_grids(data_dir=DATA_DIR, out_dir="."):
    setup_publication_params()
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "perm_field_S1.txt",
        "perm_field_S2.txt",
        "perm_field_S3.txt",
        "perm_field_SPE10.txt"
    ]
    labels = ["(a) S1", "(b) S2", "(c) S3", "(d) SPE10 Layer 3"]

    K_fields = []
    global_min = float('inf')
    global_max = float('-inf')

    for f in files:
        K = load_permeability(data_dir / f)
        K_fields.append(K)
        global_min = min(global_min, K.min())
        global_max = max(global_max, K.max())

    # Scale the coordinates by 1000 for cleaner tick labels
    x_perm = np.linspace(0, LX, NX_CELLS) / 1000.0
    y_perm = np.linspace(0, LY, NY_CELLS) / 1000.0
    X_perm, Y_perm = np.meshgrid(x_perm, y_perm, indexing='ij')

    FULL_WIDTH = 6.5
    
    # =====================================================================
    # APPROACH 1: SHARED VERTICAL COLORBAR
    # =====================================================================
    fig1, axes1 = plt.subplots(1, 4, figsize=(FULL_WIDTH, FULL_WIDTH * 0.5), dpi=300)
    norm_shared = colors.LogNorm(vmin=global_min, vmax=global_max)
    
    for i, ax in enumerate(axes1):
        cf = ax.pcolormesh(X_perm, Y_perm, K_fields[i], cmap='jet', norm=norm_shared, shading='auto')
        ax.set_aspect('equal')
        ax.set_title(labels[i], pad=4)
        style_axis(ax)
        ax.set_xticks(np.linspace(0, LX/1000.0, 5))
        ax.set_xticklabels([f"{int(x)}" if x==0 else f"{x:.1f}" for x in np.linspace(0, LX/1000.0, 5)])
        ax.set_yticks(np.linspace(0, LY/1000.0, 5))
        ax.set_yticklabels([f"{int(y)}" if y==0 else f"{y:.2f}" for y in np.linspace(0, LY/1000.0, 5)])
        
        if i == 0:
            ax.set_ylabel(r'$y$ [$\times 10^3$ ft]')
        else:
            ax.set_yticklabels([])
            ax.set_yticks([])
        
        ax.set_xlabel(r'$x$ [$\times 10^3$ ft]')

    fig1.subplots_adjust(left=0.05, right=0.88, wspace=0.1)
    
    cbar_ax = fig1.add_axes([0.90, 0.2, 0.015, 0.6])
    cb = fig1.colorbar(cf, cax=cbar_ax, label=r'$K$ [mD]')
    cb.ax.tick_params(direction='out', width=0.5, length=2)
    cb.outline.set_linewidth(0.6)
    
    fig1.savefig(out_dir / "perm_grid_shared_cbar.pdf", dpi=300, bbox_inches='tight')
    fig1.savefig(out_dir / "perm_grid_shared_cbar.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # =====================================================================
    # APPROACH 2: INDIVIDUAL HORIZONTAL COLORBARS
    # =====================================================================
    fig2, axes2 = plt.subplots(1, 4, figsize=(FULL_WIDTH, FULL_WIDTH * 0.6), dpi=300)
    
    for i, ax in enumerate(axes2):
        norm_local = colors.LogNorm(vmin=K_fields[i].min(), vmax=K_fields[i].max())
        cf = ax.pcolormesh(X_perm, Y_perm, K_fields[i], cmap='jet', norm=norm_local, shading='auto')
        ax.set_aspect('equal')
        ax.set_title(labels[i], pad=4)
        style_axis(ax)
        
        if i == 0:
            ax.set_ylabel(r'$y$ [$\times 10^3$ ft]')
        else:
            ax.set_yticklabels([])
        
        ax.set_xlabel(r'$x$ [$\times 10^3$ ft]')
        
        cb = fig2.colorbar(cf, ax=ax, orientation='horizontal', fraction=0.04, pad=0.15)
        cb.ax.tick_params(direction='out', width=0.5, length=2)
        cb.outline.set_linewidth(0.6)
        cb.set_label(r'$\log_{10}(K)$ [mD]')
        
        # Fixed LogLocator logic to prevent tick overlap on horizontal colorbars
        cb.ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))

    fig2.subplots_adjust(wspace=0.15)
    fig2.savefig(out_dir / "perm_grid_individual_cbars.pdf", dpi=300, bbox_inches='tight')
    fig2.savefig(out_dir / "perm_grid_individual_cbars.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="The permeability figure of Section 6.8 (perm_grid_shared_cbar).")
    ap.add_argument('--data-dir', default=str(DATA_DIR))
    ap.add_argument('--out', default='.')
    args = ap.parse_args()
    generate_permeability_grids(args.data_dir, args.out)