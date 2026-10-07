"""Shared matplotlib style for the LiL-Q revision figures."""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

TEXTWIDTH_IN = 468.0 / 72.27   # elsarticle 3p \textwidth (from main.log) = 6.48 in

def apply():
    mpl.rcParams.update({
        'font.family': ['cmr10', 'STIXGeneral'],
        'font.serif': ['cmr10'],
        'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True,
        'axes.unicode_minus': False,
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'legend.frameon': False,
        'legend.handlelength': 2.4,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

# Okabe-Ito colour-blind-safe palette; series are also distinguished by marker.
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']
MARKERS = ['o', 's', '^', 'D', 'v', 'p']

def save(fig, stem):
    fig.savefig(f'<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts_package1/{stem}.pdf')
    fig.savefig(f'<laptop-home>/Documents/LiL-Q/Post-JCP/package2_results/G1_release/manuscript_scripts_package1/{stem}.png', dpi=200)
    plt.close(fig)
