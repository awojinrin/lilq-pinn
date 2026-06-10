"""
Publication-Quality Plot Styling for JCP Manuscript
===================================================

Centralised matplotlib configuration matching the Elsevier 3p Times template.
Import this module at the top of every experiment runner to ensure consistent
styling across all figures.


Usage::

    from lilq import style
    # All rcParams are set on import. Use the constants below for sizing.

    fig, ax = plt.subplots(figsize=(style.FULL_WIDTH, 4.0))
    style.style_axis(ax)
"""

import matplotlib
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Page Dimensions (Elsevier 3p single-column, Times)
# ─────────────────────────────────────────────────────────────────────────────
FULL_WIDTH = 6.5      # inches — full text width
HALF_WIDTH = 3.15     # inches — half text width (for side-by-side in tabular)
FONT_SIZE = 8         # pt — matches manuscript body text

# ─────────────────────────────────────────────────────────────────────────────
# Color Palettes
# ─────────────────────────────────────────────────────────────────────────────

# Method colors: high-contrast, print-safe, colourblind-friendly
METHOD_COLORS = {
    'std_pinn': '#D62728',   # red
    'ql_pinn':  '#1F77B4',   # blue
    'nl_lil':   '#FF7F0E',   # orange
    'ql_lil':   '#2CA02C',   # green
}

# Network size colors (for by-method plots where each curve = one N)
N_COLORS_DEFAULT = {
    5:  '#1F77B4',   # blue
    8:  '#1F77B4',
    10: '#FF7F0E',   # orange
    16: '#FF7F0E',
    15: '#2CA02C',   # green
    20: '#D62728',   # red
    24: '#2CA02C',
    25: '#9467BD',   # purple
    32: '#D62728',
}

# Method display configuration
METHOD_CONFIG = {
    'std_pinn': {'label': 'NiL-N',  'color': METHOD_COLORS['std_pinn'],
                 'linestyle': '-', 'filename_prefix': 'NiLN'},
    'ql_pinn':  {'label': 'NiL-Q',  'color': METHOD_COLORS['ql_pinn'],
                 'linestyle': '-', 'filename_prefix': 'NiLQ'},
    'nl_lil':   {'label': 'LiL-N',  'color': METHOD_COLORS['nl_lil'],
                 'linestyle': '-', 'filename_prefix': 'LiLN'},
    'ql_lil':   {'label': 'LiL-Q',  'color': METHOD_COLORS['ql_lil'],
                 'linestyle': '-', 'filename_prefix': 'LiLQ'},
}

METHOD_ORDER = ['std_pinn', 'nl_lil', 'ql_pinn', 'ql_lil']

# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib rcParams
# ─────────────────────────────────────────────────────────────────────────────

_RC_PARAMS = {
    # Font — match Elsevier Times template
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'font.size':          FONT_SIZE,
    'axes.labelsize':     FONT_SIZE,
    'axes.titlesize':     FONT_SIZE,
    'xtick.labelsize':    FONT_SIZE - 1,
    'ytick.labelsize':    FONT_SIZE - 1,
    'legend.fontsize':    FONT_SIZE - 1,
    'legend.title_fontsize': FONT_SIZE - 1,

    # Tick marks — outside, no minor ticks by default
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':   4,
    'ytick.major.size':   4,
    'xtick.minor.size':   2,
    'ytick.minor.size':   2,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,

    # Axes
    'axes.linewidth':     0.6,
    'axes.grid':          False,
    'axes.spines.top':    True,
    'axes.spines.right':  True,

    # Lines
    'lines.linewidth':    1.2,
    'lines.markersize':   5,

    # Legend
    'legend.frameon':       True,
    'legend.framealpha':    1.0,
    'legend.edgecolor':     '0.8',
    'legend.fancybox':      False,
    'legend.borderpad':     0.4,
    'legend.handlelength':  1.5,

    # Figure
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.02,

    # PDF backend — embed fonts
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
}

matplotlib.rcParams.update(_RC_PARAMS)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def style_axis(ax, grid: bool = False) -> None:
    """Apply consistent styling to a single matplotlib axis."""
    if grid:
        ax.grid(True, which='major', linewidth=0.4, alpha=0.4, color='0.7')
    else:
        ax.grid(False)
    ax.tick_params(which='both', direction='out')


def get_n_color(N: int, n_colors: dict = None) -> str:
    """Get color for a given network/basis size N."""
    if n_colors is not None:
        return n_colors.get(N, '#333333')
    return N_COLORS_DEFAULT.get(N, '#333333')


def add_target_lines(ax, target_losses: dict, n_colors: dict = None) -> None:
    """Add horizontal dashed target lines for each N value."""
    for N, target in target_losses.items():
        color = get_n_color(N, n_colors)
        ax.axhline(y=target, color=color, linestyle='--', linewidth=0.7, alpha=0.6)
