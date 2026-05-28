import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def style_axis(ax, grid=False):
    """Apply consistent styling to a single axis."""
    if grid:
        ax.grid(True, which='major', linewidth=0.4, alpha=0.4, color='0.7')
    else:
        ax.grid(False)
        
    # Apply direction styling strictly to major ticks
    ax.tick_params(which='major', direction='out')
    
    # Explicitly turn off minor ticks to enforce the intent of rcParams
    ax.minorticks_off()

def generate_polished_combined_plot(data_dir="."):
    # Set the overall page width and height
    FULL_WIDTH = 6.5
    HEIGHT = FULL_WIDTH * 0.3
    FONT_SIZE = 8

    # CMAME / Elsevier 3p Times Publication-ready rcParams
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
        'legend.title_fontsize': FONT_SIZE,
        
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        'xtick.major.width': 0.3,
        'ytick.major.width': 0.3,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
        
        'axes.linewidth': 0.3,
        'lines.linewidth': 1.0,
        'lines.markersize': 2, # Slightly smaller than 5 to prevent crowding
        'lines.markerfacecolor': 'none',
        
        'legend.frameon': False,  # Kept False for a cleaner centralized legend
        
        # PDF backend — embed fonts properly for publication
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    Ns = [5, 10, 15]
    
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, HEIGHT), dpi=300)
    
    # We will store the legend handles and labels from the first plot
    handles, labels = [], []

    for i, N in enumerate(Ns):
        file_path = Path(data_dir) / 'bratu_experiments_fourier' / f'theorem2_bratu_N{N}.json'
        if not file_path.exists():
            file_path = Path(data_dir) / f'theorem2_bratu_N{N}.json'
        if not file_path.exists():
            file_path = Path(__file__).parent.parent / 'bratu_experiments_fourier' / f'theorem2_bratu_N{N}.json'

        if not file_path.exists():
            print(f"Warning: {file_path} not found. Skipping plot {i+1}.")
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        tracker = data['tracker']
        nonlin = np.array(tracker['nonlinear_res'])
        lin = np.array(tracker['linear_res'])
        thm2_lower = np.array(tracker['thm2_lower'])
        thm2_upper = np.array(tracker['thm2_upper'])

        K = len(lin)
        iterates = np.arange(K + 1)
        band_x = np.arange(1, K + 1)

        eps_L = np.min(lin)
        eps_U = np.max(lin)

        plot_floor = 0.1 * eps_L
        lower_clip = np.maximum(thm2_lower, plot_floor)

        ax = axes[i]

        # Plot data
        ax.fill_between(band_x, lower_clip, thm2_upper,
                        alpha=0.25, color='#2A9D8F', label='Computed band', zorder=2, linewidth=0.2)

        ax.semilogy(iterates, nonlin, 'o-', color='#D71C2C',
                    label='Nonlinear MSE', lw=0.6, ms=1.8, markeredgewidth=0.9, zorder=5)

        ax.semilogy(band_x, lin, 's-', color="#070572",
                    label='Linear MSE', lw=0.5, ms=1.5, markeredgewidth=0.75, alpha=0.85, zorder=4)

        ax.axhline(y=eps_L, color="#1D7B41", linestyle=':', lw=0.5, 
                   label=r'$\varepsilon_L$', zorder=1)
        ax.axhline(y=eps_U, color='#000000', linestyle=':', lw=0.5,
                   label=r'$\varepsilon_U$', zorder=1)

        ax.set_xlabel('Iterations')
        ax.set_xlim(0, K)
        
        if i == 0:
            ax.set_ylabel('MSE')

        ax.set_xticks(range(0, len(iterates) + 2, 5))
        
        # Place the network size text inside the plot body (upper right corner)
        ax.text(0.95, 0.93, f'P = {N**2}', transform=ax.transAxes, 
                ha='right', va='top', fontsize=FONT_SIZE, zorder=6)
        
        # Apply the CMAME strict axis styling and custom grid
        style_axis(ax, grid=True)

        # Grab the legend handles from the first valid plot
        if not handles:
            handles, labels = ax.get_legend_handles_labels()

    # Adjust layout to make room for the shared legend at the top
    plt.tight_layout()
    fig.subplots_adjust(top=0.85) 

    # Place a single shared legend at the top center of the entire figure
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01),
               ncol=5, columnspacing=1.5, handletextpad=0.5)

    out_pdf = Path(data_dir) / 'Bratu_theorem2_combined.pdf'
    out_png = Path(data_dir) / 'Bratu_theorem2_combined.png'

    # bbox_inches='tight' ensures the external legend isn't cut off when saving
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved polished plot to {out_pdf} and {out_png}")

if __name__ == '__main__':
    generate_polished_combined_plot()
