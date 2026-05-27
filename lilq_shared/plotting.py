import json
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D
from pathlib import Path
from . import style as pub_style

def load_and_parse_data(file_path):
    """Robust JSON loading to extract N keys and target losses."""
    with open(file_path, 'r') as f:
        raw_data = json.load(f)
        
    target_losses = {}
    if 'config' in raw_data and 'target_losses' in raw_data['config']:
        for k, v in raw_data['config']['target_losses'].items():
            if k.isdigit():
                target_losses[int(k)] = v

    Ns = []
    target_data = raw_data
    if 'results' in raw_data:
        target_data = raw_data['results']
        
    digit_keys = [k for k in target_data.keys() if k.isdigit()]
    if digit_keys:
        Ns = [int(k) for k in digit_keys]
                    
    Ns.sort()
    return target_data, Ns, target_losses

def extract_loss_array(result):
    """Robustly find the loss history array across different naming conventions."""
    loss_candidates = []
    
    if 'loss_history' in result:
        lh = result['loss_history']
        if isinstance(lh, dict):
            loss_candidates.append(lh.get('loss', []))
            loss_candidates.append(lh.get('total_loss', []))
        else:
            loss_candidates.append(lh)
            
    if 'metrics' in result:
        m = result['metrics']
        loss_candidates.append(m.get('loss', []))
        loss_candidates.append(m.get('total_loss', []))
        
    for cand in loss_candidates:
        if cand is not None and len(cand) > 0:
            return cand
    return []

def generate_convergence_plots(results_file, output_prefix, target_losses=None, data_dir=".", grid_height_by_size=None):
    """
    Generate unified, publication-quality convergence plots.
    
    Parameters
    ----------
    results_file : str or Path
        Path to the master results JSON file
    output_prefix : str
        Prefix for output filenames (e.g. 'BL_gravity' or 'Bratu')
    target_losses : dict, optional
        Target threshold losses mapping N -> threshold. If None, loaded from JSON.
    data_dir : str or Path
        Directory to save the plots
    grid_height_by_size : float, optional
        Specific height adjustment for the by-size plot
    """
    FULL_WIDTH = pub_style.FULL_WIDTH
    FONT_SIZE = pub_style.FONT_SIZE
    
    METHOD_CONFIG = pub_style.METHOD_CONFIG
    METHOD_ORDER = pub_style.METHOD_ORDER
    N_COLORS = ['#0A64A4', '#FF7F0E', "#058E05", '#B91515', "#6510B4", "#643D35"]

    file_path = Path(results_file)
    if not file_path.exists():
        print(f"Error: {file_path} not found.")
        return

    data, Ns, json_target_losses = load_and_parse_data(file_path)
    if not target_losses:
        target_losses = json_target_losses
        
    if not Ns:
        print("Error: Could not automatically detect N values from JSON keys.")
        return

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PLOT 1: Grouped by Network Size (P)
    # =========================================================================
    n_panels = len(Ns)
    if n_panels <= 3:
        n_cols, n_rows = n_panels, 1
    elif n_panels == 4:
        n_cols, n_rows = 2, 2
    else:
        n_cols, n_rows = 3, math.ceil(n_panels / 3)
    
    if grid_height_by_size is None:
        grid_height = 2.2 if n_rows == 1 else (FULL_WIDTH * 0.6)
    else:
        grid_height = grid_height_by_size
    
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(FULL_WIDTH, grid_height), dpi=300)
    
    if n_panels == 1:
        axes1 = [axes1]
    else:
        axes1 = axes1.flatten()
        
    for i in range(n_panels, len(axes1)):
        axes1[i].set_visible(False)

    handles1, labels1 = [], []

    for i, N in enumerate(Ns):
        ax = axes1[i]
        N_key = str(N)
        P = N**2
        n_data = data.get(N_key, {})

        for mk in METHOD_ORDER:
            if mk not in n_data: 
                continue
            
            result = n_data[mk]
            loss_history = extract_loss_array(result)
            
            if not loss_history or len(loss_history) == 0:
                continue

            iterations = np.arange(len(loss_history))
            line, = ax.semilogy(iterations, loss_history,
                                color=METHOD_CONFIG[mk]['color'], 
                                label=METHOD_CONFIG[mk]['label'],
                                lw=0.8, zorder=4)
            
            if i == 0 and METHOD_CONFIG[mk]['label'] not in labels1:
                handles1.append(line)
                labels1.append(METHOD_CONFIG[mk]['label'])

        # Plot the target loss line
        if N in target_losses:
            t_line = ax.axhline(y=target_losses[N], color='#333333', linestyle=':', lw=0.8, zorder=1)
            if i == 0 and 'Target' not in labels1:
                handles1.append(t_line)
                labels1.append('Target')

        ax.set_title(f'P = {P}')
        
        if i % n_cols == 0:
            ax.set_ylabel('Loss')

        if i + n_cols >= n_panels:
            ax.set_xlabel('Iterations')
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', length=0)
            
        pub_style.style_axis(ax, grid=True)

    fig1.tight_layout()
    
    bottom_adjust = 0.28 if n_rows == 1 else 0.18
    fig1.subplots_adjust(bottom=bottom_adjust, hspace=0.3, wspace=0.2) 
    fig1.legend(handles1, labels1, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                ncol=len(METHOD_ORDER) + 1, columnspacing=1.5, handletextpad=0.5)

    out1_pdf = data_dir / f'{output_prefix}_convergence_by_size.pdf'
    out1_png = data_dir / f'{output_prefix}_convergence_by_size.png'
    fig1.savefig(out1_pdf, dpi=300, bbox_inches='tight')
    fig1.savefig(out1_png, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved Convergence by Size plot to {out1_pdf}")

    # =========================================================================
    # PLOT 2: Grouped by Method
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.6), dpi=300)
    axes2 = axes2.flatten()
    
    handles2, labels2 = [], []

    for i, mk in enumerate(METHOD_ORDER):
        ax = axes2[i]
        
        for j, N in enumerate(Ns):
            N_key = str(N)
            P = N**2
            n_data = data.get(N_key, {})
            
            if mk not in n_data: 
                continue
            
            result = n_data[mk]
            loss_history = extract_loss_array(result)
            
            if not loss_history or len(loss_history) == 0:
                continue

            c_idx = j % len(N_COLORS)
            iterations = np.arange(len(loss_history))
            line, = ax.semilogy(iterations, loss_history,
                                color=N_COLORS[c_idx], 
                                label=f'P = {P}',
                                lw=0.8, zorder=4)
            
            if i == 0: 
                handles2.append(line)
                labels2.append(f'P = {P}')

            if N in target_losses:
                ax.axhline(y=target_losses[N], color=N_COLORS[c_idx], linestyle=':', lw=0.8, alpha=1.0, zorder=1)

        ax.set_title(METHOD_CONFIG[mk]['label'])
        
        if i % 2 == 0:
            ax.set_ylabel('Loss')
        
        if i >= 2:
            ax.set_xlabel('Iterations')
            
        pub_style.style_axis(ax, grid=True)

    target_proxy = Line2D([0], [0], color='black', linestyle=':', lw=0.8)
    handles2.append(target_proxy)
    labels2.append('Target')

    fig2.tight_layout()
    
    fig2.subplots_adjust(bottom=0.18, hspace=0.3, wspace=0.16) 
    fig2.legend(handles2, labels2, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                ncol=min(len(Ns) + 1, 6), columnspacing=1.5, handletextpad=0.5)

    out2_pdf = data_dir / f'{output_prefix}_convergence_by_method.pdf'
    out2_png = data_dir / f'{output_prefix}_convergence_by_method.png'
    fig2.savefig(out2_pdf, dpi=300, bbox_inches='tight')
    fig2.savefig(out2_png, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved Convergence by Method plot to {out2_pdf}")
