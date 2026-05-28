import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lilq_shared.plotting import generate_convergence_plots

def find_results_file(filename):
    # Check current directory
    if Path(filename).exists():
        return Path(filename)
    
    # Check default fourier directory
    default_dir = Path(__file__).parent / "bratu_experiments_fourier" / filename
    if default_dir.exists():
        return default_dir

    # Recursively search subdirectories
    for p in Path(__file__).parent.glob(f"**/{filename}"):
        if "__pycache__" not in str(p) and ".git" not in str(p):
            return p
            
    return None

def main():
    # Find results file dynamically
    results_path = find_results_file('bratu_master_results.json')
    if results_path is not None:
        print(f"Plotting Bratu convergence from: {results_path}")
        generate_convergence_plots(
            results_file=str(results_path),
            output_prefix='Bratu',
            data_dir=str(Path(__file__).parent)
        )
    else:
        print("No master results file found for Bratu plotting.")

if __name__ == '__main__':
    main()