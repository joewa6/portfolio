"""
DYNAMICS-10 Analysis Utilities

Shared utilities for generating figures across the series.
No executable code should live in the blog posts themselves.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Figure output directory
FIG_DIR = Path(__file__).parent.parent / "assets" / "img" / "blog" / "dynamics-10"


def save_figure(fig, day: str, filename: str, dpi: int = 300):
    """
    Save a matplotlib figure to the correct day subfolder.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    day : str
        Day identifier (e.g., "day01")
    filename : str
        Figure filename (e.g., "distributions.png")
    dpi : int
        Resolution for saved figure
    """
    day_dir = FIG_DIR / day
    day_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = day_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")


def set_plot_style():
    """
    Consistent plotting style across all DYNAMICS-10 figures.
    """
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (8, 5),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
    })
