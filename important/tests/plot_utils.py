"""
Plot utilities for quantum battery test artifacts.

Provides consistent styling and helper functions for generating
publication-quality visualization artifacts from test runs.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any


# Consistent plot style for all test artifacts
PLOT_STYLE: Dict[str, Any] = {
    'figsize': (10, 6),
    'dpi': 150,
    'title_fontsize': 16,
    'label_fontsize': 14,
    'tick_fontsize': 11,
    'legend_fontsize': 12,
    'grid_alpha': 0.3,
    'linewidth': 2,
    'save_dpi': 300,
}


def apply_plot_style() -> None:
    """Apply consistent plot style to all subsequent plots."""
    plt.rcParams.update({
        'figure.figsize': PLOT_STYLE['figsize'],
        'figure.dpi': PLOT_STYLE['dpi'],
        'axes.titlesize': PLOT_STYLE['title_fontsize'],
        'axes.labelsize': PLOT_STYLE['label_fontsize'],
        'xtick.labelsize': PLOT_STYLE['tick_fontsize'],
        'ytick.labelsize': PLOT_STYLE['tick_fontsize'],
        'legend.fontsize': PLOT_STYLE['legend_fontsize'],
        'grid.alpha': PLOT_STYLE['grid_alpha'],
        'lines.linewidth': PLOT_STYLE['linewidth'],
    })


def plot_hamiltonian_heatmap(
    H: np.ndarray,
    title: str = 'Hamiltonian',
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot a heatmap of |H| (absolute value of Hamiltonian matrix).

    Args:
        H: Hamiltonian matrix (2D array, may be complex).
        title: Plot title.
        output_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'])
    im = ax.imshow(np.abs(H), cmap='viridis', aspect='equal')
    fig.colorbar(im, ax=ax, label='$|H_{ij}|$')
    ax.set_title(f'{title}  (dim={H.shape[0]})', fontsize=PLOT_STYLE['title_fontsize'])
    ax.set_xlabel('Column', fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_ylabel('Row', fontsize=PLOT_STYLE['label_fontsize'])
    if output_path is not None:
        fig.savefig(output_path, dpi=PLOT_STYLE['save_dpi'], bbox_inches='tight')
    return fig


def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    title: str = 'Eigenvalue Spectrum',
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot eigenvalue spectrum as a stem plot.

    Args:
        eigenvalues: Sorted array of eigenvalues.
        title: Plot title.
        output_path: If provided, save figure here.

    Returns:
        Matplotlib Figure object.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'])
    indices = np.arange(len(eigenvalues))
    ax.stem(indices, eigenvalues, linefmt='C0-', markerfmt='C0o', basefmt='k-')
    ax.set_xlabel('Level index $n$', fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_ylabel('Energy $E_n$', fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_title(title, fontsize=PLOT_STYLE['title_fontsize'])
    ax.grid(True, alpha=PLOT_STYLE['grid_alpha'])
    if output_path is not None:
        fig.savefig(output_path, dpi=PLOT_STYLE['save_dpi'], bbox_inches='tight')
    return fig


def plot_state_bar(
    amplitudes: np.ndarray,
    title: str = 'State Amplitudes',
    ylabel: str = '$|\\psi_n|$',
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot state vector amplitudes as a bar chart.

    Args:
        amplitudes: 1-D array (can be complex — absolute values are plotted).
        title: Plot title.
        ylabel: Y-axis label.
        output_path: If provided, save figure here.

    Returns:
        Matplotlib Figure object.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'])
    indices = np.arange(len(amplitudes))
    ax.bar(indices, np.abs(amplitudes), color='steelblue', edgecolor='navy', alpha=0.85)
    ax.set_xlabel('Basis index', fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_ylabel(ylabel, fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_title(title, fontsize=PLOT_STYLE['title_fontsize'])
    ax.grid(True, alpha=PLOT_STYLE['grid_alpha'], axis='y')
    if output_path is not None:
        fig.savefig(output_path, dpi=PLOT_STYLE['save_dpi'], bbox_inches='tight')
    return fig


def plot_density_matrix_heatmap(
    rho: np.ndarray,
    title: str = 'Density Matrix',
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot real and imaginary parts of a density matrix side by side.

    Args:
        rho: Density matrix (2-D complex array).
        title: Plot title prefix.
        output_path: If provided, save figure here.

    Returns:
        Matplotlib Figure object.
    """
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, part, label in zip(axes, [np.real(rho), np.imag(rho)], ['Real', 'Imag']):
        im = ax.imshow(part, cmap='RdBu', aspect='equal')
        fig.colorbar(im, ax=ax)
        ax.set_title(f'{title} ({label})', fontsize=PLOT_STYLE['title_fontsize'])
        ax.set_xlabel('Column', fontsize=PLOT_STYLE['label_fontsize'])
        ax.set_ylabel('Row', fontsize=PLOT_STYLE['label_fontsize'])
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=PLOT_STYLE['save_dpi'], bbox_inches='tight')
    return fig


def plot_sparsity_pattern(
    matrix: np.ndarray,
    title: str = 'Sparsity Pattern',
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Visualise the non-zero pattern of a matrix.

    Args:
        matrix: 2-D array (dense). Non-zero entries are shown.
        title: Plot title.
        output_path: If provided, save figure here.

    Returns:
        Matplotlib Figure object.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'])
    ax.spy(matrix, markersize=2, aspect='equal')
    ax.set_title(title, fontsize=PLOT_STYLE['title_fontsize'])
    ax.set_xlabel('Column', fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_ylabel('Row', fontsize=PLOT_STYLE['label_fontsize'])
    if output_path is not None:
        fig.savefig(output_path, dpi=PLOT_STYLE['save_dpi'], bbox_inches='tight')
    return fig


def plot_energy_evolution(
    times: np.ndarray,
    energies: np.ndarray,
    title: str = 'Energy vs Time',
    ylabel: str = '$\\langle H_0 \\rangle$',
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot energy (or any scalar observable) as a function of time.

    Args:
        times: 1-D time array.
        energies: 1-D energy array (same length as *times*).
        title: Plot title.
        ylabel: Y-axis label.
        output_path: If provided, save figure here.

    Returns:
        Matplotlib Figure object.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'])
    ax.plot(times, energies, linewidth=PLOT_STYLE['linewidth'])
    ax.set_xlabel('Time $t$', fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_ylabel(ylabel, fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_title(title, fontsize=PLOT_STYLE['title_fontsize'])
    ax.grid(True, alpha=PLOT_STYLE['grid_alpha'])
    if output_path is not None:
        fig.savefig(output_path, dpi=PLOT_STYLE['save_dpi'], bbox_inches='tight')
    return fig


def plot_omega_scan_heatmap(
    omega_values: np.ndarray,
    times: np.ndarray,
    energy_matrix: np.ndarray,
    title: str = r'$\langle H_0 \rangle$ vs $\omega$',
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot a 2-D heatmap of energy vs (omega, time).

    Args:
        omega_values: 1-D array of scanned omega values.
        times: 1-D time array.
        energy_matrix: 2-D array of shape (len(omega_values), len(times)).
        title: Plot title.
        output_path: If provided, save figure here.

    Returns:
        Matplotlib Figure object.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'])
    im = ax.pcolormesh(times, omega_values, energy_matrix, cmap='inferno', shading='auto')
    fig.colorbar(im, ax=ax, label='$\\langle H_0 \\rangle$')
    ax.set_xlabel('Time $t$', fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_ylabel('$\\omega$', fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_title(title, fontsize=PLOT_STYLE['title_fontsize'])
    if output_path is not None:
        fig.savefig(output_path, dpi=PLOT_STYLE['save_dpi'], bbox_inches='tight')
    return fig
