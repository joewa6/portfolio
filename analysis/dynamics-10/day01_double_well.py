"""
Day 01: Distributions Are Not Dynamics

Demonstrates that fitting the correct equilibrium distribution
does not guarantee correct transition dynamics (MFPT).

System: 1D double-well potential with Langevin dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from utils import save_figure, set_plot_style

# Set consistent plotting style
set_plot_style()


def double_well_potential(x, a=1.0, b=4.0):
    """
    V(x) = -a*x^2/2 + b*x^4/4
    """
    return -a * x**2 / 2 + b * x**4 / 4


def potential_derivative(x, a=1.0, b=4.0):
    """
    dV/dx = -a*x + b*x^3
    """
    return -a * x + b * x**3


def run_langevin(x0, T, dt, gamma, a=1.0, b=4.0, seed=None):
    """
    Simple Euler-Maruyama integration of overdamped Langevin dynamics.
    
    dx/dt = -gamma * dV/dx + sqrt(2*gamma*kT) * eta(t)
    
    Parameters
    ----------
    x0 : float
        Initial position
    T : float
        Total simulation time
    dt : float
        Time step
    gamma : float
        Friction coefficient
    a, b : float
        Potential parameters
    seed : int, optional
        Random seed
    
    Returns
    -------
    t : ndarray
        Time points
    x : ndarray
        Positions at each time point
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)
    x = np.zeros(n_steps)
    x[0] = x0
    
    kT = 1.0  # Temperature in reduced units
    noise_prefactor = np.sqrt(2 * gamma * kT * dt)
    
    for i in range(1, n_steps):
        force = -potential_derivative(x[i-1], a, b)
        noise = np.random.randn()
        x[i] = x[i-1] + gamma * force * dt + noise_prefactor * noise
    
    return t, x


def main():
    """
    Generate all figures for Day 01.
    """
    # Example: Plot the double-well potential
    x = np.linspace(-2, 2, 200)
    V = double_well_potential(x)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, V, 'k-', linewidth=2)
    ax.set_xlabel('Reaction Coordinate x')
    ax.set_ylabel('Potential Energy V(x)')
    ax.set_title('Double-Well Potential')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    save_figure(fig, "day01", "double_well_potential.png")
    plt.close(fig)
    
    # TODO: Add figures showing:
    # 1. Equilibrium distribution comparison
    # 2. MFPT as a function of barrier height
    # 3. Transition matrix comparison
    
    print("\nDay 01 figures generated.")
    print("Next steps:")
    print("  1. Add MFPT calculations")
    print("  2. Compare models with correct distribution but wrong dynamics")
    print("  3. Show decision quantity divergence")


if __name__ == "__main__":
    main()
