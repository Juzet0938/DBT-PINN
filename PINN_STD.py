# =============================================================
# Title: STD-PINN (Standard PINN)
# Description:  All state variables share a single neural network with joint training strategy.

# Author: zzhu
# Date: June 24, 2026
# Contact: zhuzhi@nudt.edu.cn
# =============================================================

# Avoid Intel OpenMP error
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import time
import copy

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# =============================================================
# Constants
# =============================================================
g = 9.81

# =============================================================
# Control inputs (constant)
# =============================================================
nx = 0.1      # Tangential overload
nz = 1.2      # Normal overload
mu = np.pi/3  # Bank angle (rad)

def control_inputs(t):
    """Control inputs (constant)"""
    return nx, nz, mu

# =============================================================
# Aircraft point-mass dynamics (for RK45)
# =============================================================
def aircraft_dynamics(t, state):
    """
    Aircraft point-mass model
    state = [x, y, z, v, gamma, chi]
    """
    x, y, z, v, gamma, chi = state
    
    # Get control inputs
    nx_val, nz_val, mu_val = control_inputs(t)
    
    # Dynamics equations
    dx_dt = v * np.cos(gamma) * np.cos(chi)
    dy_dt = v * np.cos(gamma) * np.sin(chi)
    dz_dt = v * np.sin(gamma)
    dv_dt = g * (nx_val - np.sin(gamma))
    dgamma_dt = (g / (v + 1e-6)) * (nz_val * np.cos(mu_val) - np.cos(gamma))
    dchi_dt = (g / ((v + 1e-6) * (np.cos(gamma) + 1e-6))) * nz_val * np.sin(mu_val)
    
    return [dx_dt, dy_dt, dz_dt, dv_dt, dgamma_dt, dchi_dt]

# =============================================================
# Solve with RK45
# =============================================================
def solve_with_rk45(initial_state, t_span, t_eval):
    """Solve using RK45 integrator"""
    solution = solve_ivp(
        aircraft_dynamics, 
        t_span, 
        initial_state, 
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8, 
        atol=1e-10
    )
    
    if not solution.success:
        print("RK45 solution failed:", solution.message)
        return None
    
    return solution

# =============================================================
# PINN model (single network, 6 outputs)
# =============================================================
class PINN(nn.Module):
    def __init__(self, layers=[1, 256, 256, 256, 6]):
        super(PINN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers)-2):
            self.net.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
            self.net.add_module(f'activation_{i}', nn.Tanh())
        self.net.add_module(f'linear_{len(layers)-2}', nn.Linear(layers[-2], layers[-1]))
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, t):
        return self.net(t)

# =============================================================
# Physics residuals
# =============================================================
def physics_residuals(t, model):
    t = t.clone().detach().requires_grad_(True)
    pred = model(t)
    
    # Split outputs
    x = pred[:, 0:1]
    y = pred[:, 1:2]
    z = pred[:, 2:3]
    v = pred[:, 3:4]
    gamma = pred[:, 4:5]
    chi = pred[:, 5:6]
    
    # Compute time derivatives
    dx_dt = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    dy_dt = torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]
    dz_dt = torch.autograd.grad(z, t, torch.ones_like(z), create_graph=True)[0]
    dv_dt = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    dgamma_dt = torch.autograd.grad(gamma, t, torch.ones_like(gamma), create_graph=True)[0]
    dchi_dt = torch.autograd.grad(chi, t, torch.ones_like(chi), create_graph=True)[0]
    
    # Numerical stability handling
    v_safe = torch.abs(v) + 1e-6
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)
    cos_chi = torch.cos(chi)
    sin_chi = torch.sin(chi)
    cos_gamma_safe = torch.where(torch.abs(cos_gamma) > 0.01, cos_gamma, torch.sign(cos_gamma) * 0.01)
    
    # Control inputs (constant, converted to tensor)
    nx_tensor = torch.ones_like(t) * nx
    nz_tensor = torch.ones_like(t) * nz
    mu_tensor = torch.ones_like(t) * mu
    
    # Dynamics residuals
    f1 = dx_dt - v * cos_gamma * cos_chi
    f2 = dy_dt - v * cos_gamma * sin_chi
    f3 = dz_dt - v * sin_gamma
    f4 = dv_dt - g * (nx_tensor - sin_gamma)
    f5 = dgamma_dt - (g / v_safe) * (nz_tensor * torch.cos(mu_tensor) - cos_gamma)
    f6 = dchi_dt - (g / (v_safe * cos_gamma_safe)) * nz_tensor * torch.sin(mu_tensor)
    
    return f1, f2, f3, f4, f5, f6

# =============================================================
# PINN loss function
# =============================================================
def pinn_loss(model, t_f, initial_state_tensor):
    f1, f2, f3, f4, f5, f6 = physics_residuals(t_f, model)
    physics_loss = (f1**2 + f2**2 + f3**2 + f4**2 + f5**2 + f6**2).mean()
    
    # Initial condition
    t0 = torch.tensor([[0.0]])
    pred0 = model(t0)
    ic_loss = torch.mean((pred0 - initial_state_tensor)**2)
    
    # Total loss
    total_loss = physics_loss + 10.0 * ic_loss  # Initial condition weight
    
    return total_loss, physics_loss, ic_loss

# =============================================================
# Compute RMSE
# =============================================================
def compute_rmse(pinn_pred, rk_solution):
    """Compute RMSE for each state variable"""
    rmse = {}
    state_names = ['x', 'y', 'z', 'v', 'gamma', 'chi']
    
    for i, name in enumerate(state_names):
        pinn_vals = pinn_pred[:, i]
        rk_vals = rk_solution[:, i]
        
        if name in ['gamma', 'chi']:
            error = np.degrees(pinn_vals) - np.degrees(rk_vals)
        else:
            error = pinn_vals - rk_vals
        rmse[name] = np.sqrt(np.mean(error**2))
    
    return rmse

# =============================================================
# Compute error statistics (full version)
# =============================================================
def compute_errors(pinn_solution, rk_solution):
    """Compute error statistics between PINN and RK45"""
    errors = {}
    state_names = ['x', 'y', 'z', 'v', 'gamma', 'chi']
    
    for i, name in enumerate(state_names):
        pinn_vals = pinn_solution[:, i]
        rk_vals = rk_solution[:, i]
        
        abs_error = np.abs(pinn_vals - rk_vals)
        rms_rk = np.sqrt(np.mean(rk_vals**2))
        
        if rms_rk > 1e-6:
            rel_error = np.sqrt(np.mean(abs_error**2)) / rms_rk
        else:
            rel_error = np.sqrt(np.mean(abs_error**2))
        
        errors[name] = {
            'max_abs_error': np.max(abs_error),
            'mean_abs_error': np.mean(abs_error),
            'rms_abs_error': np.sqrt(np.mean(abs_error**2)),
            'rel_error': rel_error
        }
    
    return errors

# =============================================================
# Train PINN (supports saving predictions at checkpoints)
# =============================================================
def train_pinn(initial_state_tensor, t_train, t_test, rk_states, epochs=20001):
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.5)
    
    loss_history = []
    physics_history = []
    ic_history = []
    
    # Store RMSE at intermediate iterations
    checkpoint_epochs = [5000, 10000, 15000, 20000]
    checkpoint_predictions = {}
    checkpoint_rmse = {}
    
    print("\nStarting PINN training (single network joint training)...")
    print("="*60)
    
    # Training start time
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss, physics_loss, ic_loss = pinn_loss(model, t_train, initial_state_tensor)
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(total_loss)
        
        loss_history.append(total_loss.item())
        physics_history.append(physics_loss.item())
        ic_history.append(ic_loss.item())
        
        # Save predictions and compute RMSE at specified iterations
        if (epoch + 1) in checkpoint_epochs:
            model.eval()
            with torch.no_grad():
                t_test_tensor = torch.tensor(t_test.reshape(-1, 1), dtype=torch.float32)
                pinn_pred = model(t_test_tensor).numpy()
            checkpoint_predictions[epoch + 1] = pinn_pred
            
            # Compute RMSE
            rmse = compute_rmse(pinn_pred, rk_states)
            checkpoint_rmse[epoch + 1] = rmse
            
            # Compute average RMSE
            avg_rmse = np.mean([rmse[name] for name in ['x', 'y', 'z', 'v', 'gamma', 'chi']])
            print(f"\n>>> Iteration {epoch + 1}:")
            print(f"  Average RMSE: {avg_rmse:.4f}")
            print(f"  Position RMSE: x={rmse['x']:.4f}, y={rmse['y']:.4f}, z={rmse['z']:.4f}")
            print(f"  Velocity RMSE: v={rmse['v']:.4f}")
            print(f"  Angle RMSE: gamma={rmse['gamma']:.4f}, chi={rmse['chi']:.4f}")

            # Training end
            end_time = time.time()
            total_time = end_time - start_time
            print(f"\nTraining time: {total_time:.2f} seconds")
            
            model.train()
        
        if epoch % 1000 == 0 and epoch > 0:
            print(f"Epoch {epoch:5d}, Loss: {total_loss.item():.3e}, "
                  f"Physics: {physics_loss.item():.3e}, IC: {ic_loss.item():.3e}")
    
    return model, loss_history, physics_history, ic_history, checkpoint_predictions, checkpoint_rmse

# =============================================================
# Print results table
# =============================================================
def print_checkpoint_results_table(checkpoint_rmse):
    """Print RMSE results table at checkpoints"""
    print("\n" + "="*80)
    print("Std-PINN RMSE Comparison at Different Iterations")
    print("="*80)
    print(f"\n{'Iterations':<12} {'x RMSE(m)':<15} {'y RMSE(m)':<15} {'z RMSE(m)':<15} "
          f"{'v RMSE(m/s)':<15} {'gamma RMSE(deg)':<15} {'chi RMSE(deg)':<15} {'Average RMSE':<12}")
    print("-"*130)
    
    for epoch in sorted(checkpoint_rmse.keys()):
        rmse = checkpoint_rmse[epoch]
        avg_rmse = np.mean([rmse[name] for name in ['x', 'y', 'z', 'v', 'gamma', 'chi']])
        print(f"{epoch:<12} {rmse['x']:<15.4f} {rmse['y']:<15.4f} {rmse['z']:<15.4f} "
              f"{rmse['v']:<15.4f} {rmse['gamma']:<15.4f} {rmse['chi']:<15.4f} {avg_rmse:<12.4f}")
    
    print("="*80)

# =============================================================
# Plot RMSE convergence curves
# =============================================================
def plot_rmse_convergence(checkpoint_rmse):
    """Plot RMSE convergence curves over iterations"""
    epochs = sorted(checkpoint_rmse.keys())
    state_names = ['x', 'y', 'z', 'v', 'gamma', 'chi']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    units = ['m', 'm', 'm', 'm/s', 'deg', 'deg']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, color, unit) in enumerate(zip(state_names, colors, units)):
        ax = axes[idx]
        rmse_values = [checkpoint_rmse[epoch][name] for epoch in epochs]
        
        ax.plot(epochs, rmse_values, 'o-', color=color, linewidth=2, markersize=8)
        ax.set_xlabel('Training Iterations', fontsize=11)
        ax.set_ylabel(f'RMSE ({unit})', fontsize=11)
        title_name = {'x': 'X Position', 'y': 'Y Position', 'z': 'Z Position',
                      'v': 'Velocity', 'gamma': 'gamma Angle', 'chi': 'chi Angle'}[name]
        ax.set_title(f'{title_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(epochs, rmse_values):
            ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 8), 
                       textcoords='offset points', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('PINNSTD_rmse_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nRMSE convergence curve saved as: PINNSTD_rmse_convergence.png")

# =============================================================
# Plot comparison figures
# =============================================================
def plot_comparison(t_test, rk_states, pinn_states, errors, loss_history, physics_history, ic_history):
    """Plot comparison figures"""
    state_names = ['X (m)', 'Y (m)', 'Z (m)', 'Velocity (m/s)', 
                   'Flight Path Angle (deg)', 'Heading Angle (deg)']
    state_keys = ['x', 'y', 'z', 'v', 'gamma', 'chi']
    
    # Convert angles
    rk_states_plot = rk_states.copy()
    pinn_states_plot = pinn_states.copy()
    rk_states_plot[:, 4] = np.degrees(rk_states_plot[:, 4])
    rk_states_plot[:, 5] = np.degrees(rk_states_plot[:, 5])
    pinn_states_plot[:, 4] = np.degrees(pinn_states_plot[:, 4])
    pinn_states_plot[:, 5] = np.degrees(pinn_states_plot[:, 5])
    
    # Create large figure
    fig = plt.figure(figsize=(18, 12))
    
    # State comparison (6 subplots)
    for i in range(6):
        ax = fig.add_subplot(2, 3, i+1)
        ax.plot(t_test, rk_states_plot[:, i], 'b-', linewidth=2, label='RK45 (Reference)')
        ax.plot(t_test, pinn_states_plot[:, i], 'r--', linewidth=2, label='Std-PINN', alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(state_names[i])
        rel_error_pct = errors[state_keys[i]]['rel_error'] * 100
        ax.set_title(f'{state_names[i]}\nRel Error: {rel_error_pct:.2f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('StdPINN_states_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3D trajectory comparison
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(rk_states[:, 0], rk_states[:, 1], rk_states[:, 2], 'b-', linewidth=2, label='RK45')
    ax1.plot(pinn_states[:, 0], pinn_states[:, 1], pinn_states[:, 2], 'r--', linewidth=2, label='Std-PINN')
    ax1.scatter(rk_states[0, 0], rk_states[0, 1], rk_states[0, 2], c='g', marker='o', s=100, label='Start')
    ax1.scatter(rk_states[-1, 0], rk_states[-1, 1], rk_states[-1, 2], c='r', marker='*', s=200, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = fig.add_subplot(132)
    ax2.plot(rk_states[:, 0], rk_states[:, 1], 'b-', linewidth=2, label='RK45')
    ax2.plot(pinn_states[:, 0], pinn_states[:, 1], 'r--', linewidth=2, label='Std-PINN')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    ax3 = fig.add_subplot(133)
    ax3.plot(rk_states[:, 0], rk_states[:, 2], 'b-', linewidth=2, label='RK45')
    ax3.plot(pinn_states[:, 0], pinn_states[:, 2], 'r--', linewidth=2, label='Std-PINN')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ Projection')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('PINNSTD_trajectory.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Loss history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].semilogy(loss_history)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss History')
    axes[0].grid(True)
    
    axes[1].semilogy(physics_history)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Physics Loss')
    axes[1].set_title('Physics Residual Loss')
    axes[1].grid(True)
    
    axes[2].semilogy(ic_history)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IC Loss')
    axes[2].set_title('Initial Condition Loss')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('PINNSTD_loss_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Error distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, (name, error_metrics) in enumerate(errors.items()):
        abs_error = np.abs(rk_states[:, i] - pinn_states[:, i])
        if name in ['gamma', 'chi']:
            abs_error = np.degrees(abs_error)
        axes[i].plot(t_test, abs_error, 'b-', linewidth=1)
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel(f'{name.upper()} Error')
        unit = 'deg' if name in ['gamma', 'chi'] else 'm' if name in ['x','y','z'] else 'm/s'
        axes[i].set_title(f'{name.upper()} Error\nMax: {error_metrics["max_abs_error"]:.4f} {unit}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('PINNSTD_error_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================
# Print final results table
# =============================================================
def print_results_table(errors):
    """Print formatted results table"""
    print("\n" + "="*80)
    print("Std-PINN Final Prediction Error Statistics (vs RK45)")
    print("="*80)
    print(f"\n{'Variable':<12} {'Max Abs Error':<20} {'Mean Abs Error':<20} {'RMS Error':<20} {'Relative Error':<15}")
    print("-"*90)
    
    for name, metrics in errors.items():
        unit = 'deg' if name in ['gamma', 'chi'] else 'm' if name in ['x','y','z'] else 'm/s'
        if name in ['gamma', 'chi']:
            max_err = np.degrees(metrics['max_abs_error'])
            mean_err = np.degrees(metrics['mean_abs_error'])
            rms_err = np.degrees(metrics['rms_abs_error'])
        else:
            max_err = metrics['max_abs_error']
            mean_err = metrics['mean_abs_error']
            rms_err = metrics['rms_abs_error']
        print(f"{name.upper():<12} {max_err:<20.4f} {mean_err:<20.4f} {rms_err:<20.4f} {metrics['rel_error']*100:<14.2f}%")
    
    print("="*80)

# =============================================================
# Main program
# =============================================================
def main():
    # Initial state [x, y, z, v, gamma, chi]
    initial_state = np.array([0.0, 0.0, 0.0, 200.0, np.pi/3, np.pi/3])
    initial_state_tensor = torch.tensor(initial_state, dtype=torch.float32)
    
    # Parameter settings
    t_start, t_end = 0.0, 10.0
    t_span = (t_start, t_end)
    n_points = 500   # Number of training sample points
    n_test = 500     # Number of test points
    t_test = np.linspace(t_start, t_end, n_test)  # Test time points
    t_train = torch.linspace(t_start, t_end, n_points).view(-1, 1)
    
    print("="*80)
    print("Std-PINN (Single Network Joint Training) - Baseline Method")
    print("="*80)
    print(f"Time range: [{t_start}, {t_end}]")
    print(f"Training sample points: {n_points}")
    print(f"Test points: {n_test}")
    print(f"Initial state: x={initial_state[0]}, y={initial_state[1]}, z={initial_state[2]}, "
          f"v={initial_state[3]}, gamma={np.degrees(initial_state[4]):.1f} deg, chi={np.degrees(initial_state[5]):.1f} deg")
    print(f"Control inputs: nx={nx}, nz={nz}, mu={np.degrees(mu):.1f} deg")
    print("="*80)
    
    # 1. Solve reference solution with RK45
    print("\nComputing RK45 reference solution...")
    rk_solution = solve_with_rk45(initial_state, t_span, t_test)
    if rk_solution is None:
        return
    
    rk_states = rk_solution.y.T  # (n_test, 6)
    
    # 2. Train Std-PINN (record predictions at intermediate iterations)
    model, loss_history, physics_history, ic_history, checkpoint_predictions, checkpoint_rmse = train_pinn(
        initial_state_tensor, t_train, t_test, rk_states, epochs=20001
    )
    
    # 3. Print RMSE results at checkpoints
    print_checkpoint_results_table(checkpoint_rmse)
    
    # 4. Plot RMSE convergence curves
    plot_rmse_convergence(checkpoint_rmse)
    
    # 5. Final prediction
    model.eval()
    with torch.no_grad():
        pinn_pred = model(torch.tensor(t_test.reshape(-1, 1), dtype=torch.float32)).numpy()
    
    # 6. Compute final errors
    errors = compute_errors(pinn_pred, rk_states)
    
    # 7. Print final error statistics
    print_results_table(errors)
    
    # 8. Visualization comparison
    plot_comparison(t_test, rk_states, pinn_pred, errors, loss_history, physics_history, ic_history)

if __name__ == "__main__":
    main()