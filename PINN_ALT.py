# =============================================================
# Title: ALT-PINN (Alternating PINN)
# Description: Each state variable has its own independent neural network with an alternating training strategy, equivalent to using a fixed single inner-loop step.

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
import copy
import time
import pandas as pd

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# =============================================================
# Constants
# =============================================================
g = 9.81
nx = 0.1
nz = 1.2
mu = np.pi/3

# =============================================================
# PINN network definition
# =============================================================
class PINN(nn.Module):
    def __init__(self, hidden_layers=[64, 64, 64]):
        super(PINN, self).__init__()
        layers = []
        input_dim = 1
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, t):
        return self.net(t)

# =============================================================
# Iterative PINN solver
# =============================================================
class IterativePINN:
    def __init__(self, t_start, t_end, n_points, network_configs):
        self.t_start = t_start
        self.t_end = t_end
        self.n_points = n_points
        self.t = torch.linspace(t_start, t_end, n_points).view(-1, 1)
        
        self.nets = {}
        for name, config in network_configs.items():
            self.nets[name] = PINN(hidden_layers=config)
        
        self.optimizers = {
            name: optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5) 
            for name, net in self.nets.items()
        }
        
        self.ic = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'v': 200.0, 'gamma': np.pi/3, 'chi': np.pi/3
        }
        
        self.loss_history = {name: [] for name in self.nets.keys()}
        self.loss_history['total'] = []
        self.checkpoint_predictions = {}
        self.checkpoint_t = None
    
    def compute_derivative(self, net, t):
        t.requires_grad = True
        output = net(t)
        derivative = torch.autograd.grad(output, t, torch.ones_like(output), create_graph=True)[0]
        return derivative, output
    
    def get_all_prev_values(self, t, prev_nets):
        values = {}
        with torch.no_grad():
            for name, net in prev_nets.items():
                values[name] = net(t)
        return values
    
    def compute_loss(self, name, t, prev_values):
        t_train = t.clone().detach().requires_grad_(True)
        net = self.nets[name]
        derivative, pred = self.compute_derivative(net, t_train)
        
        if name == 'x':
            v = prev_values['v']; gamma = prev_values['gamma']; chi = prev_values['chi']
            residual = derivative - v * torch.cos(gamma) * torch.cos(chi)
        elif name == 'y':
            v = prev_values['v']; gamma = prev_values['gamma']; chi = prev_values['chi']
            residual = derivative - v * torch.cos(gamma) * torch.sin(chi)
        elif name == 'z':
            v = prev_values['v']; gamma = prev_values['gamma']
            residual = derivative - v * torch.sin(gamma)
        elif name == 'v':
            gamma = prev_values['gamma']
            residual = derivative - g * (nx - torch.sin(gamma))
        elif name == 'gamma':
            v = prev_values['v']; gamma_prev = prev_values['gamma']
            residual = derivative - (g / (v + 1e-6)) * (nz * np.cos(mu) - torch.cos(gamma_prev))
        elif name == 'chi':
            v = prev_values['v']; gamma = prev_values['gamma']
            cos_gamma_safe = torch.cos(gamma).clamp(min=0.01, max=0.99)
            residual = derivative - (g / (v * cos_gamma_safe + 1e-6)) * nz * np.sin(mu)
        else:
            raise ValueError(f"Unknown state name: {name}")
        
        ode_loss = (residual**2).mean()
        t0 = torch.tensor([[0.0]])
        pred0 = net(t0)
        ic_loss = ((pred0 - self.ic[name])**2).mean()
        weight_ic = 10
        total_loss = ode_loss + weight_ic * ic_loss
        return total_loss, ode_loss.item(), ic_loss.item()
    
    def train(self, n_iterations, verbose=False):
        """Train the model"""
        prev_nets = copy.deepcopy(self.nets)
        checkpoint_iterations = [5000, 10000, 15000, 20000] if n_iterations >= 20000 else [n_iterations//4, n_iterations//2, n_iterations*3//4, n_iterations]
        
        for iteration in range(n_iterations):
            total_loss = 0
            losses = {}
            prev_values = self.get_all_prev_values(self.t, prev_nets)
            
            for name in ['x', 'y', 'z', 'v', 'gamma', 'chi']:
                optimizer = self.optimizers[name]
                optimizer.zero_grad()
                loss, _, _ = self.compute_loss(name, self.t, prev_values)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nets[name].parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                losses[name] = loss.item()
            
            for name, loss in losses.items():
                self.loss_history[name].append(loss)
            self.loss_history['total'].append(total_loss)
            
            for name in self.nets.keys():
                prev_nets[name].load_state_dict(self.nets[name].state_dict())
            
            if (iteration + 1) in checkpoint_iterations:
                with torch.no_grad():
                    if self.checkpoint_t is None:
                        self.checkpoint_t = self.t.clone().detach().numpy().flatten()
                    pred = {name: net(self.t).numpy().flatten() for name, net in self.nets.items()}
                self.checkpoint_predictions[iteration + 1] = pred
                if verbose:
                    print(f"  >>> Saved predictions at iteration {iteration + 1}")
            
            if verbose and (iteration + 1) % 500 == 0:
                print(f"  Iteration {iteration + 1} completed, total loss: {total_loss:.3e}")
        
        return self
    
    def predict(self, t_eval):
        pred = {}
        for name, net in self.nets.items():
            net.eval()
            with torch.no_grad():
                t_tensor = torch.tensor(t_eval.reshape(-1, 1), dtype=torch.float32)
                pred[name] = net(t_tensor).numpy().flatten()
        return pred

# =============================================================
# RK45 reference solution
# =============================================================
def solve_with_rk45(t_start, t_end, initial_state, t_eval):
    def dynamics(t, state):
        x, y, z, v, gamma, chi = state
        dx_dt = v * np.cos(gamma) * np.cos(chi)
        dy_dt = v * np.cos(gamma) * np.sin(chi)
        dz_dt = v * np.sin(gamma)
        dv_dt = g * (nx - np.sin(gamma))
        dgamma_dt = (g / v) * (nz * np.cos(mu) - np.cos(gamma)) if v > 0 else 0
        dchi_dt = (g / (v * np.cos(gamma) + 1e-6)) * nz * np.sin(mu) if v > 0 and abs(np.cos(gamma)) > 1e-6 else 0
        return [dx_dt, dy_dt, dz_dt, dv_dt, dgamma_dt, dchi_dt]
    
    solution = solve_ivp(dynamics, (t_start, t_end), initial_state, 
                         method='RK45', t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return solution.y

# =============================================================
# Compute RMSE
# =============================================================
def compute_rmse(pinn_pred, rk45_pred):
    """Compute RMSE for each state variable"""
    rmse = {}
    state_names = ['x', 'y', 'z', 'v', 'gamma', 'chi']
    for name in state_names:
        if name in ['gamma', 'chi']:
            error = np.degrees(pinn_pred[name]) - np.degrees(rk45_pred[name])
        else:
            error = pinn_pred[name] - rk45_pred[name]
        rmse[name] = np.sqrt(np.mean(error**2))
    return rmse

# =============================================================
# Ablation experiment: comparison of different iteration counts
# =============================================================
def run_ablation_experiment(iterations_list, network_configs, t_start, t_end, n_points, t_test, rk45_pred):
    """Run ablation experiment with different iteration counts"""
    results = {}
    all_predictions = {}
    
    print("\n" + "="*80)
    print("Ablation Experiment: Comparison of Different Iteration Counts")
    print("="*80)
    
    for n_iter in iterations_list:
        print(f"\n>>> Training for {n_iter} iterations...")

        # Training start time
        start_time = time.time()
        
        # Create new model and train
        solver = IterativePINN(t_start, t_end, n_points, network_configs=network_configs)
        solver.train(n_iterations=n_iter, verbose=False)
        
        # Predict
        pinn_pred = solver.predict(t_test)
        all_predictions[n_iter] = pinn_pred
        
        # Compute RMSE
        rmse = compute_rmse(pinn_pred, rk45_pred)
        avg_rmse = np.mean([rmse[name] for name in ['x', 'y', 'z', 'v', 'gamma', 'chi']])
        results[n_iter] = {'rmse': rmse, 'avg_rmse': avg_rmse}
        
        # Print results
        print(f"  Position RMSE - x: {rmse['x']:.4f}m, y: {rmse['y']:.4f}m, z: {rmse['z']:.4f}m")
        print(f"  Velocity RMSE - v: {rmse['v']:.4f}m/s")
        print(f"  Angle RMSE - gamma: {rmse['gamma']:.4f} deg, chi: {rmse['chi']:.4f} deg")
        print(f"  Average RMSE: {avg_rmse:.4f}")

        # Training end
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTraining completed! Total training time: {total_time:.2f} seconds")
    
    return results, all_predictions

# =============================================================
# Plot ablation experiment results
# =============================================================
def plot_ablation_results(results, iterations_list):
    """Plot RMSE comparison bar chart for different iteration counts"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    state_names = ['x', 'y', 'z', 'v', 'gamma', 'chi']
    state_units = ['m', 'm', 'm', 'm/s', 'deg', 'deg']
    state_titles = ['X Position', 'Y Position', 'Z Position', 'Velocity', 'Flight Path Angle', 'Heading Angle']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (name, unit, title, color) in enumerate(zip(state_names, state_units, state_titles, colors)):
        ax = axes[idx]
        rmse_values = [results[n_iter]['rmse'][name] for n_iter in iterations_list]
        max_rmse = max(rmse_values)
        
        bars = ax.bar([str(i) for i in iterations_list], rmse_values, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Training Iterations', fontsize=11)
        ax.set_ylabel(f'RMSE ({unit})', fontsize=11)
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set Y-axis upper limit
        ax.set_ylim(0, max_rmse * 1.1)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, rmse_values):
            ax.text(bar.get_x() + bar.get_width()/2, val + max_rmse * 0.03,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('PINNALT_ablation_rmse_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nAblation result figure saved as: PINNALT_ablation_rmse_comparison.png")

def plot_ablation_rmse_curve(results, iterations_list):
    """Plot average RMSE curve as a function of iteration count"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_rmse = [results[n_iter]['avg_rmse'] for n_iter in iterations_list]
    
    ax.plot(iterations_list, avg_rmse, 'b-o', linewidth=2, markersize=8, label='Average RMSE')
    ax.set_xlabel('Training Iterations', fontsize=12)
    ax.set_ylabel('Average RMSE', fontsize=12)
    ax.set_title('Convergence Analysis: Average RMSE vs Training Iterations', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add value labels
    for x, y in zip(iterations_list, avg_rmse):
        ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('PINNALT_ablation_rmse_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("RMSE convergence curve saved as: PINNALT_ablation_rmse_curve.png")


def plot_ablation_trajectory_comparison(results, all_predictions, rk45_pred, t_test, iterations_list):
    """Plot trajectory comparison at different iteration counts"""
    fig = plt.figure(figsize=(15, 10))
    
    # Select a subset of iterations to display (avoid overcrowding)
    show_iterations = [iterations_list[0], iterations_list[len(iterations_list)//2], iterations_list[-1]]
    
    colors = ['orange', 'green', 'red']
    
    # 3D trajectory comparison
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Plot RK45 reference trajectory
    ax1.plot(rk45_pred['x'], rk45_pred['y'], rk45_pred['z'], 
            'k-', linewidth=2.5, label='RK45 (Reference)', alpha=0.9)
    
    # Plot predicted trajectories at different iterations
    for i, n_iter in enumerate(show_iterations):
        pred = all_predictions[n_iter]
        ax1.plot(pred['x'], pred['y'], pred['z'], 
                '--', color=colors[i], linewidth=2, label=f'Iteration {n_iter}', alpha=0.8)
    
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    ax1.set_title('Trajectory Convergence', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.view_init(elev=25, azim=-60)
    
    # Average RMSE convergence curve
    ax2 = fig.add_subplot(2, 2, 2)
    avg_rmse = [results[n_iter]['avg_rmse'] for n_iter in iterations_list]
    ax2.plot(iterations_list, avg_rmse, 'b-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Training Iterations', fontsize=11)
    ax2.set_ylabel('Average RMSE', fontsize=11)
    ax2.set_title('Convergence of Average RMSE', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Position RMSE convergence curves
    ax3 = fig.add_subplot(2, 2, 3)
    for name in ['x', 'y', 'z']:
        rmse_vals = [results[n_iter]['rmse'][name] for n_iter in iterations_list]
        ax3.plot(iterations_list, rmse_vals, '-o', linewidth=1.5, markersize=6, label=f'{name.upper()}')
    ax3.set_xlabel('Training Iterations', fontsize=11)
    ax3.set_ylabel('RMSE (m)', fontsize=11)
    ax3.set_title('Position RMSE Convergence', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Velocity and angle RMSE convergence curves
    ax4 = fig.add_subplot(2, 2, 4)
    rmse_v = [results[n_iter]['rmse']['v'] for n_iter in iterations_list]
    rmse_gamma = [results[n_iter]['rmse']['gamma'] for n_iter in iterations_list]
    rmse_chi = [results[n_iter]['rmse']['chi'] for n_iter in iterations_list]
    ax4.plot(iterations_list, rmse_v, '-o', linewidth=1.5, markersize=6, label='Velocity (m/s)')
    ax4.plot(iterations_list, rmse_gamma, '-s', linewidth=1.5, markersize=6, label='gamma (deg)')
    ax4.plot(iterations_list, rmse_chi, '-^', linewidth=1.5, markersize=6, label='chi (deg)')
    ax4.set_xlabel('Training Iterations', fontsize=11)
    ax4.set_ylabel('RMSE', fontsize=11)
    ax4.set_title('Velocity & Angle RMSE Convergence', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('PINNALT_ablation_convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Convergence analysis figure saved as: PINNALT_ablation_convergence_analysis.png")

# =============================================================
# Print results table
# =============================================================
def print_results_table(results, iterations_list):
    """Print formatted results table"""
    print("\n" + "="*80)
    print("Ablation Experiment Comparison Results")
    print("="*80)
    print(f"\n{'Iterations':<12} {'x RMSE(m)':<15} {'y RMSE(m)':<15} {'z RMSE(m)':<15} {'v RMSE(m/s)':<15} {'gamma RMSE(deg)':<15} {'chi RMSE(deg)':<15} {'Average RMSE':<12}")
    print("-"*130)
    
    for n_iter in iterations_list:
        rmse = results[n_iter]['rmse']
        print(f"{n_iter:<12} {rmse['x']:<15.4f} {rmse['y']:<15.4f} {rmse['z']:<15.4f} "
              f"{rmse['v']:<15.4f} {rmse['gamma']:<15.4f} {rmse['chi']:<15.4f} {results[n_iter]['avg_rmse']:<12.4f}")
    
    print("="*80)

# =============================================================
# Main program
# =============================================================
def main():
    # Parameter settings
    t_start, t_end = 0.0, 10.0
    n_points = 500
    n_test = 50
    t_test = np.linspace(t_start, t_end, n_test)
    initial_state = np.array([0.0, 0.0, 0.0, 200.0, np.pi/3, np.pi/3])
    
    # Network architecture configuration
    network_configs = {
        'x': [256, 256, 256],
        'y': [256, 256, 256],
        'z': [256, 256, 256],
        'v': [128, 128, 128],
        'gamma': [128, 128, 128],
        'chi': [128, 128, 128]
    }
    
    print("="*80)
    print("Aircraft 3-DOF Model - PINN Solution (Ablation: Iteration Count Comparison)")
    print("="*80)
    
    # Compute RK45 reference solution
    print("\nComputing RK45 reference solution...")
    rk45_solution = solve_with_rk45(t_start, t_end, initial_state, t_test)
    rk45_pred = {
        'x': rk45_solution[0], 'y': rk45_solution[1], 'z': rk45_solution[2],
        'v': rk45_solution[3], 'gamma': rk45_solution[4], 'chi': rk45_solution[5]
    }
    
    # Ablation experiment: different iteration counts
    iterations_list = [5000, 10000, 15000, 20000]
    results, all_predictions = run_ablation_experiment(
        iterations_list, network_configs, t_start, t_end, n_points, t_test, rk45_pred
    )
    
    # Print results table
    print_results_table(results, iterations_list)
    
    # Plot ablation experiment results
    plot_ablation_results(results, iterations_list)
    plot_ablation_rmse_curve(results, iterations_list)
    plot_ablation_trajectory_comparison(results, all_predictions, rk45_pred, t_test, iterations_list)
    
    print("\nAblation experiment completed!")

if __name__ == "__main__":
    main()