# =============================================================
# Title: DBT-PINN (Duel-Balancing Training PINN)
# Description: Each state variable has its own independent neural network with the DBT training strategy, incorporating both external alternating optimization and internal adaptive step sizes.
# Author: zzhu
# Date: June 24, 2026
# Contact: zhuzhi@nudt.edu.cn
# =============================================================

# Avoid Intel OpenMP error
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# Improvements:
# 1) Adaptive training steps
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import copy

import time

from matplotlib.lines import Line2D

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# =============================================================
# Constants
# =============================================================
g = 9.81  # Gravitational acceleration m/s²

# Control inputs (constant)
nx = 0.1      # Tangential overload
nz = 1.2      # Normal overload
mu = np.pi/3  # Bank angle (rad)

# =============================================================
# Configurable PINN network (supports different structures)
# =============================================================
class PINN(nn.Module):
    """
    General PINN network supporting custom hidden layers and neuron counts
    
    Args:
        hidden_layers: List of neuron counts per layer, e.g., [64, 64, 64] 
                       represents 3 hidden layers with 64 neurons each
    """
    def __init__(self, hidden_layers=[64, 64, 64]):
        super(PINN, self).__init__()
        
        layers = []
        input_dim = 1
        
        # Build hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        # Output layer
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
# Iterative PINN solver (supports different network structures)
# =============================================================
class IterativePINN:
    def __init__(self, t_start, t_end, n_points, network_configs):
        """
        Iterative PINN solver
        
        Args:
            t_start: Start time
            t_end: End time
            n_points: Number of sampling points
            network_configs: Dictionary configuring the network structure for each state variable
        """
        self.t_start = t_start
        self.t_end = t_end
        self.n_points = n_points
        self.t = torch.linspace(t_start, t_end, n_points).view(-1, 1)
        
        # Print network configuration
        print("\n" + "="*80)
        print("Network Configuration")
        print("="*80)
        for name, config in network_configs.items():
            print(f"  {name}: hidden layers {config} ({len(config)} layers, {sum(config)} neurons total)")
        print("="*80)
        
        # Store 6 networks using dictionary (each with different structure)
        self.nets = {}
        for name, config in network_configs.items():
            self.nets[name] = PINN(hidden_layers=config)
        
        # Optimizer dictionary
        self.optimizers = {
            name: optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5) 
            for name, net in self.nets.items()
        }
        
        # Learning rate scheduler
        self.schedulers = {
            name: optim.lr_scheduler.ReduceLROnPlateau(opt, patience=500, factor=0.5)
            for name, opt in self.optimizers.items()
        }
        
        # Initial conditions dictionary
        self.ic = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'v': 200.0,
            'gamma': np.pi/3,
            'chi': np.pi/3
        }
        
        # Store loss history
        self.loss_history = {name: [] for name in self.nets.keys()}
        self.loss_history['total'] = []
        
        # Store predictions from each iteration
        self.iterations_predictions = []
        self.checkpoint_predictions = {}  # Store predictions at specific iterations
        self.checkpoint_t = None  # Store time points corresponding to checkpoints

        # Add gradient history for moving average normalization
        self.gradient_history = {name: {'mu': 0.0, 'var': 1.0} for name in self.nets.keys()}
        self.grad_norm_history = {name: [] for name in self.nets.keys()}
        
        # Adaptive iteration parameters
        self.K_min = 1      # Minimum iterations
        self.K_max = 5      # Maximum iterations
        self.beta = 0.9     # Moving average momentum
        self.lam = 1.0      # Sigmoid sensitivity
    
    def compute_derivative(self, net, t):
        """Compute the derivative of network output with respect to time"""
        t.requires_grad = True
        output = net(t)
        derivative = torch.autograd.grad(output, t, torch.ones_like(output),
                                         create_graph=True)[0]
        return derivative, output
    
    def get_all_prev_values(self, t, prev_nets):
        """Get all state variable values from the previous iteration"""
        values = {}
        with torch.no_grad():
            for name, net in prev_nets.items():
                values[name] = net(t)
        return values
    
    def compute_loss(self, name, t, prev_values):
        """Compute the loss for a single network"""
        t_train = t.clone().detach().requires_grad_(True)
        net = self.nets[name]
        
        # Compute derivative
        derivative, pred = self.compute_derivative(net, t_train)
        
        # Compute residuals for different state variables
        if name == 'x':
            v = prev_values['v']
            gamma = prev_values['gamma']
            chi = prev_values['chi']
            residual = derivative - v * torch.cos(gamma) * torch.cos(chi)
            
        elif name == 'y':
            v = prev_values['v']
            gamma = prev_values['gamma']
            chi = prev_values['chi']
            residual = derivative - v * torch.cos(gamma) * torch.sin(chi)
            
        elif name == 'z':
            v = prev_values['v']
            gamma = prev_values['gamma']
            residual = derivative - v * torch.sin(gamma)
            
        elif name == 'v':
            gamma = prev_values['gamma']
            residual = derivative - g * (nx - torch.sin(gamma))
            
        elif name == 'gamma':
            v = prev_values['v']
            gamma_prev = prev_values['gamma']
            residual = derivative - (g / (v + 1e-6)) * (nz * np.cos(mu) - torch.cos(gamma_prev))
            
        elif name == 'chi':
            v = prev_values['v']
            gamma = prev_values['gamma']
            cos_gamma_safe = torch.cos(gamma).clamp(min=0.01, max=0.99)
            residual = derivative - (g / (v * cos_gamma_safe + 1e-6)) * nz * np.sin(mu)
        
        else:
            raise ValueError(f"Unknown state name: {name}")
        
        # ODE loss
        ode_loss = (residual**2).mean()
        
        # Initial condition loss (unified weight)
        t0 = torch.tensor([[0.0]])
        pred0 = net(t0)
        ic_loss = ((pred0 - self.ic[name])**2).mean()
        
        # Total loss (using unified initial condition weight)
        weight_ic = 1
        total_loss = ode_loss + weight_ic * ic_loss
        
        return total_loss, ode_loss.item(), ic_loss.item()

    def compute_adaptive_steps(self, name, grad_norm):
        """
        Compute adaptive iteration steps based on gradient norm
        
        Args:
            name: Network name
            grad_norm: Current gradient norm
        
        Returns:
            k: Adaptive iteration steps (integer between 1 and 5)
        """
        # Get historical statistics
        hist = self.gradient_history[name]
        
        # Update moving average and variance
        hist['mu'] = self.beta * hist['mu'] + (1 - self.beta) * grad_norm
        hist['var'] = self.beta * hist['var'] + (1 - self.beta) * (grad_norm - hist['mu']) ** 2
        
        # Normalize
        sigma = np.sqrt(hist['var'] + 1e-8)
        g_std = (grad_norm - hist['mu']) / sigma
        
        # Sigmoid mapping
        p = 1 / (1 + np.exp(-self.lam * g_std))
        
        # Compute iteration steps
        k = self.K_min + round((self.K_max - self.K_min) * (1 - p))
        
        # Boundary clipping
        k = max(self.K_min, min(self.K_max, k))
        
        # Record gradient norm (optional, for debugging)
        self.grad_norm_history[name].append(grad_norm)
        
        return k
    
    def train_step(self, t, prev_nets):
        """Single training step: train all networks and return gradient norms"""
        total_loss = 0
        losses = {}
        grad_norms = {}
        
        # Get all state variable values from previous iteration
        prev_values = self.get_all_prev_values(t, prev_nets)
        
        # Train each network sequentially
        for name in ['x', 'y', 'z', 'v', 'gamma', 'chi']:
            optimizer = self.optimizers[name]
            optimizer.zero_grad()
            
            loss, ode_loss, ic_loss = self.compute_loss(name, t, prev_values)
            loss.backward()
            
            # Compute gradient norm (before gradient clipping)
            total_norm = 0
            for p in self.nets[name].parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = np.sqrt(total_norm)
            grad_norms[name] = grad_norm
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.nets[name].parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            losses[name] = loss.item()
    
        return total_loss, losses, grad_norms
    
    def iterative_train(self, n_iterations, initial_steps_per_iteration=1):
        """Iterative training with adaptive iteration steps"""
        print("\n" + "="*80)
        print("Iterative PINN training started (adaptive iteration steps)")
        print("="*80)
        print(f"Total iterations: {n_iterations}")
        print(f"Initial steps per iteration: {initial_steps_per_iteration}")
        print(f"Adaptive iteration step range: [{self.K_min}, {self.K_max}]")
        print(f"Control inputs: nx={nx}, nz={nz}, mu={mu}")
        print(f"Initial state: (0,0,1000,200,π/3,π/3)")
        print("="*80)
        
        # Initialize previous network
        prev_nets = copy.deepcopy(self.nets)
        
        # Iterations to save as checkpoints
        checkpoint_iterations = [500, 1000, 1500, 2500, 5000]
        
        # Record adaptive step history
        steps_history = {name: [] for name in self.nets.keys()}
        
        # Training start time
        start_time = time.time()
        
        for iteration in range(n_iterations):
            # First compute gradients to get gradient norms for each network
            _, _, grad_norms = self.train_step(self.t, prev_nets)
            
            # Compute adaptive iteration steps for each network based on gradient norms
            adaptive_steps = {}
            for name in self.nets.keys():
                adaptive_steps[name] = self.compute_adaptive_steps(name, grad_norms[name])
                steps_history[name].append(adaptive_steps[name])
            
            if iteration % 100 == 0:
                print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
                print(f"  Adaptive steps: x={adaptive_steps['x']}, y={adaptive_steps['y']}, z={adaptive_steps['z']}, "
                    f"v={adaptive_steps['v']}, gamma={adaptive_steps['gamma']}, chi={adaptive_steps['chi']}")
            
            # For steps exceeding each network's required steps, that network is not updated
            total_loss = 0
            losses = {}
            for name in self.nets.keys():
                # Inner training loop (using adaptive steps)
                for step in range(adaptive_steps[name]):
                    
                    # Get all state variable values from previous iteration
                    prev_values = self.get_all_prev_values(self.t, prev_nets)
                
                    optimizer = self.optimizers[name]
                    optimizer.zero_grad()
                        
                    loss, ode_loss, ic_loss = self.compute_loss(name, self.t, prev_values)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.nets[name].parameters(), max_norm=1.0)
                    optimizer.step()
                        
                    # Record loss (only at the last step)
                    if step == adaptive_steps[name]-1:
                        total_loss += loss.item()
                        losses[name] = loss.item()

            for name, loss in losses.items():
                self.loss_history[name].append(loss)
            self.loss_history['total'].append(total_loss)
            
            # Update previous networks
            for name in self.nets.keys():
                prev_nets[name].load_state_dict(self.nets[name].state_dict())
            
            # Save predictions at specific iterations
            if (iteration + 1) in checkpoint_iterations:
                with torch.no_grad():
                    if self.checkpoint_t is None:
                        self.checkpoint_t = self.t.clone().detach().numpy().flatten()
                    pred = {name: net(self.t).numpy().flatten() 
                        for name, net in self.nets.items()}
                self.checkpoint_predictions[iteration + 1] = pred
                print(f"  >>> Saved predictions at iteration {iteration + 1}")
            
            # Print progress every 100 iterations
            if (iteration + 1) % 100 == 0:
                with torch.no_grad():
                    t0 = torch.tensor([[0.0]])
                    ic_errors = {}
                    for name in self.nets.keys():
                        pred0 = self.nets[name](t0).item()
                        ic_errors[name] = abs(pred0 - self.ic[name])
                avg_steps = np.mean([steps_history[name][-100:] for name in self.nets.keys()])
                print(f"  Iteration {iteration + 1} completed, total loss: {total_loss:.3e}, "
                    f"IC error: {max(ic_errors.values()):.3e}, avg steps: {avg_steps:.2f}")
        
        # Training end
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTraining completed! Total training time: {total_time:.2f} seconds")
        
        # Print adaptive step statistics
        print("\n" + "="*80)
        print("Adaptive Iteration Statistics")
        print("="*80)
        for name in self.nets.keys():
            steps_array = np.array(steps_history[name])
            print(f"  {name}: avg={np.mean(steps_array):.2f}, "
                f"max={np.max(steps_array)}, min={np.min(steps_array)}")
        
        return self.iterations_predictions
    
    def predict(self, t_eval):
        """Make predictions using the trained network"""
        pred = {}
        for name, net in self.nets.items():
            net.eval()
            with torch.no_grad():
                t_tensor = torch.tensor(t_eval.reshape(-1, 1), dtype=torch.float32)
                pred[name] = net(t_tensor).numpy().flatten()
        return pred

# =============================================================
# Numerical integration reference solution (for validation)
# =============================================================
def solve_with_rk45(t_start, t_end, initial_state, t_eval):
    """Use RK45 as reference solution"""
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
# Zoomed region display
# =============================================================
def plot_figure_combined(rk45_pred, pinn_pred, loss_history):
    """
    Combined figure: Left - 3D trajectory comparison, Right - training loss history (with embedded zoomed view)
    """
    # Set academic style parameters
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'mathtext.fontset': 'stix',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.5,
        'grid.linewidth': 0.4,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    # Create side-by-side subplots
    fig = plt.figure(figsize=(15, 6))
    
    # ==================== Left: 3D trajectory ====================
    ax1 = fig.add_subplot(121, projection='3d')
    
    colors = {
        'rk45': '#1f77b4',      # Professional blue
        'pinn': '#d62728'       # Professional red
    }
    
    # Plot RK45 reference trajectory
    ax1.plot(rk45_pred['x'], rk45_pred['y'], rk45_pred['z'], 
            color=colors['rk45'], linewidth=2.2, 
            label='RK45', alpha=0.9, zorder=2)
    
    # Plot PINN predicted trajectory
    ax1.plot(pinn_pred['x'], pinn_pred['y'], pinn_pred['z'], 
            color=colors['pinn'], linewidth=2.0, linestyle='--', 
            label='DBT-PINN', alpha=0.85, zorder=2)
    
    # Start and end coordinates (using PINN predictions)
    start_x, start_y, start_z = pinn_pred['x'][0], pinn_pred['y'][0], pinn_pred['z'][0]
    end_x, end_y, end_z = pinn_pred['x'][-1], pinn_pred['y'][-1], pinn_pred['z'][-1]
    
    # Start text box
    start_text = f'Start: ({start_x:.2f}, {start_y:.2f}, {start_z:.2f})'
    ax1.text(start_x, start_y, start_z, start_text, 
            color='black', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='green', alpha=0.85, linewidth=1.0),
            ha='center', va='bottom', zorder=4)
    
    # End text box
    end_text = f'End: ({end_x:.2f}, {end_y:.2f}, {end_z:.2f})'
    ax1.text(end_x, end_y, end_z, end_text,
            color='black', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='orange', alpha=0.85, linewidth=1.0),
            ha='center', va='top', zorder=4)
    
    # Axis labels
    ax1.set_xlabel('$X$ (m)', fontsize=10, labelpad=5)
    ax1.set_ylabel('$Y$ (m)', fontsize=10, labelpad=5)
    ax1.set_zlabel('$Z$ (m)', fontsize=10, labelpad=5)
    
    # Legend
    legend1 = ax1.legend(loc='best', frameon=True, fancybox=False, 
                         edgecolor='black', framealpha=0.95, fontsize=10)
    legend1.get_frame().set_linewidth(0.8)
    
    # Axis limits
    ax1.set_xlim([min(rk45_pred['x'])-5, max(rk45_pred['x'])+5])
    ax1.set_ylim([min(rk45_pred['y'])-5, max(rk45_pred['y'])+5])
    ax1.set_zlim([min(rk45_pred['z'])-5, max(rk45_pred['z'])+5])
    
    # Grid lines
    ax1.xaxis._axinfo['grid'].update({'linewidth': 0.3, 'alpha': 0.2})
    ax1.yaxis._axinfo['grid'].update({'linewidth': 0.3, 'alpha': 0.2})
    ax1.zaxis._axinfo['grid'].update({'linewidth': 0.3, 'alpha': 0.2})
    
    # Background transparent
    #ax1.patch.set_facecolor('white')
    #ax1.patch.set_alpha(0.0)
    
    # View angle
    ax1.view_init(elev=25, azim=-60)
    
    # ==================== Right: Training loss ====================
    ax2 = fig.add_subplot(122)
    
    # Define Greek character labels for state variables
    state_labels = {
        'x': r'$x$',
        'y': r'$y$',
        'z': r'$z$',
        'v': r'$v$',
        'gamma': r'$\gamma$',
        'chi': r'$\chi$'
    }
    
    # Define academic color scheme
    state_names = ['x', 'y', 'z', 'v', 'gamma', 'chi']
    colors2 = ['#1f77b4', '#17becf', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    linewidths = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
    
    # Plot losses for each state variable
    for name, color, ls, lw in zip(state_names, colors2, linestyles, linewidths):
        if len(loss_history[name]) > 0:
            ax2.semilogy(loss_history[name], linestyle=ls, color=color, 
                        linewidth=lw, label=state_labels[name], alpha=0.95)
    
    # Plot total loss
    total_loss_data = None
    if len(loss_history['total']) > 0:
        total_loss_data = loss_history['total']
        ax2.semilogy(total_loss_data, 'k-', linewidth=2.0, 
                    label=r'$\mathcal{L}_{total}$', alpha=0.9, zorder=10)
    
    # Axis labels
    ax2.set_xlabel('Training Iterations', fontsize=12, fontweight='normal', labelpad=8)
    ax2.set_ylabel('Loss', fontsize=12, fontweight='normal', labelpad=8)
    
    # Legend
    legend2 = ax2.legend(loc='best', frameon=True, fancybox=False, 
                         edgecolor='black', framealpha=0.95, 
                         fontsize=10, ncol=1, columnspacing=1.0,
                         handlelength=2.0, handletextpad=0.8)
    legend2.get_frame().set_linewidth(0.8)
    
    # Grid
    ax2.grid(True, which='major', linestyle='-', linewidth=0.5, 
             alpha=0.3, color='gray')
    ax2.grid(True, which='minor', linestyle=':', linewidth=0.3, 
             alpha=0.15, color='gray')
    
    # Ticks
    ax2.tick_params(axis='both', which='major', direction='in', 
                    length=5, width=1.0, colors='black', top=True, right=True)
    ax2.tick_params(axis='both', which='minor', direction='in', 
                    length=3, width=0.8, colors='black', top=True, right=True)
    
    # Background color
    #ax2.set_facecolor('#f8f9fa')
    
    # ==================== Embedded zoomed view (loss curve late stage) ====================
    if total_loss_data is not None and len(total_loss_data) > 1000:
        # Select zoom region (late stage of training, after loss convergence)
        # Can adjust these parameters based on actual data
        iter_start = 4000   # Start iteration of zoom region
        iter_end = 5001     # End iteration of zoom region
        loss_min = 1e-1     # Minimum loss range
        loss_max = 1e2      # Maximum loss range
        
        # Create embedded subplot (in the upper right blank area of the main figure)
        # Parameters [left, bottom, width, height] use relative coordinates
        ax_zoom = fig.add_axes([0.62, 0.708, 0.2, 0.25]) 
        
        # Plot losses for each state variable in the zoom region
        for name, color, ls, lw in zip(state_names, colors2, linestyles, linewidths):
            if len(loss_history[name]) >= iter_end:
                zoom_data = loss_history[name][iter_start:iter_end]
                ax_zoom.semilogy(zoom_data, linestyle=ls, color=color, 
                                linewidth=lw, alpha=0.9)
        
        # Plot total loss in the zoom region
        zoom_total = total_loss_data[iter_start:iter_end]
        ax_zoom.semilogy(zoom_total, 'k-', linewidth=2.0, alpha=0.9, zorder=10)
        
        # Set coordinate range for zoom region
        x_range = list(range(iter_start, iter_end))
        ax_zoom.set_xlim(0, len(zoom_total) - 1)
        ax_zoom.set_ylim(loss_min, loss_max)
        
        # Set x-axis tick labels (display actual iteration numbers)
        def format_func(value, tick_number):
            return f'{int(iter_start + value)}'
        ax_zoom.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

        # ==================== Tick settings ====================
        # Y-axis: show only major ticks, hide minor ticks
        ax_zoom.tick_params(axis='y', which='major', direction='out', 
                    length=4, width=0.8, labelsize=9, colors='black')
        ax_zoom.tick_params(axis='y', which='minor', length=0)  # Hide minor ticks
        
        # Grid
        ax_zoom.grid(True, which='major', linestyle='-', linewidth=0.3, alpha=0.3)
        ax_zoom.grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.15)
        
        # Background semi-transparent
        ax_zoom.set_facecolor('white')
        ax_zoom.patch.set_alpha(0.9)
        
        # Draw rectangle on main figure to indicate zoom region
        rect_x = iter_start
        rect_width = iter_end - iter_start
        rect_y = loss_min
        rect_height = loss_max - loss_min
        
        from matplotlib.patches import Rectangle
        rect = Rectangle((rect_x, rect_y), rect_width, rect_height, 
                         linewidth=1.5, edgecolor='red', facecolor='none',
                         linestyle='-', alpha=0.97, zorder=100)
        ax2.add_patch(rect)
        
        # ==================== Draw two connection lines using ConnectionPatch ====================
        from matplotlib.patches import ConnectionPatch
        
        # Four vertices of the rectangle (data coordinates)
        rect_left = iter_start
        rect_right = iter_end
        rect_top = loss_max
        rect_bottom = loss_min
        
        # Boundaries of the zoomed view (convert relative coordinates to data coordinates)
        zoom_xlim = ax_zoom.get_xlim()
        zoom_ylim = ax_zoom.get_ylim()
        
        # Zoomed view lower left corner (data coordinates)
        zoom_left = zoom_xlim[0]
        zoom_right = zoom_xlim[1]
        zoom_bottom = zoom_ylim[0]
        zoom_top = zoom_ylim[1]

        # Use data coordinates for direct connection
        # Connection line 1: upper-left corner of rectangle → lower-left corner of zoomed view
        xy1_main = (rect_left, rect_top)
        xy1_zoom = (zoom_left, zoom_bottom)
        
        # Connection line 2: upper-right corner of rectangle → lower-right corner of zoomed view
        xy2_main = (rect_right, rect_top)
        xy2_zoom = (zoom_right, zoom_bottom)
        
        # Create ConnectionPatch objects
        # Connection line 1
        con1 = ConnectionPatch(
            xyA=xy1_zoom,
            xyB=xy1_main,
            coordsA='data',
            coordsB='data',
            axesA=ax_zoom,
            axesB=ax2,
            color='black',
            linewidth=0.8,
            linestyle='--',
            alpha=0.6,
            arrowstyle='-',
            zorder=5
        )

        # Connection line 2
        con2 = ConnectionPatch(
            xyA=xy2_zoom,
            xyB=xy2_main,
            coordsA='data',
            coordsB='data',
            axesA=ax_zoom,
            axesB=ax2,
            color='black',
            linewidth=0.8,
            linestyle='--',
            alpha=0.6,
            arrowstyle='-',
            zorder=5
        )
        
        # Add connection lines to figure
        fig.add_artist(con1)
        fig.add_artist(con2)
    
    plt.tight_layout()
    
    # Save image
    plt.savefig('PINNDBT_figure_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("\nSaved as PINNDBT_figure_combined.png")

def plot_figure_comparison(t_test, rk45_pred, pinn_pred):
    """
    Figure 2: 6 subplots showing x, y, z, v, gamma, chi predictions compared with RK45
    """
    state_names = ['x', 'y', 'z', 'v', 'gamma', 'chi']
    # display_names = ['X Position (m)', 'Y Position (m)', 'Z Position (m)', 
    #                  'Velocity (m/s)', 'Flight Path Angle (deg)', 'Heading Angle (deg)']
    display_names = [r'$x$ (m)', r'$y$ (m)', r'$z$ (m)', r'$v$ (m/s)', r'$\gamma$ (deg)', r'$\chi$ (deg)']
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    rk45_color = "#1f77b4"
    pinn_color = "#d62728"
    
    for idx, (name, display_name) in enumerate(zip(state_names, display_names)):
        ax = axes[idx]
        
        # Plot RK45 reference solution
        if name in ['gamma', 'chi']:
            ax.plot(t_test, np.degrees(rk45_pred[name]), color=rk45_color, linewidth=2.5, 
                   label='RK45', alpha=0.9)
            ax.plot(t_test, np.degrees(pinn_pred[name]), color=pinn_color, linewidth=2, linestyle='--', label='DBT-PINN', alpha=0.8)
        else:
            ax.plot(t_test, rk45_pred[name], color=rk45_color, linewidth=2.5, 
                   label='RK45', alpha=0.9)
            ax.plot(t_test, pinn_pred[name], color=pinn_color, linewidth=2, linestyle='--', label='DBT-PINN', alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel(display_name, fontsize=11)
        #ax.set_title(f'{display_name}', fontsize=12, fontweight='bold')
        #ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

    # ==========================================================
    # Unified legend
    # ==========================================================
    legend_handles = [
        Line2D(
            [0], [0],
            color=rk45_color,
            linewidth=2.5,
            label='RK45'
        ),
        Line2D(
            [0], [0],
            color=pinn_color,
            linewidth=2.0,
            linestyle='--',
            label='DBT-PINN'
        )
    ]

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        fontsize=10,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.5
    )

    plt.tight_layout(rect=[0, 0.045, 1, 1])
        
    #plt.tight_layout()
    plt.savefig('PINNDBT_figure_states_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nSaved as PINNDBT_figure_states_comparison.png")

def plot_figure_convergence(t_test, rk45_pred, checkpoint_predictions, checkpoint_t):
    """
    Figure 3: 6 subplots showing x, y, z, v, gamma, chi at different iteration counts compared with RK45
    Note: checkpoint_t is the time points during training, t_test is the test time points
    """
    state_names = ['x', 'y', 'z', 'v', 'gamma', 'chi']
    display_names = [r'$x$ (m)', r'$y$ (m)', r'$z$ (m)', r'$v$ (m/s)', r'$\gamma$ (deg)', r'$\chi$ (deg)']
    
    # Get iteration counts and sort
    iterations = sorted(checkpoint_predictions.keys())
    colors = ['orange', 'green', 'blue', 'purple', 'red']
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()
    
    for idx, (name, display_name) in enumerate(zip(state_names, display_names)):
        ax = axes[idx]
        
        # Plot RK45 reference solution (using t_test)
        if name in ['gamma', 'chi']:
            ax.plot(t_test, np.degrees(rk45_pred[name]), 'k-', linewidth=3, 
                   label='RK45', alpha=0.9)
        else:
            ax.plot(t_test, rk45_pred[name], 'k-', linewidth=3, 
                   label='RK45', alpha=0.9)
        
        # Plot predictions at different iteration counts (using checkpoint_t)
        for i, iter_num in enumerate(iterations):
            pred = checkpoint_predictions[iter_num]
            if name in ['gamma', 'chi']:
                # Ensure length matching, interpolate if checkpoint_t is longer than t_test
                if len(checkpoint_t) > len(t_test):
                    pred_interp = np.interp(t_test, checkpoint_t, np.degrees(pred[name]))
                    ax.plot(t_test, pred_interp, '--', color=colors[i], linewidth=1.8,
                           label=f'Iteration {iter_num}', alpha=0.8)
                else:
                    ax.plot(checkpoint_t, np.degrees(pred[name]), '--', color=colors[i], linewidth=1.8,
                           label=f'Iteration {iter_num}', alpha=0.8)
            else:
                # Interpolate predictions to t_test time points
                if len(checkpoint_t) > len(t_test):
                    pred_interp = np.interp(t_test, checkpoint_t, pred[name])
                    ax.plot(t_test, pred_interp, '--', color=colors[i], linewidth=1.8,
                           label=f'Iteration {iter_num}', alpha=0.8)
                else:
                    ax.plot(checkpoint_t, pred[name], '--', color=colors[i], linewidth=1.8,
                           label=f'Iteration {iter_num}', alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(display_name, fontsize=12)
        #ax.set_title(f'{display_name}', fontsize=12, fontweight='bold')
        #ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Create legend handles manually
    legend_handles = []
    legend_labels = []
    
    # RK45 reference
    rk45_handle, = axes[0].plot(
        [],
        [],
        'k-',
        linewidth=3,
        alpha=0.9
    )
    legend_handles.append(rk45_handle)
    legend_labels.append('RK45')
    
    # Iteration curves
    for i, iter_num in enumerate(iterations):
        handle, = axes[0].plot(
            [],
            [],
            '--',
            color=colors[i],
            linewidth=1.8,
            alpha=0.8
        )
        legend_handles.append(handle)
        legend_labels.append(f'Iteration {iter_num}')

    # ================================================================
    # Global legend at the bottom
    # ================================================================
    fig.legend(
        legend_handles,
        legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(legend_labels),
        fontsize=10,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.5
    )
    
    # Leave space at the bottom for the shared legend
    plt.tight_layout(rect=[0, 0.045, 1, 1])

    #plt.tight_layout()
    plt.savefig('PINNDBT_figure_iteration_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nSaved as PINNDBT_figure_iteration_convergence.png")

# =============================================================
# Main program
# =============================================================
def main():
    # Parameter settings
    t_start, t_end = 0.0, 10.0
    n_points = 500  # Training sample points
    n_test = 50     # Test points (can be changed to 50 or 500)
    t_test = np.linspace(t_start, t_end, n_test)  # Test points
    
    # Initial state [x, y, z, v, gamma, chi]
    initial_state = np.array([0.0, 0.0, 0.0, 200.0, np.pi/3, np.pi/3])
    
    # =========================================================
    # Custom network structure configuration
    # =========================================================
    network_configs = {
        'x': [256, 256, 256],
        'y': [256, 256, 256],
        'z': [256, 256, 256],
        'v': [128, 128, 128],
        'gamma': [128, 128, 128],
        'chi': [128, 128, 128]
    }
    
    print("="*80)
    print("Aircraft 3-DOF Model - Iterative PINN Solver (Heterogeneous Network Structure)")
    print("="*80)
    print(f"Time range: [{t_start}, {t_end}]")
    print(f"Training sample points: {n_points}")
    print(f"Test points: {n_test}")
    print(f"Initial state: x={initial_state[0]}, y={initial_state[1]}, z={initial_state[2]}, "
          f"v={initial_state[3]}, gamma={np.degrees(initial_state[4]):.1f} deg, chi={np.degrees(initial_state[5]):.1f} deg")
    print(f"Control inputs: nx={nx}, nz={nz}, mu={np.degrees(mu):.1f} deg")
    print("="*80)
    
    # =========================================================
    # Iterative PINN solution
    # =========================================================
    solver = IterativePINN(t_start, t_end, n_points, network_configs=network_configs)
    iterations_history = solver.iterative_train(n_iterations=5001, initial_steps_per_iteration=1)
    
    # Final prediction (using t_test time points)
    pinn_pred = solver.predict(t_test)
    
    # =========================================================
    # RK45 reference solution (using t_test time points)
    # =========================================================
    print("\nComputing RK45 reference solution...")
    rk45_solution = solve_with_rk45(t_start, t_end, initial_state, t_test)
    rk45_pred = {
        'x': rk45_solution[0], 'y': rk45_solution[1], 'z': rk45_solution[2],
        'v': rk45_solution[3], 'gamma': rk45_solution[4], 'chi': rk45_solution[5]
    }
    
    # =========================================================
    # Generate figures
    # =========================================================
    print("\n" + "="*80)
    print("Generating comparison figures...")
    print("="*80)
    
    # Figure 1: 3D trajectory + loss trajectory
    plot_figure_combined(rk45_pred, pinn_pred, solver.loss_history)
    
    # Figure 2: All state variable comparisons
    plot_figure_comparison(t_test, rk45_pred, pinn_pred)
    
    # Figure 3: Different iteration comparisons
    if len(solver.checkpoint_predictions) >= 3:
        # Get training time points
        checkpoint_t = solver.checkpoint_t.flatten() if solver.checkpoint_t is not None else np.linspace(t_start, t_end, n_points)
        plot_figure_convergence(t_test, rk45_pred, solver.checkpoint_predictions, checkpoint_t)
    else:
        print("\nWarning: Complete iteration checkpoints (500,1000,1500) not found")
    
    # =========================================================
    # Error analysis
    # =========================================================
    print("\n" + "="*80)
    print("Final Error Analysis (vs RK45)")
    print("="*80)
    
    for name in ['x', 'y', 'z', 'v', 'gamma', 'chi']:
        if name in ['gamma', 'chi']:
            error = np.abs(np.degrees(pinn_pred[name]) - np.degrees(rk45_pred[name]))
            unit = 'deg'
        else:
            error = np.abs(pinn_pred[name] - rk45_pred[name])
            unit = 'm' if name in ['x', 'y', 'z'] else 'm/s'
        print(f"{name.upper():6s} ({unit}): Max={np.max(error):.6e}, Mean={np.mean(error):.6e}, RMSE={np.sqrt(np.mean(error**2)):.6e}")
    
    # Print final state
    print("\n" + "="*80)
    print("Final State (t=10s)")
    print("="*80)
    print(f"{'Variable':<12} {'RK45':<15} {'Iterative PINN':<15} {'Error':<15}")
    print("-"*60)
    for name in ['x', 'y', 'z', 'v', 'gamma', 'chi']:
        error = np.abs(pinn_pred[name][-1] - rk45_pred[name][-1])
        unit = 'm' if name in ['x', 'y', 'z'] else ('m/s' if name == 'v' else 'rad')
        if name in ['gamma', 'chi']:
            print(f"{name:>3} (rad):  {rk45_pred[name][-1]:<15.4f} {pinn_pred[name][-1]:<15.4f} {error:<15.4e}")
            print(f"{name:>3} (deg):  {np.degrees(rk45_pred[name][-1]):<15.2f} {np.degrees(pinn_pred[name][-1]):<15.2f} {np.degrees(error):<15.2e}")
        else:
            print(f"{name:>3} ({unit}):  {rk45_pred[name][-1]:<15.2f} {pinn_pred[name][-1]:<15.2f} {error:<15.2e}")

if __name__ == "__main__":
    main()