"""
Project 08: Visualizing Gradients Using PyTorch Hooks
Framework: PyTorch
Author: Gideon Mwiti

Key Features:
- Forward and Backward Hooks using torch.nn.Module.register_forward_hook 
  and torch.nn.Module.register_full_backward_hook.
- Systematic gradient flow tracker across deep architectures (6-layer MLP).
- Vanishing Gradient diagnosis: Controlled comparison between Deep Sigmoid 
  (vanishing gradient pathology) and Deep ReLU (healthy gradient propagation).
- Layer-wise L2 gradient norm monitoring across training iterations.
- High-resolution 4-panel diagnostic visual artifact:
  1) Dynamic Layer-wise Gradient Norm Flow over Optimization Steps
  2) Vanishing vs. Healthy Gradient Flow Comparison (Sigmoid vs. ReLU)
  3) Intermediate Layer Gradient Distribution Boxplots
  4) Forward Activations vs. Backward Gradients Dual-Hook Correlation
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GradientHookTracker:
    """
    Diagnostic utility that registers forward and full backward hooks
    on neural network modules to capture layer activations and gradients.
    """
    def __init__(self, model, track_layers=None):
        self.model = model
        self.forward_hooks = []
        self.backward_hooks = []
        
        # Storage containers
        self.activations = {}      # layer_name -> list of np.ndarray
        self.grad_inputs = {}       # layer_name -> list of np.ndarray
        self.grad_outputs = {}      # layer_name -> list of np.ndarray
        self.grad_norms = {}        # layer_name -> list of float (L2 norm)
        
        self._register(track_layers)

    def _register(self, track_layers):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if track_layers is None or name in track_layers:
                    self.activations[name] = []
                    self.grad_inputs[name] = []
                    self.grad_outputs[name] = []
                    self.grad_norms[name] = []
                    
                    # Forward hook: captures intermediate representation
                    def get_fwd_hook(layer_name):
                        def hook(mod, inp, out):
                            self.activations[layer_name].append(out.detach().cpu().numpy())
                        return hook
                    
                    # Backward hook: captures flowing gradient tensor
                    def get_bwd_hook(layer_name):
                        def hook(mod, grad_in, grad_out):
                            # grad_out is a tuple containing the gradient wrt module output
                            if grad_out[0] is not None:
                                g_out = grad_out[0].detach()
                                norm = torch.norm(g_out, p=2).item()
                                self.grad_norms[layer_name].append(norm)
                                self.grad_outputs[layer_name].append(g_out.cpu().numpy())
                            if grad_in[0] is not None:
                                self.grad_inputs[layer_name].append(grad_in[0].detach().cpu().numpy())
                        return hook
                    
                    h_fwd = module.register_forward_hook(get_fwd_hook(name))
                    h_bwd = module.register_full_backward_hook(get_bwd_hook(name))
                    self.forward_hooks.append(h_fwd)
                    self.backward_hooks.append(h_bwd)

    def remove_hooks(self):
        """Cleanly deregister hooks to prevent memory leaks."""
        for h in self.forward_hooks:
            h.remove()
        for h in self.backward_hooks:
            h.remove()
        self.forward_hooks.clear()
        self.backward_hooks.clear()

    def clear_step_cache(self):
        """Retain norms over time, but clear heavy raw tensors if needed."""
        self.activations = {k: [] for k in self.activations}
        self.grad_inputs = {k: [] for k in self.grad_inputs}
        self.grad_outputs = {k: [] for k in self.grad_outputs}


class DeepMLP(nn.Module):
    """
    Configurable 6-Layer Deep Multi-Layer Perceptron.
    Supports 'relu' and 'sigmoid' activations to demonstrate gradient flow dynamics.
    """
    def __init__(self, input_dim=64, hidden_dim=64, num_classes=10, activation='relu'):
        super().__init__()
        act_fn = nn.ReLU if activation.lower() == 'relu' else nn.Sigmoid
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = act_fn()
        
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = act_fn()
        
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.act3 = act_fn()
        
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.act4 = act_fn()
        
        self.layer5 = nn.Linear(hidden_dim, hidden_dim)
        self.act5 = act_fn()
        
        self.layer6 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act4(self.layer4(x))
        x = self.act5(self.layer5(x))
        x = self.layer6(x)
        return x


def generate_synthetic_dataset(num_samples=2000, input_dim=64, num_classes=10, seed=42):
    """Generates a reproducible non-linear classification dataset."""
    np.random.seed(seed)
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    
    # Non-linear projection to generate discrete classes
    W_proj = np.random.randn(input_dim, num_classes).astype(np.float32)
    logits = np.tanh(X @ W_proj) + 0.1 * np.random.randn(num_samples, num_classes)
    y = np.argmax(logits, axis=1).astype(np.int64)
    
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    
    # Split 80/20 train/test
    split = int(0.8 * num_samples)
    train_dataset = TensorDataset(X_tensor[:split], y_tensor[:split])
    test_dataset = TensorDataset(X_tensor[split:], y_tensor[split:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader


def train_with_gradient_tracking(model, train_loader, epochs=5, lr=0.01):
    """Trains a model while recording gradient flow through hooks."""
    tracker = GradientHookTracker(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    step_count = 0
    print(f"\n--- Training with Hooks (Capturing Layer Gradients in Backward Pass) ---")
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # During loss.backward(), all registered backward hooks trigger automatically!
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
            step_count += 1
            
        epoch_loss = total_loss / len(train_loader.dataset)
        
        # Print gradient norms for each layer at end of epoch
        norms_str = " | ".join([f"{k}: {np.mean(v[-len(train_loader):]):.4f}" for k, v in tracker.grad_norms.items()])
        print(f"Epoch {epoch:02d}/{epochs:02d} (Loss: {epoch_loss:.4f}) | Grad Norms -> {norms_str}")
        
    return tracker


def generate_visual_artifact(tracker_relu, tracker_sigmoid, save_path):
    """Creates a comprehensive 4-panel diagnostic visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)
    
    layer_names = list(tracker_relu.grad_norms.keys())
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # -------------------------------------------------------------
    # Panel 1: Dynamic Layer-wise Gradient Norm Flow over Training Steps (ReLU)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    total_steps = len(tracker_relu.grad_norms[layer_names[0]])
    steps = np.arange(1, total_steps + 1)
    
    for i, name in enumerate(layer_names):
        # Smoothed moving average for visual clarity
        raw_norms = tracker_relu.grad_norms[name]
        window = 10
        smoothed = np.convolve(raw_norms, np.ones(window)/window, mode='valid')
        smoothed_steps = steps[:len(smoothed)]
        ax1.plot(smoothed_steps, smoothed, label=f"{name} ({name.replace('layer', 'L')})", color=palette[i % len(palette)], linewidth=2.0)
        
    ax1.set_title("Layer-wise Gradient Flow over Training Iterations (ReLU)\nCaptured via torch.nn.Module.register_full_backward_hook", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Optimization Step (Mini-Batch)", fontsize=10)
    ax1.set_ylabel("L2 Gradient Norm (grad_output)", fontsize=10)
    ax1.set_yscale('log')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    
    # -------------------------------------------------------------
    # Panel 2: Vanishing Gradient Pathology (Sigmoid vs. ReLU)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    mean_norms_relu = [np.mean(tracker_relu.grad_norms[name]) for name in layer_names]
    mean_norms_sigmoid = [np.mean(tracker_sigmoid.grad_norms[name]) for name in layer_names]
    
    x = np.arange(len(layer_names))
    width = 0.35
    
    rects1 = ax2.bar(x - width/2, mean_norms_relu, width, label='Deep ReLU (Healthy Flow)', color='#2ca02c', alpha=0.85)
    rects2 = ax2.bar(x + width/2, mean_norms_sigmoid, width, label='Deep Sigmoid (Vanishing Flow)', color='#d62728', alpha=0.85)
    
    ax2.set_title("Vanishing Gradient Diagnostics: Deep ReLU vs. Deep Sigmoid\nGradient Magnitude Decreases Drastically in Early Layers of Sigmoid", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Network Layer (Early Input -> Deep Output)", fontsize=10)
    ax2.set_ylabel("Mean L2 Gradient Norm", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Layer {i+1}" for i in range(len(layer_names))], fontsize=9)
    ax2.set_yscale('log')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', frameon=True, fontsize=9)
    
    # Annotate ratio on early layer
    if mean_norms_sigmoid[0] > 0:
        ratio = mean_norms_relu[0] / mean_norms_sigmoid[0]
        ax2.annotate(f"{ratio:.0f}x stronger in Layer 1",
                     xy=(0 - width/2, mean_norms_relu[0]),
                     xytext=(0, mean_norms_relu[0] * 3),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                     fontsize=9, fontweight='bold', ha='center')

    # -------------------------------------------------------------
    # Panel 3: Layer Gradient Value Distribution Boxplots (ReLU)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    grad_samples = []
    for name in layer_names:
        # Sample raw gradient elements from the last recorded backward pass
        if len(tracker_relu.grad_outputs[name]) > 0:
            last_grad = tracker_relu.grad_outputs[name][-1].flatten()
            grad_samples.append(last_grad)
        else:
            grad_samples.append(np.array([0.0]))
            
    bplot = ax3.boxplot(grad_samples, patch_artist=True,
                        showfliers=False, medianprops=dict(color='black', linewidth=1.5))
    ax3.set_xticks(range(1, len(layer_names) + 1))
    ax3.set_xticklabels([f"L{i+1}" for i in range(len(layer_names))])
    for patch, color in zip(bplot['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    ax3.set_title("Gradient Value Dispersion Across Layers (Interquartile Range)\nVerifies Symmetrical and Non-Zero Gradient Spread", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Layer Depth", fontsize=10)
    ax3.set_ylabel("Gradient Tensor Values (Raw Output Gradient)", fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # -------------------------------------------------------------
    # Panel 4: Forward Activations vs Backward Gradients (Dual-Hook Inspection)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    # Extract mean absolute activation vs mean absolute gradient for each layer
    mean_acts = [np.mean(np.abs(tracker_relu.activations[name][-1])) for name in layer_names]
    mean_grads = [np.mean(np.abs(tracker_relu.grad_outputs[name][-1])) for name in layer_names]
    
    for i, name in enumerate(layer_names):
        ax4.scatter(mean_acts[i], mean_grads[i], s=200, color=palette[i % len(palette)], 
                    label=f"Layer {i+1}", edgecolors='black', linewidth=1.5, zorder=5)
        ax4.annotate(f" L{i+1}", (mean_acts[i], mean_grads[i]), fontsize=9, fontweight='bold', va='center')
        
    ax4.set_title("Dual-Hook Inspection: Forward Activation vs. Backward Gradient\nTracking Representation Energy vs. Backpropagated Signal Strength", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Mean Absolute Activation Magnitude (|a| from Forward Hook)", fontsize=10)
    ax4.set_ylabel("Mean Absolute Gradient Magnitude (|g| from Backward Hook)", fontsize=10)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='lower right', frameon=True, fontsize=9)

    plt.suptitle("PyTorch Hook Diagnostics: Forward Activations & Backward Gradient Flow Analysis", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisual diagnostic report saved successfully at: {os.path.abspath(save_path)}")


def main():
    print("=" * 68)
    print("      PROJECT 08: VISUALIZING GRADIENTS USING PYTORCH HOOKS")
    print("=" * 68)
    
    set_seed(42)
    device = torch.device("cpu")
    print(f"Execution Device    : {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "gradient_flow_and_hooks_analysis.png")
    
    # 1. Generate Synthetic Multi-Class Dataset
    train_loader, test_loader = generate_synthetic_dataset(num_samples=2000, input_dim=64, num_classes=10, seed=42)
    print(f"Dataset             : Synthetic 64-dim Feature Manifold (2,000 samples, 10 classes)")
    print(f"Architecture        : 6-Layer Deep MLP (64 -> 64 -> 64 -> 64 -> 64 -> 64 -> 10)")
    print(f"Hook Mechanisms     : register_forward_hook & register_full_backward_hook")
    print("-" * 68)
    
    # 2. Experiment A: Deep ReLU Network (Healthy Gradient Flow)
    print("\n[Experiment 1] Deep ReLU Network (Healthy Flow Baseline)")
    set_seed(42)
    model_relu = DeepMLP(input_dim=64, hidden_dim=64, num_classes=10, activation='relu').to(device)
    tracker_relu = train_with_gradient_tracking(model_relu, train_loader, epochs=5, lr=0.01)
    
    # 3. Experiment B: Deep Sigmoid Network (Vanishing Gradient Pathology)
    print("\n[Experiment 2] Deep Sigmoid Network (Vanishing Gradient Pathology)")
    set_seed(42)
    model_sigmoid = DeepMLP(input_dim=64, hidden_dim=64, num_classes=10, activation='sigmoid').to(device)
    tracker_sigmoid = train_with_gradient_tracking(model_sigmoid, train_loader, epochs=5, lr=0.01)
    
    # 4. Summary Diagnostics
    layer_names = list(tracker_relu.grad_norms.keys())
    print("\n" + "=" * 68)
    print("              GRADIENT FLOW COMPARISON (MEAN L2 NORM)")
    print("=" * 68)
    print(f" Layer              | Deep ReLU (Healthy) | Deep Sigmoid (Pathology) | Ratio (ReLU/Sig)")
    print("-" * 68)
    for name in layer_names:
        norm_r = np.mean(tracker_relu.grad_norms[name])
        norm_s = np.mean(tracker_sigmoid.grad_norms[name])
        ratio = norm_r / norm_s if norm_s > 0 else float('inf')
        print(f" {name:<18} | {norm_r:>19.5f} | {norm_s:>24.5f} | {ratio:>15.1f}x")
    print("=" * 68)
    
    # 5. Generate Visual Artifact
    generate_visual_artifact(tracker_relu, tracker_sigmoid, save_path)
    
    # Clean up hooks
    tracker_relu.remove_hooks()
    tracker_sigmoid.remove_hooks()
    print("\nHooks removed cleanly. Execution complete.")


if __name__ == "__main__":
    main()

