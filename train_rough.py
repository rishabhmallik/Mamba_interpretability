import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import Tuple, List
import os
from torch.amp import autocast, GradScaler
from data_modadd import (
    generate_modular_addition_dataset,
    generate_balanced_modular_dataset
)

# Import the dataset generation functions (assuming they're available)
# If mamba_ssm is not available, we'll create a simple substitute
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("mamba_ssm not available, using simple RNN substitute")

# Simple Mamba substitute if the library is not available
class SimpleMambaSubstitute(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        
        # Simple LSTM-based substitute with optimizations
        self.lstm = nn.LSTM(d_model, d_model * expand, batch_first=True, dropout=0.1)
        self.output_proj = nn.Linear(d_model * expand, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
    def forward(self, x):
        # x shape: (batch, length, d_model)
        lstm_out, _ = self.lstm(x)
        output = self.output_proj(lstm_out)
        return self.norm(output + x)  # Residual connection

# Optimized Mamba-based model for modular addition
class MambaModularAddition(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding layer with better initialization
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=None)
        nn.init.normal_(self.embedding.weight, 0, 0.1)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) if MAMBA_AVAILABLE
            else SimpleMambaSubstitute(d_model=d_model)
            for _ in range(n_layers)
        ])
        
        # Output head with layer norm for stability
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize output head
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        
    def forward(self, x):
        # x shape: (batch, sequence_length)
        # Embed inputs
        x = self.embedding(x)  # (batch, seq_len, d_model)
        
        # Pass through Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        # Only use the last position for prediction
        last_hidden = x[:, -1, :]  # (batch, d_model)
        #last_hidden = self.norm(last_hidden)
        #last_hidden = self.dropout(last_hidden)
        
        # Output projection
        logits = self.output_head(last_hidden)  # (batch, vocab_size)
        
        return logits

def create_optimized_dataloader(X: np.ndarray, Y: np.ndarray, batch_size: int = 32, 
                               shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """Optimized dataloader with proper num_workers and pin_memory"""
    X_tensor = torch.from_numpy(X).long()
    Y_tensor = torch.from_numpy(Y).long()
    dataset = TensorDataset(X_tensor, Y_tensor)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=min(num_workers, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )

def train_epoch_optimized(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, 
                         criterion: nn.Module, device: torch.device, scaler=None) -> Tuple[float, float]:
    """Optimized training loop with reduced GPU-CPU synchronization"""
    model.train()
    
    # Use running metrics to avoid frequent .item() calls
    running_loss = 0.0
    running_correct = 0
    num_samples = 0
    
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
        batch_size = batch_x.size(0)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        if scaler is not None:
            with autocast(device_type=device.type):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Accumulate metrics (reduce GPU-CPU sync)
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            running_loss += loss.item() * batch_size
            running_correct += (predictions == batch_y).sum().item()
            num_samples += batch_size
    
    avg_loss = running_loss / num_samples
    accuracy = running_correct / num_samples
    
    return avg_loss, accuracy

@torch.no_grad()
def evaluate_optimized(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                      device: torch.device) -> Tuple[float, float]:
    """Optimized evaluation with no_grad and reduced synchronization"""
    model.eval()
    
    running_loss = 0.0
    running_correct = 0
    num_samples = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
        batch_size = batch_x.size(0)
        
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        predictions = torch.argmax(logits, dim=1)
        
        running_loss += loss.item() * batch_size
        running_correct += (predictions == batch_y).sum().item()
        num_samples += batch_size
    
    avg_loss = running_loss / num_samples
    accuracy = running_correct / num_samples
    
    return avg_loss, accuracy

def plot_training_curves(train_losses: List[float], train_accuracies: List[float],
                        test_losses: List[float], test_accuracies: List[float],
                        save_path: str = None):
    """Optimized plotting function"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7, linewidth=1)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Accuracy curves
    ax2.plot(epochs, train_accuracies, 'b-', label='Train Accuracy', alpha=0.7, linewidth=1)
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', alpha=0.7, linewidth=1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Optimized configuration
    config = {
        'base': 20,  # Modulo base
        'sequence_length': 4,
        'n_samples_train': 60000,
        'n_samples_test': 20000,
        'd_model': 128,
        'n_layers': 4,
        'batch_size': 256,  # Larger batch size for better GPU utilization
        'learning_rate': 3e-4,  # Adjusted for larger batch size
        'weight_decay': 1e-1,  # Reduced weight decay
        'dropout': 0.1,
        'epochs': 5000,
        'seed': 28,
        'num_workers': 4,  # For faster data loading
        'eval_every': 25,  # Evaluate less frequently to save time
    }
    
    print("Optimized Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
    
    # Device and optimization settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enable optimizations for better performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul on A100
        torch.backends.cudnn.allow_tf32 = True  # Faster convolutions on A100
    
    # Initialize mixed precision scaler
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    print(f"Using automatic mixed precision: {use_amp}")
    
    # Generate datasets (moved to GPU immediately if available)
    print("\nGenerating datasets...")
    X_train, Y_train = generate_modular_addition_dataset(
        config['n_samples_train'],
        config['sequence_length'], 
        config['base'], 
        seed=config['seed']
    )

    X_test, Y_test = generate_modular_addition_dataset(
        config['n_samples_test'],
        config['sequence_length'],
        config['base'],
        seed=config['seed'] + 1
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create optimized dataloaders
    train_loader = create_optimized_dataloader(
        X_train, Y_train, 
        config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers']
    )
    test_loader = create_optimized_dataloader(
        X_test, Y_test, 
        config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Initialize model with optimizations
    print(f"\nInitializing model ({'Mamba' if MAMBA_AVAILABLE else 'LSTM substitute'})...")
    model = MambaModularAddition(
        vocab_size=config['base'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Skip torch.compile for Mamba models due to custom CUDA kernels
    if hasattr(torch, 'compile') and torch.cuda.is_available() and not MAMBA_AVAILABLE:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled with torch.compile for optimization")
        except Exception as e:
            print(f"Could not compile model: {e}")
    elif MAMBA_AVAILABLE:
        print("Skipping torch.compile for Mamba (incompatible with custom CUDA kernels)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Initialize optimizer with optimized settings
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        eps=1e-6,  # More stable for mixed precision
        fused=torch.cuda.is_available()  # Faster fused AdamW on GPU
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for regularization
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['learning_rate'],
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader)
    )
    
    # Training loop with optimizations
    print(f"\nStarting optimized training for {config['epochs']} epochs...")
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in tqdm(range(config['epochs']), desc="Training", dynamic_ncols=True):
        # Train
        train_loss, train_acc = train_epoch_optimized(
            model, train_loader, optimizer, criterion, device, scaler
        )
        
        # Step scheduler
        if hasattr(scheduler, 'step'):
            scheduler.step()
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Evaluate less frequently to save time
        if (epoch + 1) % config['eval_every'] == 0 or epoch == config['epochs'] - 1:
            test_loss, test_acc = evaluate_optimized(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            
            print(f"Epoch {epoch+1:3d}: Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}, "
                  f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Final evaluation
    final_test_loss, final_test_acc = evaluate_optimized(model, test_loader, criterion, device)
    
    print(f"\nTraining completed!")
    print(f"Final train accuracy: {train_accuracies[-1]:.3f}")
    print(f"Final test accuracy: {final_test_acc:.3f}")
    
    # Interpolate test metrics for plotting (since we evaluated less frequently)
    if len(test_losses) < len(train_losses):
        eval_epochs = list(range(config['eval_every'] - 1, config['epochs'], config['eval_every']))
        if config['epochs'] - 1 not in eval_epochs:
            eval_epochs.append(config['epochs'] - 1)
        
        # Simple interpolation for visualization
        test_losses_interp = np.interp(range(config['epochs']), eval_epochs, test_losses)
        test_accuracies_interp = np.interp(range(config['epochs']), eval_epochs, test_accuracies)
        test_losses = test_losses_interp.tolist()
        test_accuracies = test_accuracies_interp.tolist()
    
    # Plot results
    print("\nGenerating training curves...")
    plot_training_curves(train_losses, train_accuracies, test_losses, test_accuracies,
                         save_path='optimized_mamba_training_curves.png')
    
    # Save results
    results = {
        'config': config,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'final_train_acc': train_accuracies[-1],
        'final_test_acc': final_test_acc,
        'total_params': total_params
    }
    
    with open('optimized_mamba_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to 'optimized_mamba_results.json'")
    print("Training curves saved to 'optimized_mamba_training_curves.png'")

if __name__ == "__main__":
    main()