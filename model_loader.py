import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from transformers import MambaConfig, MambaForCausalLM

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
        
        # Simple LSTM-based substitute
        self.lstm = nn.LSTM(d_model, d_model * expand, batch_first=True)
        self.output_proj = nn.Linear(d_model * expand, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (batch, length, d_model)
        lstm_out, _ = self.lstm(x)
        output = self.output_proj(lstm_out)
        return self.norm(output + x)  # Residual connection


class MambaModularAddition(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # RMSNorm layers - one before each Mamba layer
        self.pre_norms = nn.ModuleList([
            nn.RMSNorm(d_model) for _ in range(n_layers)
        ])
        
        # Mamba layers
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) if MAMBA_AVAILABLE
            else SimpleMambaSubstitute(d_model=d_model)
            for _ in range(n_layers)
        ])
        
        # Final RMSNorm after all layers
        self.final_norm = nn.RMSNorm(d_model)
        
        # Output head - only predicts at the last position
        self.output_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch, sequence_length)
        batch_size, seq_len = x.shape
        
        # Embed inputs
        x = self.embedding(x)  # (batch, seq_len, d_model)
        
        # Pass through Mamba layers with pre-normalization
        for pre_norm, layer in zip(self.pre_norms, self.layers):
            # Pre-normalize, then apply Mamba layer with residual connection
            normed_x = pre_norm(x)
            x = x + layer(normed_x)  # Residual connection
        
        # Final normalization after all layers
        x = self.final_norm(x)
        
        # Only use the last position for prediction
        last_hidden = x[:, -1, :]  # (batch, d_model)
        last_hidden = self.dropout(last_hidden)
        
        # Output projection
        logits = self.output_head(last_hidden)  # (batch, vocab_size)
        
        return logits

'''
# Mamba-based model for modular addition
class MambaModularAddition(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) if MAMBA_AVAILABLE
            else SimpleMambaSubstitute(d_model=d_model)
            for _ in range(n_layers)
        ])
        
        # Output head - only predicts at the last position
        self.output_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch, sequence_length)
        batch_size, seq_len = x.shape
        
        # Embed inputs
        x = self.embedding(x)  # (batch, seq_len, d_model)
        
        # Pass through Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        # Only use the last position for prediction
        last_hidden = x[:, -1, :]  # (batch, d_model)
        
        # Output projection
        logits = self.output_head(last_hidden)  # (batch, vocab_size)
        
        return logits
'''
class Mamba_hugginface(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 d_model=128, 
                 n_layers=4, 
                 pad_token_id=0, 
                 state_size=16,
                 expand=2,
                 d_conv=4):

        super(Mamba_hugginface, self).__init__()

        # Create Mamba config
        self.config = MambaConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=n_layers,
            pad_token_id=pad_token_id,
            state_size=state_size,
            expand=expand,
            conv_kernel=d_conv,
            use_mambapy=False
        )
        
        # Initialize Mamba model
        self.mamba = MambaForCausalLM(self.config)
        
    def forward(self, input_ids, attention_mask=None):
        # Forward pass through Mamba
        outputs = self.mamba(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        return outputs.logits[:, -1]


def force_sequential_training(model):
    for layer in model.mamba.backbone.layers:
        mixer = layer.mixer
        # Disable mambapy
        mixer.use_mambapy = False
        
        # Force slow path by making fast path unavailable
        original_forward = mixer.forward
        def sequential_forward(hidden_states, cache_params=None, cache_position=None, attention_mask=None):
            # Always use slow_forward, never cuda_kernels_forward
            return mixer.slow_forward(hidden_states, cache_params, cache_position, attention_mask)
        
        mixer.forward = sequential_forward
        
def create_model(config: Dict[str, Any], 
                 device: torch.device,
                 sequential: bool = False,
                 huggingface_model: bool = False) -> nn.Module:
    """
    Create and initialize a Mamba model based on configuration.
    
    Args:
        config: Dictionary containing model configuration
        device: Device to move the model to
        sequential: Whether to use sequential training

    Returns:
        Initialized model moved to the specified device
    """
    print(f"\nInitializing model ({'Mamba' if MAMBA_AVAILABLE else 'LSTM substitute'})...")
    if not huggingface_model:
        model = MambaModularAddition(
            vocab_size=config['base'],
            d_model=config['d_model'],
            n_layers=config['n_layers']
        ).to(device)
    else:
        if sequential:
            model = Mamba_hugginface(
                vocab_size=config['base'],
                d_model=config['d_model'],
                n_layers=config['n_layers'],
                pad_token_id=config['base']
            ).to(device)
            force_sequential_training(model)
        else:
            model = Mamba_hugginface(
                vocab_size=config['base'],
                d_model=config['d_model'],
                n_layers=config['n_layers'],
                pad_token_id=config['base']
            ).to(device)
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    return model

def load_model_weights(model: nn.Module, weights_path: str, device: torch.device) -> nn.Module:
    """
    Load pre-trained weights into the model.
    
    Args:
        model: The model to load weights into
        weights_path: Path to the saved model weights
        device: Device to load the weights to
    
    Returns:
        Model with loaded weights
    """
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model weights from {weights_path}")
            if 'epoch' in checkpoint:
                print(f"Checkpoint was saved at epoch {checkpoint['epoch']}")
        else:
            # Assume the checkpoint contains only the state dict
            model.load_state_dict(checkpoint)
            print(f"Loaded model state dict from {weights_path}")
            
    except FileNotFoundError:
        print(f"Warning: Weights file {weights_path} not found. Using randomly initialized weights.")
    except Exception as e:
        print(f"Warning: Error loading weights from {weights_path}: {e}")
        print("Using randomly initialized weights.")
    
    return model

def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                         epoch: int, loss: float, save_path: str,
                         config: Optional[Dict[str, Any]] = None):
    """
    Save a model checkpoint including model weights, optimizer state, and metadata.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        save_path: Path to save the checkpoint
        config: Optional configuration dictionary to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_full_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                        checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """
    Load a full checkpoint including model weights, optimizer state, and metadata.
    
    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        checkpoint_path: Path to the checkpoint file
        device: Device to load to
    
    Returns:
        Dictionary containing checkpoint metadata (epoch, loss, etc.)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded full checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {checkpoint.get('epoch', 'unknown')}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', float('inf')),
            'config': checkpoint.get('config', {})
        }
        
    except FileNotFoundError:
        print(f"Checkpoint file {checkpoint_path} not found. Starting from scratch.")
        return {'epoch': 0, 'loss': float('inf'), 'config': {}}
    except Exception as e:
        print(f"Error loading checkpoint from {checkpoint_path}: {e}")
        return {'epoch': 0, 'loss': float('inf'), 'config': {}}

def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about the model architecture and parameters.
    
    Args:
        model: The model to analyze
    
    Returns:
        Dictionary containing model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_type': 'Mamba' if MAMBA_AVAILABLE else 'LSTM Substitute',
        'vocab_size': model.vocab_size,
        'd_model': model.d_model,
        'n_layers': model.n_layers,
    }
    
    return info

def load_checkpoint_with_config(checkpoint_path: str, 
                                device: torch.device,
                                sequential: bool,
                                huggingface: bool = True) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Load a model checkpoint and automatically recover the configuration.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
    
    Returns:
        Tuple of (model, config, checkpoint_info)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Try to extract config from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
            print("Found config in checkpoint:")
            for key, value in config.items():
                print(f"  {key}: {value}")
        else:
            # Try to infer config from model state dict
            raise ValueError("Checkpoint does not contain configuration information.")
            
        # Create model with recovered config
        model = create_model(config, device, sequential, huggingface)
        
        # Load the weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint contains only the state dict
            model.load_state_dict(checkpoint)
        
        # Extract additional checkpoint info
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'loss': checkpoint.get('loss', 'unknown'),
            'training_completed': checkpoint.get('training_completed', False)
        }
        
        print(f"Successfully loaded model from {checkpoint_path}")
        print(f"Checkpoint info: {checkpoint_info}")
        
        return model, config, checkpoint_info
        
    except Exception as e:
        print(f"Error loading checkpoint from {checkpoint_path}: {e}")
        raise e
