import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import Tuple, List, Dict
import os
from torch.amp import autocast, GradScaler
from data_modadd import (
    generate_modular_addition_dataset,
    generate_balanced_modular_dataset,
    generate_exhaustive_modular_dataset,
    load_dataset
)
from model_loader import (
    create_model, 
    load_model_weights, 
    save_model_checkpoint
)

#scaler = GradScaler()
scaler = None
use_lr_scheduler = False
seq_training = True
huggingface_model = True

def split_dataset(dataset, split_ratio=0.3):  # Using 30% for training to induce grokking
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))
    return train_dataset, test_dataset

def create_dataloader(X: np.ndarray, Y: np.ndarray, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    X_tensor = torch.from_numpy(X).long()
    Y_tensor = torch.from_numpy(Y).long()
    dataset = TensorDataset(X_tensor, Y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, 
                lr_scheduler: optim.lr_scheduler._LRScheduler, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        if scaler is not None:
            # Forward pass
            with autocast(device_type=device.type):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            predictions = torch.argmax(logits, dim=1)
            scaler.scale(loss).backward()
            # Backward pass
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            predictions = torch.argmax(logits, dim=1)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        correct += (predictions == batch_y).sum().item()
        #correct += (batch_y == batch_y).sum().item()
        total += batch_y.size(0)
    if lr_scheduler is not None:
        lr_scheduler.step()
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
            device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            predictions = torch.argmax(logits, dim=1)
            
            total_loss += loss.item()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

# Calculating average accuracy for different training set sizes and plotting
def calculate_average_accuracy(model: nn.Module,
                               sequence_lengths: List[int], samples: int = 100,
                               batch_size: int = 32, base: int = 20,
                               device: torch.device = torch.device('cpu')):

    accuracies = []
    for length in sequence_lengths:

        data_path = f'/Data/pgi-14/mallik/modular_addition_seqlen={length}_base={base}_data.pkl'
        _, X_test, _, Y_test = load_dataset(data_path)
    
        # Create a balanced dataset of the given size
        X_subset, Y_subset = X_test[:samples], Y_test[:samples]
        dataloader = create_dataloader(X_subset, Y_subset, batch_size=batch_size, shuffle=False)
        _, avg_acc = evaluate(model, dataloader, nn.CrossEntropyLoss(), device)
        
        accuracies.append(avg_acc)
        print(f"Sequence length: {length}, Average Accuracy: {avg_acc:.4f}")
    return accuracies

def plot_training_curves(train_losses: List[float], train_accuracies: List[float],
                        test_losses: List[float], test_accuracies: List[float],
                        save_path: str = None):
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    #ax1.set_yscale('log')
    
    # Accuracy curves
    ax2.plot(epochs, train_accuracies, 'b-', label='Train Accuracy', alpha=0.7)
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    ax2.set_yticks(np.arange(0, 1.2, 0.2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def main():

    model_path = 'modular_addition_mamba.pth'
    
    # Configuration
    config = {
        'base': 20,  # Modulo base
        'sequence_length': 4,
        'n_samples_train': 60000,
        'n_samples_test': 10000,
        #'samples_per_class_train': 200,
        #'samples_per_class_test': 50,
        'd_model': 128,
        'n_layers': 4,
        'batch_size': 256,
        'learning_rate': 3e-4,
        'weight_decay': 1e-1, # Default is 1e-2 in AdamW
        'epochs': 500,
        'seed': 28, 
        'save_checkpoint_every': 50,
        'model_save_path': model_path
    }
    data_path = f'/Data/pgi-14/mallik/modular_addition_seqlen={config["sequence_length"]}_base={config["base"]}_data.pkl'

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    '''
    # Generate datasets
    print("\nGenerating datasets...")
    X_train, Y_train = generate_modular_addition_dataset(
        config['n_samples_train'],
        #config['samples_per_class_train'], 
        config['sequence_length'], 
        config['base'], 
        seed=config['seed']
    )

    X_test, Y_test = generate_modular_addition_dataset(
        config['n_samples_test'],
        #config['samples_per_class_test'],
        config['sequence_length'],
        config['base'],
        seed=config['seed'] + 1
    )
    '''
    # Loading datasets
    X_train, X_test, Y_train, Y_test = load_dataset(data_path)
    X_train = X_train[5000:5000+config['n_samples_train']]
    Y_train = Y_train[5000:5000+config['n_samples_train']]
    X_test = X_test[5000:5000+config['n_samples_test']]
    Y_test = Y_test[5000:5000+config['n_samples_test']]

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create dataloaders
    train_loader = create_dataloader(X_train, Y_train, config['batch_size'], shuffle=True)
    test_loader = create_dataloader(X_test, Y_test, config['batch_size'], shuffle=False)
    
    # Initialize model
    model = create_model(config, 
                         device=device, 
                         sequential=seq_training,
                         huggingface_model=huggingface_model)

    # Initialize optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), 
                            lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    if use_lr_scheduler:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-5)
    else:
        lr_scheduler = None

    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in tqdm(range(config['epochs']), desc="Training"):
        # Train
        train_loss, train_acc = train_epoch(model, 
                                            train_loader, 
                                            optimizer, 
                                            lr_scheduler,
                                            criterion, 
                                            device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Save periodic checkpoints
        if (epoch + 1) % config['save_checkpoint_every'] == 0:
            save_model_checkpoint(
                model, optimizer, epoch, test_loss, 
                f"checkpoint_epoch_{epoch+1}.pth", config
            )
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}: Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}, "
                  f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    save_model_checkpoint(
        model, optimizer, config['epochs']-1, test_losses[-1], 
        'final_' + config['model_save_path'], config
    )
    
    print(f"\nTraining completed!")
    
    # Plot results
    print("\nGenerating training curves...")
    plot_training_curves(train_losses, train_accuracies, test_losses, test_accuracies,
                         save_path='mamba_modular_training_curves.png')
    
    # Save results
    results = {
        'config': config,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
    }
    
    with open('mamba_modular_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to 'mamba_modular_results.json'")
    print("Training curves saved to 'mamba_modular_training_curves.png'")
    print(f"Model checkpoints saved with prefix: {config['model_save_path']}")

if __name__ == "__main__":
    main()