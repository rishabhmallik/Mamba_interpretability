import numpy as np
import torch
from typing import Tuple, List, Optional
import random
import pickle, itertools
from torch.utils.data import TensorDataset, random_split

def generate_modular_addition_dataset(
    n_samples: int,
    sequence_length: int,
    base: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of (x, y) pairs for modular addition.
    
    Args:
        n_samples: Number of samples to generate
        sequence_length: Length of each sequence (n)
        base: Base for the numbers (d). Numbers will be in range [0, d-1]
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (X, Y) where:
        - X: Array of shape (n_samples, sequence_length) containing sequences
        - Y: Array of shape (n_samples,) containing sums modulo base
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Generate sequences of numbers from 0 to base-1
    X = np.random.randint(0, base, size=(n_samples, sequence_length))
    
    # Calculate sum of each sequence modulo base
    Y = np.sum(X, axis=1) % base
    
    return X, Y

def generate_exhaustive_modular_dataset(
    sequence_length: int,
    base: int,
    train_ratio: float = 0.5,
    seed: Optional[int] = None,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate all possible combinations for modular addition and split into train/test.
    
    Args:
        sequence_length: Length of each sequence (n)
        base: Base for the numbers (d). Numbers will be in range [0, d-1]
        test_ratio: Fraction of data to use for testing
        seed: Random seed for train/test split
        save_path: Optional path to save the dataset
    
    Returns:
        Tuple of (X_train, X_test, Y_train, Y_test)
    """
    print(f"Generating all combinations for sequence_length={sequence_length}, base={base}")
    
    # Calculate total number of combinations
    total_combinations = base ** sequence_length
    print(f"Total combinations: {total_combinations:,}")
    
    # Generate all possible sequences using itertools.product
    print("Generating all sequences...")
    all_sequences = list(itertools.product(range(base), repeat=sequence_length))
    
    # Convert to numpy array
    X = np.array(all_sequences, dtype=np.int32)
    
    # Calculate modular sums
    print("Calculating modular sums...")
    Y = np.sum(X, axis=1) % base
    
    print(f"Generated {len(X):,} total samples")
    print(f"Sequence shape: {X.shape}")
    print(f"Target distribution: {np.bincount(Y)}")
    
    # Split into train and test using torch.utils.data.random_split
    print(f"Splitting into train ({train_ratio:.1%}) and test ({1-train_ratio:.1%})...")
    
    # Create a TensorDataset
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    # Set random seed for reproducible split
    if seed is not None:
        torch.manual_seed(seed)
    
    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Extract numpy arrays from the split datasets
    train_indices = train_dataset.indices
    test_indices = test_dataset.indices
    
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    X_test = X[test_indices] 
    Y_test = Y[test_indices]
    
    print(f"Train set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print(f"Train target distribution: {np.bincount(Y_train)}")
    print(f"Test target distribution: {np.bincount(Y_test)}")
    
    # Save dataset if path provided
    if save_path:
        save_dataset(X_train, X_test, Y_train, Y_test, save_path)
    
    return X_train, X_test, Y_train, Y_test

def save_dataset(X_train, X_test, Y_train, Y_test, save_path):
    """Save the dataset to disk with metadata"""
    dataset = {
        'X_train': X_train,
        'X_test': X_test,
        'Y_train': Y_train,
        'Y_test': Y_test,
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"✓ Dataset saved to {save_path}")

def load_dataset(load_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a saved dataset from disk.
    
    Args:
        load_path: Path to the saved dataset file
    
    Returns:
        Tuple of (X_train, X_test, Y_train, Y_test, metadata)
    """
    with open(load_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"✓ Dataset loaded from {load_path}")
    
    return (dataset['X_train'], dataset['X_test'], 
            dataset['Y_train'], dataset['Y_test'])

def generate_balanced_modular_dataset(
    samples_per_class: int,
    sequence_length: int,
    base: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced dataset where each possible sum (mod base) appears equally often.
    
    Args:
        samples_per_class: Number of samples for each possible sum (0 to base-1)
        sequence_length: Length of each sequence
        base: Base for the numbers
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (X, Y) with balanced class distribution
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    total_samples = samples_per_class * base
    X = []
    Y = []
    
    for target_sum in range(base):
        for _ in range(samples_per_class):
            # Generate a sequence that sums to target_sum (mod base)
            sequence = generate_sequence_with_target_sum(sequence_length, base, target_sum)
            X.append(sequence)
            Y.append(target_sum)
    
    # Convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    
    # Shuffle the dataset
    indices = np.random.permutation(len(X))
    return X[indices], Y[indices]

def generate_sequence_with_target_sum(
    sequence_length: int, 
    base: int, 
    target_sum: int
) -> np.ndarray:
    """
    Generate a sequence of given length that sums to target_sum modulo base.
    """
    if sequence_length == 1:
        return np.array([target_sum % base])
    
    # Generate random first n-1 elements
    sequence = np.random.randint(0, base, size=sequence_length - 1)
    
    # Calculate what the last element should be to achieve target sum
    current_sum = np.sum(sequence) % base
    last_element = (target_sum - current_sum) % base
    
    # Add the calculated last element
    sequence = np.append(sequence, last_element)
    
    return sequence

def create_torch_dataset(X: np.ndarray, Y: np.ndarray) -> torch.utils.data.TensorDataset:
    """
    Convert numpy arrays to PyTorch TensorDataset.
    """
    X_tensor = torch.from_numpy(X).long()
    Y_tensor = torch.from_numpy(Y).long()
    return torch.utils.data.TensorDataset(X_tensor, Y_tensor)

def print_dataset_stats(X: np.ndarray, Y: np.ndarray, base: int):
    """
    Print statistics about the generated dataset.
    """
    print(f"Dataset Statistics:")
    print(f"  Number of samples: {len(X)}")
    print(f"  Sequence length: {X.shape[1]}")
    print(f"  Base: {base}")
    print(f"  Input range: [0, {base-1}]")
    print(f"  Output range: [0, {base-1}]")
    
    # Class distribution
    unique, counts = np.unique(Y, return_counts=True)
    print(f"\nClass distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Sum ≡ {cls} (mod {base}): {count} samples")
    
    # Sample examples
    print(f"\nSample examples:")
    for i in range(min(5, len(X))):
        sequence_str = " + ".join(map(str, X[i]))
        actual_sum = np.sum(X[i])
        print(f"  [{sequence_str}] = {actual_sum} ≡ {Y[i]} (mod {base})")

# Example usage and testing
if __name__ == "__main__":
    # Generate a small dataset for testing
    
    base = 20
    sequence_length = 7
    
    X_train, X_test, Y_train, Y_test = generate_exhaustive_modular_dataset(
        sequence_length=sequence_length,
        base=base,
        train_ratio=0.5,
        seed=42,
        save_path=f'/Data/pgi-14/mallik/modular_addition_seqlen={sequence_length}_base={base}_data.pkl'
    )

    print_dataset_stats(X_train, Y_train, base=base)
    print_dataset_stats(X_test, Y_test, base=base)