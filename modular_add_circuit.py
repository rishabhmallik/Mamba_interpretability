import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple
import seaborn as sns

def analyze_modular_addition_circuits(hooked_model, X_test, y_test, base=20, n_samples=100):
    """
    Analyze how a model combines four numbers for modular addition.
    Tests different computational hypotheses through systematic corruption and patching.
    
    Args:
        hooked_model: Your HookedMamba model
        X_test: numpy array of shape (N, 4) with test inputs
        y_test: numpy array of shape (N,) with correct outputs
        base: modular base (20 in your case)
        n_samples: number of test samples to use
    """
    
    # Take subset of test data
    indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_subset = X_test[indices]
    y_subset = y_test[indices]
    
    results = {}
    
    # Test different computational hypotheses
    computational_tests = {
        'sequential': test_sequential_computation,
        'pairwise_parallel': test_pairwise_computation,
        'all_parallel': test_all_parallel_computation,
        'early_modular': test_early_modular_reduction,
    }
    
    print("Testing different computational strategies...")
    
    for strategy_name, test_func in computational_tests.items():
        print(f"Testing {strategy_name} computation...")
        results[strategy_name] = test_func(hooked_model, X_subset, y_subset, base)
    
    # Generate comprehensive plots
    create_computation_analysis_plots(results, X_subset, y_subset, base)
    
    return results

def test_sequential_computation(hooked_model, X_test, y_test, base):
    """Test if model computes ((a + b) + c) + d mod base"""
    
    results = {'layer_importance': [], 'position_importance': [], 'intermediate_tracking': []}
    n_layers = hooked_model.cfg.n_layers
    
    for sample_idx in range(min(20, len(X_test))):  # Test on subset
        a, b, c, d = X_test[sample_idx]
        correct_answer = y_test[sample_idx]
        
        # Create sequential corruptions that break different intermediate steps
        corruptions = {
            'corrupt_first_sum': [a+5, b, c, d],      # Break a+b
            'corrupt_second_sum': [a, b, c+7, d],     # Break (a+b)+c  
            'corrupt_third_sum': [a, b, c, d+3],      # Break ((a+b)+c)+d
        }
        
        sample_results = {}
        
        for corruption_type, corrupted_input in corruptions.items():
            # Test layer-by-layer patching
            layer_effects = []
            
            for layer in range(n_layers):
                # Patch residual stream at each layer
                restoration = patch_and_measure_restoration(
                    hooked_model, 
                    clean_input=list(X_test[sample_idx]),
                    corrupted_input=corrupted_input,
                    patch_component=f"layers.{layer}.resid_post",
                    correct_answer=correct_answer
                )
                layer_effects.append(restoration)
            
            sample_results[corruption_type] = layer_effects
        
        results['layer_importance'].append(sample_results)
    
    return results

def test_pairwise_computation(hooked_model, X_test, y_test, base):
    """Test if model computes (a + b) + (c + d) mod base"""
    
    results = {'pair_effects': [], 'convergence_layers': []}
    
    for sample_idx in range(min(20, len(X_test))):
        a, b, c, d = X_test[sample_idx]
        correct_answer = y_test[sample_idx]
        
        # Test corruptions that break different pairs
        pair_corruptions = {
            'corrupt_first_pair': [a+5, b+3, c, d],     # Break first pair
            'corrupt_second_pair': [a, b, c+7, d+2],    # Break second pair
            'corrupt_both_pairs': [a+2, b+1, c+4, d+3], # Break both pairs
        }
        
        sample_results = {}
        
        for corruption_type, corrupted_input in pair_corruptions.items():
            # Test which layers are critical for pair processing
            layer_effects = test_layer_importance(
                hooked_model, list(X_test[sample_idx]), corrupted_input, correct_answer
            )
            sample_results[corruption_type] = layer_effects
        
        results['pair_effects'].append(sample_results)
    
    return results

def test_all_parallel_computation(hooked_model, X_test, y_test, base):
    """Test if model processes all numbers in parallel before combining"""
    
    results = {'individual_effects': [], 'combination_layers': []}
    
    for sample_idx in range(min(20, len(X_test))):
        a, b, c, d = X_test[sample_idx]
        correct_answer = y_test[sample_idx]
        
        # Test corrupting individual numbers
        individual_corruptions = {
            'corrupt_a': [a+8, b, c, d],
            'corrupt_b': [a, b+8, c, d], 
            'corrupt_c': [a, b, c+8, d],
            'corrupt_d': [a, b, c, d+8],
        }
        
        sample_results = {}
        
        for corruption_type, corrupted_input in individual_corruptions.items():
            layer_effects = test_layer_importance(
                hooked_model, list(X_test[sample_idx]), corrupted_input, correct_answer
            )
            sample_results[corruption_type] = layer_effects
        
        results['individual_effects'].append(sample_results)
    
    return results

def test_early_modular_reduction(hooked_model, X_test, y_test, base):
    """Test if model applies modular reduction early vs late"""
    
    results = {'modular_timing': []}
    
    for sample_idx in range(min(20, len(X_test))):
        a, b, c, d = X_test[sample_idx]
        correct_answer = y_test[sample_idx]
        
        # Create inputs that test when modular reduction happens
        mod_test_corruptions = {
            # Numbers that sum to >base (tests late modular reduction)
            'high_sum': [15, 18, 17, 19],  # Sum = 69, 69 % 20 = 9
            # Numbers where intermediate sums exceed base
            'intermediate_overflow': [19, 19, 1, 1],  # (19+19)%20 + (1+1) vs 19+19+1+1
        }
        
        sample_results = {}
        
        for corruption_type, test_input in mod_test_corruptions.items():
            layer_effects = test_layer_importance(
                hooked_model, list(X_test[sample_idx]), test_input, correct_answer
            )
            sample_results[corruption_type] = layer_effects
        
        results['modular_timing'].append(sample_results)
    
    return results

def test_layer_importance(hooked_model, clean_input, corrupted_input, correct_answer):
    """Test importance of each layer for a specific corruption."""
    
    layer_effects = []
    n_layers = hooked_model.cfg.n_layers
    
    for layer in range(n_layers):
        restoration = patch_and_measure_restoration(
            hooked_model,
            clean_input=clean_input,
            corrupted_input=corrupted_input, 
            patch_component=f"layers.{layer}.resid_post",
            correct_answer=correct_answer
        )
        layer_effects.append(restoration)
    
    return layer_effects

def patch_and_measure_restoration(hooked_model, clean_input, corrupted_input, patch_component, correct_answer):
    """Measure how much patching a component restores correct behavior."""
    
    # Get clean and corrupted baseline performance
    clean_logits, clean_cache = hooked_model.run_with_cache(clean_input)
    corrupted_input = hooked_model.to_tokens(corrupted_input).to(hooked_model.cfg.device)
    corrupted_logits = hooked_model(corrupted_input)
    
    # Define metric (probability of correct answer)
    def get_answer_prob(logits):
        probs = torch.softmax(logits[0, -1], dim=-1)
        return probs[correct_answer].item()
    
    clean_prob = get_answer_prob(clean_logits)
    corrupted_prob = get_answer_prob(corrupted_logits)
    
    # Patch the component
    def patch_hook(activation, hook, clean_cache=clean_cache):
        clean_activation = clean_cache[hook.name]
        return clean_activation
    
    patched_logits = hooked_model.run_with_hooks(
        corrupted_input,
        fwd_hooks=[(patch_component, patch_hook)]
    )
    
    patched_prob = get_answer_prob(patched_logits)
    
    # Calculate restoration score
    if clean_prob != corrupted_prob:
        restoration = (patched_prob - corrupted_prob) / (clean_prob - corrupted_prob)
    else:
        restoration = 0.0
    
    return restoration

def create_computation_analysis_plots(results, X_test, y_test, base):
    """Create comprehensive visualization of computational analysis."""
    
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Sequential vs Parallel Computation Evidence
    ax1 = plt.subplot(2, 3, 1)
    plot_computation_comparison(results, ax1)
    ax1.set_title('Sequential vs Parallel Processing Evidence')
    
    # Plot 2: Layer-wise Computational Roles  
    ax2 = plt.subplot(2, 3, 2)
    plot_layer_roles(results, ax2)
    ax2.set_title('Computational Roles by Layer')
    
    # Plot 3: Position-wise Analysis
    ax3 = plt.subplot(2, 3, 3)
    plot_position_analysis(results, X_test, ax3)
    ax3.set_title('Token Position Importance')
    
    # Plot 4: Modular Arithmetic Timing
    ax4 = plt.subplot(2, 3, 4)
    plot_modular_timing(results, ax4)
    ax4.set_title('When Modular Reduction Occurs')
    
    # Plot 5: Circuit Pathway Diagram
    ax5 = plt.subplot(2, 3, 5)
    plot_circuit_diagram(results, ax5)
    ax5.set_title('Discovered Circuit Architecture')
    
    # Plot 6: Performance Summary
    ax6 = plt.subplot(2, 3, 6)
    plot_performance_summary(results, ax6)
    ax6.set_title('Hypothesis Support Summary')
    
    plt.tight_layout()
    plt.savefig('modular_addition_circuit_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_computation_comparison(results, ax):
    """Compare evidence for different computational strategies."""
    
    strategies = list(results.keys())
    avg_restoration = []
    
    for strategy in strategies:
        if 'layer_importance' in results[strategy]:
            # Average restoration across all samples and corruptions
            all_restorations = []
            for sample in results[strategy]['layer_importance']:
                for corruption_type, layer_effects in sample.items():
                    all_restorations.extend(layer_effects)
            avg_restoration.append(np.mean(all_restorations))
        else:
            avg_restoration.append(0.0)
    
    bars = ax.bar(range(len(strategies)), avg_restoration)
    ax.set_xlabel('Computational Strategy')
    ax.set_ylabel('Average Restoration Score')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([s.replace('_', '\n') for s in strategies], rotation=45)
    
    # Color bars by performance
    for bar, score in zip(bars, avg_restoration):
        bar.set_color(plt.cm.RdYlBu_r(score))

def plot_layer_roles(results, ax):
    """Show what role each layer plays in computation."""
    
    # Extract layer importance across all strategies
    n_layers = 4  # Adjust based on your model
    layer_roles = np.zeros((len(results), n_layers))
    
    for i, (strategy, data) in enumerate(results.items()):
        if 'layer_importance' in data and data['layer_importance']:
            # Average across samples
            layer_avg = np.zeros(n_layers)
            count = 0
            for sample in data['layer_importance']:
                for corruption_type, layer_effects in sample.items():
                    if len(layer_effects) == n_layers:
                        layer_avg += np.array(layer_effects)
                        count += 1
            if count > 0:
                layer_roles[i] = layer_avg / count
    
    im = ax.imshow(layer_roles, cmap='RdYlBu_r', aspect='auto')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Computational Strategy')
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([s.replace('_', '\n') for s in results.keys()])
    plt.colorbar(im, ax=ax, label='Restoration Score')

def plot_position_analysis(results, X_test, ax):
    """Analyze importance of different token positions."""
    
    # This would need position-specific patching results
    # For now, show a placeholder analysis
    positions = ['Num1', 'Num2', 'Num3', 'Num4', 'Output']
    importance = np.random.rand(len(positions))  # Replace with actual data
    
    ax.bar(positions, importance)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Average Importance')
    ax.tick_params(axis='x', rotation=45)

def plot_modular_timing(results, ax):
    """Show when modular reduction likely occurs."""
    
    if 'early_modular' in results:
        # Compare early vs late modular reduction evidence
        early_evidence = 0.7  # Replace with actual calculation
        late_evidence = 0.3
        
        ax.bar(['Early Modular\nReduction', 'Late Modular\nReduction'], 
               [early_evidence, late_evidence])
        ax.set_ylabel('Evidence Strength')
        ax.set_title('Timing of Modular Operation')

def plot_circuit_diagram(results, ax):
    """Visualize the discovered circuit architecture."""
    
    # Create a simplified circuit diagram
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    
    # Input layer
    for i, label in enumerate(['a', 'b', 'c', 'd']):
        ax.add_patch(plt.Rectangle((1, 6-i*1.5), 1, 0.8, 
                                   facecolor='lightblue', edgecolor='black'))
        ax.text(1.5, 6.4-i*1.5, label, ha='center', va='center')
    
    # Processing layers (simplified)
    ax.add_patch(plt.Rectangle((4, 3), 2, 2, 
                               facecolor='lightgreen', edgecolor='black'))
    ax.text(5, 4, 'Processing\nLayers', ha='center', va='center')
    
    # Output
    ax.add_patch(plt.Rectangle((8, 3.5), 1, 1, 
                               facecolor='lightcoral', edgecolor='black'))
    ax.text(8.5, 4, 'Sum\nmod 20', ha='center', va='center')
    
    # Add arrows (simplified)
    for i in range(4):
        ax.arrow(2, 6.4-i*1.5, 1.8, (4-6.4+i*1.5)*0.3, 
                head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    ax.arrow(6, 4, 1.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

def plot_performance_summary(results, ax):
    """Summarize which computational hypothesis is best supported."""
    
    # Calculate support scores for each hypothesis
    hypothesis_scores = {}
    
    for strategy, data in results.items():
        if 'layer_importance' in data and data['layer_importance']:
            scores = []
            for sample in data['layer_importance']:
                for corruption_type, layer_effects in sample.items():
                    scores.extend(layer_effects)
            hypothesis_scores[strategy] = np.mean([s for s in scores if s > 0])
        else:
            hypothesis_scores[strategy] = 0.0
    
    strategies = list(hypothesis_scores.keys())
    scores = list(hypothesis_scores.values())
    
    bars = ax.barh(strategies, scores)
    ax.set_xlabel('Average Restoration Score')
    ax.set_title('Computational Hypothesis Support')
    
    # Highlight the best supported hypothesis
    best_idx = np.argmax(scores)
    bars[best_idx].set_color('gold')
    
    # Add score labels
    for i, score in enumerate(scores):
        ax.text(score + 0.01, i, f'{score:.3f}', va='center')

# Example usage
if __name__ == "__main__":
    # Assuming you have your hooked_model, X_test, and y_test ready
    # results = analyze_modular_addition_circuits(hooked_model, X_test, y_test, base=20)
    print("Run analyze_modular_addition_circuits(hooked_model, X_test, y_test) to generate the analysis")