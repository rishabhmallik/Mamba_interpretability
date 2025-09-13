import torch
import torch.nn as nn
from transformers import MambaForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Callable, Optional
import functools
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class HookedMamba:
    """
    A wrapper for HuggingFace Mamba models that adds hooks for activation patching.
    Similar to TransformerLens HookedTransformer but for Mamba models.
    Enhanced with position-specific hooks for hidden states.
    """
    
    def __init__(self, model: MambaForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        
        # Storage for hooks and activations
        self.hooks = {}
        self.activation_cache = {}
        self.forward_hooks = []
        self.position_hooks = {}  # For position-specific hooks
        
        # Model configuration
        self.cfg = self._create_config()
        
        # Add permanent hooks for caching
        self._add_permanent_hooks()
    
    def _create_config(self):
        """Create a config object similar to TransformerLens."""
        config = self.model.config
        device = self.model.device
        
        class MambaConfig:
            def __init__(self, hf_config):
                self.n_layers = hf_config.num_hidden_layers
                self.d_model = hf_config.hidden_size
                self.vocab_size = hf_config.vocab_size
                self.device = device
                # Add other relevant config parameters
                self.state_size = hf_config.state_size
                self.conv_kernel = hf_config.conv_kernel
                self.expand = hf_config.expand
        
        return MambaConfig(config)
    
    def _add_permanent_hooks(self):
        """Add hooks to all relevant layers for caching activations."""
        
        # Hook the embedding layer
        self.model.backbone.embeddings.register_forward_hook(
            self._make_cache_hook("embeddings")
        )
        
        # Hook each Mamba layer
        for i, layer in enumerate(self.model.backbone.layers):
            # resid_pre: Input to layer (before any processing)
            layer.register_forward_pre_hook(
                self._make_pre_cache_hook(f"layers.{i}.resid_pre")
            )
            
            # resid_post: Output of layer (after residual addition)
            layer.register_forward_hook(
                self._make_cache_hook(f"layers.{i}.resid_post")
            )
            
            # After layer norm (input to mixer)
            layer.norm.register_forward_hook(
                self._make_cache_hook(f"layers.{i}.norm_output")
            )
            
            # After mixer (before residual add)
            layer.mixer.register_forward_hook(
                self._make_cache_hook(f"layers.{i}.mixer_output")
            )
            
            # Add position-specific hooks for hidden states
            self._add_position_hooks(layer, i)
    
        # Hook the final norm
        self.model.backbone.norm_f.register_forward_hook(
            self._make_cache_hook("norm_f")
        )
        
        # Hook the language model head
        self.model.lm_head.register_forward_hook(
            self._make_cache_hook("lm_head")
        )
    
    def _add_position_hooks(self, layer, layer_idx):
        """Add position-specific hooks for hidden states h."""
        # Hook the mixer to capture hidden states at each position
        def position_aware_hook(module, input, output):
            hidden_states = output  # The output hidden states
            if hasattr(hidden_states, 'clone'):
                # Cache the full hidden states
                self.activation_cache[f"layers.{layer_idx}.hidden_states"] = hidden_states.clone().detach()
                
                # Cache position-specific hidden states
                seq_len = hidden_states.shape[1]
                for pos in range(seq_len):
                    hook_name = f"blocks.{layer_idx}.hook_h.{pos}"
                    self.activation_cache[hook_name] = hidden_states[:, pos, :].clone().detach()
        
        layer.mixer.register_forward_hook(position_aware_hook)
    
    def _make_pre_cache_hook(self, name: str):
        """Create a pre-hook function that caches input activations."""
        def pre_hook(module, input):
            # input is a tuple, input[0] is the hidden_states
            if hasattr(input[0], 'clone'):
                self.activation_cache[name] = input[0].clone().detach()
            # Pre-hooks don't return anything
        return pre_hook
    
    def _make_cache_hook(self, name: str):
        """Create a hook function that caches activations."""
        def hook(module, input, output):
            if hasattr(output, 'clone'):
                self.activation_cache[name] = output.clone().detach()
            else:
                # Handle tuple outputs
                self.activation_cache[name] = output
        return hook
    
    def to_tokens(self, input_data) -> torch.Tensor:
        """Convert text to tokens, similar to TransformerLens."""
        
        # If already a tensor, assume it's tokens
        if isinstance(input_data, torch.Tensor):
            if input_data.dim() == 1:
                input_data = input_data.unsqueeze(0)  # Add batch dim if missing
            return input_data.to(self.device)
        
        elif isinstance(input_data, list):
            # Check if all elements are numeric (handles numpy types too)
            if all(isinstance(x, (int, np.integer)) or hasattr(x, 'item') for x in input_data):
                int_list = []
                for x in input_data:
                    if isinstance(x, (int, np.integer)):
                        int_list.append(int(x))  # Convert to Python int
                    elif hasattr(x, 'item'):
                        int_list.append(int(x.item()))
                    else:
                        int_list.append(int(x))
                
                return torch.tensor(int_list, dtype=torch.long).unsqueeze(0).to(self.device)
            
        elif isinstance(input_data, str):
            tokens = self.tokenizer(input_data, return_tensors="pt", add_special_tokens=False)
            return tokens.input_ids.to(self.device)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}. Expected str, list of ints, or torch.Tensor.")

    def to_single_token(self, input_data) -> int:
        """Convert text or token ID to single token ID, raises error if not exactly one token."""
        
        # If it's already an integer, assume it's a token ID
        if isinstance(input_data, (int, np.integer)):
            return input_data
        
        # If it's a tensor with single element
        elif isinstance(input_data, torch.Tensor):
            if input_data.numel() == 1:
                return input_data.item()
            else:
                raise ValueError(f"Tensor contains {input_data.numel()} elements, expected exactly 1")
        
        # If it's a string, tokenize and check
        elif isinstance(input_data, str):
            tokens = self.tokenizer(input_data, add_special_tokens=False)['input_ids']
            if len(tokens) != 1:
                raise ValueError(f"Text '{input_data}' is not a single token. Got {len(tokens)} tokens: {tokens}")
            return tokens[0]
        
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}. Expected str, int, or torch.Tensor")
        
    def to_str_tokens(self, input) -> List[str]:
        """Convert text to list of token strings."""
        tokens = self.to_tokens(input)[0]
        return [self.tokenizer.decode(token) for token in tokens]
    
    def forward(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the model."""
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        
        outputs = self.model(tokens, **kwargs)
        return outputs.logits
    
    def __call__(self, tokens, **kwargs):
        """Make the object callable like the original model."""
        return self.forward(tokens, **kwargs)
    
    def run_with_cache(self, text_or_tokens, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Run model and return outputs with cached activations.
        Similar to TransformerLens run_with_cache.
        """
        # Clear previous cache
        self.activation_cache.clear()
        
        # Convert to tokens if needed
        tokens = self.to_tokens(text_or_tokens)
        
        # Run forward pass (hooks automatically cache activations)
        logits = self.forward(tokens, **kwargs)
        
        # Return logits and cache
        return logits, self.activation_cache.copy()
    
    def run_with_hooks(self, 
                      text_or_tokens, 
                      fwd_hooks: List[Tuple[str, Callable]] = None,
                      **kwargs) -> torch.Tensor:
        """
        Run model with temporary hooks for activation patching.
        Similar to TransformerLens run_with_hooks.
        """
        # Convert to tokens
        tokens = self.to_tokens(text_or_tokens)
        
        # Register temporary hooks
        temp_hook_handles = []
        if fwd_hooks:
            for hook_name, hook_fn in fwd_hooks:
                module = self._get_module_by_name(hook_name)
                if module is not None:
                    # Check if this is a resid_pre hook (needs pre-hook)
                    if "resid_pre" in hook_name:
                        handle = module.register_forward_pre_hook(
                            lambda module, input, fn=hook_fn: (fn(input[0], MockHookPoint(hook_name)),)
                        )
                    else:
                        # Handle position-specific hooks
                        if "hook_h" in hook_name:
                            # Extract position from hook name
                            position = int(hook_name.split('.')[-1])
                            handle = module.register_forward_hook(
                                lambda module, input, output, fn=hook_fn, pos=position: 
                                self._position_specific_hook(output, fn, pos, hook_name)
                            )
                        else:
                            # Regular forward hook
                            handle = module.register_forward_hook(
                                lambda module, input, output, fn=hook_fn: fn(output, MockHookPoint(hook_name))
                            )
                    temp_hook_handles.append(handle)
        
        try:
            # Run forward pass
            logits = self.forward(tokens, **kwargs)
        finally:
            # Remove temporary hooks
            for handle in temp_hook_handles:
                handle.remove()
        
        return logits
    
    def _position_specific_hook(self, output, hook_fn, position, hook_name):
        """Handle position-specific hooks for hidden states."""
        if hasattr(output, 'clone') and output.dim() >= 2:
            # Extract the specific position
            if position < output.shape[1]:
                pos_output = output[:, position, :]
                # Apply the hook function
                modified_pos = hook_fn(pos_output, MockHookPoint(hook_name))
                # Put it back
                output[:, position, :] = modified_pos
        return output
    
    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """Get module by hook name."""
        if name == "embeddings":
            return self.model.backbone.embeddings
        elif name == "norm_f":
            return self.model.backbone.norm_f
        elif name == "lm_head":
            return self.model.lm_head
        elif name.startswith("layers."):
            # Parse layer number and component
            parts = name.split(".")
            if len(parts) >= 3:
                layer_idx = int(parts[1])
                component = ".".join(parts[2:])
                
                layer = self.model.backbone.layers[layer_idx]
                
                if component == "resid_pre":
                    return layer
                elif component == "resid_post":
                    return layer
                elif component == "norm_output":
                    return layer.norm
                elif component == "mixer_output":
                    return layer.mixer
        elif name.startswith("blocks."):
            # Handle MambaLens style hooks: blocks.{layer}.hook_h.{position}
            parts = name.split(".")
            if len(parts) >= 4 and parts[2] == "hook_h":
                layer_idx = int(parts[1])
                return self.model.backbone.layers[layer_idx].mixer
        
        return None
    
    def logit_lens(self, input_data) -> Dict[str, torch.Tensor]:
        """
        Apply logit lens to see layer-by-layer predictions.
        
        Args:
            input_data: Text or tokens to analyze
            
        Returns:
            Dictionary with layer predictions and probabilities
        """
        # Run with cache to get all activations
        logits, cache = self.run_with_cache(input_data)
        
        # Get components for logit lens
        final_norm = self.model.backbone.norm_f
        lm_head = self.model.lm_head
        num_positions = logits.shape[-2]
        num_layers = self.cfg.n_layers
        
        predictions = torch.zeros((logits.shape[0], num_layers + 1, num_positions))
        probabilities = torch.zeros((logits.shape[0], num_layers + 1, num_positions, self.cfg.vocab_size))

        # Apply logit lens to each layer's residual stream
        for layer in range(self.cfg.n_layers):
            # Get residual stream at this layer
            resid_key = f"layers.{layer}.resid_pre"
            if resid_key in cache:
                resid = cache[resid_key]  # [batch, seq, d_model]
                
                # Apply final norm and language model head
                normed_resid = final_norm(resid)
                layer_logits = lm_head(normed_resid)  # [batch, seq, vocab]
                layer_probabilities = torch.softmax(layer_logits, dim=-1)
                
                predictions[:, layer, :] = torch.argmax(layer_logits, dim=-1)
                probabilities[:, layer, :, :] = layer_probabilities
            
        # Add final prediction for comparison
        predictions[:, -1, :] = torch.argmax(logits, dim=-1)
        probabilities[:, -1, :, :] = torch.softmax(logits, dim=-1)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }


class MockHookPoint:
    """Mock HookPoint class to mimic TransformerLens interface."""
    def __init__(self, name: str):
        self.name = name


def logits_to_logit_diff(hooked_mamba: HookedMamba, 
                        logits: torch.Tensor, 
                        correct_answer, 
                        incorrect_answer) -> torch.Tensor:
    """
    Calculate logit difference between correct and incorrect answers.
    Similar to the TransformerLens example.
    """
    correct_index = hooked_mamba.to_single_token(correct_answer)
    incorrect_index = hooked_mamba.to_single_token(incorrect_answer)
    
    return logits[0, -1, correct_index] - logits[0, -1, incorrect_index]


def position_specific_patching_hook(
    activation: torch.Tensor,
    hook: MockHookPoint,
    clean_cache: Dict,
    patch_position: int = None
) -> torch.Tensor:
    """
    Patch activation at specific position with clean activation.
    Enhanced for position-specific patching.
    """
    clean_activation = clean_cache[hook.name]
    
    if "hook_h" in hook.name:
        # Position-specific hidden state patching
        activation[:] = clean_activation[:]
    else:
        # Regular activation patching
        if activation.dim() == 3:  # [batch, seq, hidden]
            if patch_position is not None:
                activation[:, patch_position, :] = clean_activation[:, patch_position, :]
            else:
                activation[:] = clean_activation[:]
        elif activation.dim() == 2:  # [batch, hidden]
            activation[:] = clean_activation[:]
    
    return activation


def comprehensive_activation_patching_experiment(
    model=None, 
    tokenizer=None,
    clean_input="After John and Mary went to the store, Mary gave a bottle of milk to",
    corrupted_input="After John and Mary went to the store, John gave a bottle of milk to",
    correct_answer=" John",
    incorrect_answer=" Mary"
):
    """
    Complete activation patching experiment for Mamba models with position-specific hooks.
    """
    if model is None:
        model_name = "state-spaces/mamba-130m-hf"
        model = MambaForCausalLM.from_pretrained(model_name)
    if tokenizer is None:
        model_name = "state-spaces/mamba-130m-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create hooked version
    hooked_mamba = HookedMamba(model, tokenizer)
    
    # Convert to tokens
    clean_tokens = hooked_mamba.to_tokens(clean_input)
    corrupted_tokens = hooked_mamba.to_tokens(corrupted_input)

    if isinstance(clean_input, (list, torch.Tensor)):
        print(f"Clean input: {clean_tokens}")
        print(f"Corrupted input: {corrupted_tokens}")
    else:
        print(f"Clean tokens: {hooked_mamba.to_str_tokens(clean_input)}")
        print(f"Corrupted tokens: {hooked_mamba.to_str_tokens(corrupted_input)}")

    # Run clean prompt with cache
    clean_logits, clean_cache = hooked_mamba.run_with_cache(clean_tokens)
    clean_logit_diff = logits_to_logit_diff(hooked_mamba, clean_logits, correct_answer, incorrect_answer)
    print(f"Clean logit difference: {clean_logit_diff.item():.3f}")
    
    # Run corrupted prompt
    corrupted_logits = hooked_mamba(corrupted_tokens)
    corrupted_logit_diff = logits_to_logit_diff(hooked_mamba, corrupted_logits, correct_answer, incorrect_answer)
    print(f"Corrupted logit difference: {corrupted_logit_diff.item():.3f}")
    
    # Perform comprehensive activation patching
    num_positions = len(clean_tokens[0])
    num_layers = hooked_mamba.cfg.n_layers
    
    # Storage for results - separate matrices for different components
    results = {
        'resid_pre': torch.zeros((num_layers, num_positions), device=hooked_mamba.device),
        'resid_post': torch.zeros((num_layers, num_positions), device=hooked_mamba.device),
        'hidden_states': torch.zeros((num_layers, num_positions), device=hooked_mamba.device),
    }
    
    print("Starting comprehensive activation patching...")
    
    # Patch residual stream components
    for component in ['resid_pre', 'resid_post']:
        print(f"Patching {component}...")
        
        for layer in tqdm(range(num_layers), desc=f"Layers ({component})"):
            component_name = f"layers.{layer}.{component}"
            
            for position in range(num_positions):
                # Create temporary hook function
                temp_hook_fn = functools.partial(
                    position_specific_patching_hook,
                    clean_cache=clean_cache,
                    patch_position=position
                )
                
                # Run with patching hook
                patched_logits = hooked_mamba.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(component_name, temp_hook_fn)]
                )
                
                # Calculate patched logit difference
                patched_logit_diff = logits_to_logit_diff(
                    hooked_mamba, patched_logits, correct_answer, incorrect_answer
                ).detach()
                
                # Store normalized result
                if clean_logit_diff != corrupted_logit_diff:
                    normalized_effect = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                else:
                    normalized_effect = 0.0
                
                results[component][layer, position] = normalized_effect
    
    # Patch position-specific hidden states
    print("Patching position-specific hidden states...")
    for layer in tqdm(range(num_layers), desc="Layers (hidden states)"):
        for position in range(num_positions):
            hook_name = f"blocks.{layer}.hook_h.{position}"
            
            # Create temporary hook function for position-specific patching
            temp_hook_fn = functools.partial(
                position_specific_patching_hook,
                clean_cache=clean_cache
            )
            
            # Run with patching hook
            patched_logits = hooked_mamba.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, temp_hook_fn)]
            )
            
            # Calculate patched logit difference
            patched_logit_diff = logits_to_logit_diff(
                hooked_mamba, patched_logits, correct_answer, incorrect_answer
            ).detach()
            
            # Store normalized result
            if clean_logit_diff != corrupted_logit_diff:
                normalized_effect = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
            else:
                normalized_effect = 0.0
            
            results['hidden_states'][layer, position] = normalized_effect
    
    return results, hooked_mamba, clean_cache


def plot_comprehensive_patching_results(
    results: Dict[str, torch.Tensor], 
    hooked_mamba: HookedMamba, 
    clean_input,
    save_path="mamba_comprehensive_patching.png"
):
    """
    Plot comprehensive activation patching results for all components and positions.
    """
    # Convert to numpy
    results_np = {k: v.detach().cpu().numpy() for k, v in results.items()}
    
    # Get logit lens results for reference
    logit_lens_results = hooked_mamba.logit_lens(clean_input)
    preds = logit_lens_results['predictions'][0].detach().cpu().numpy()
    probs = logit_lens_results['probabilities'][0].detach().cpu().numpy()
    best_probs = np.max(probs, axis=-1)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Get token labels
    if isinstance(clean_input, str):
        tokens = hooked_mamba.to_str_tokens(clean_input)
    else:
        tokens = [str(t) for t in clean_input]
    
    # Plot activation patching results
    components = ['resid_pre', 'resid_post', 'hidden_states']
    titles = ['Residual Pre', 'Residual Post', 'Hidden States (Position-Specific)']
    
    for i, (component, title) in enumerate(zip(components, titles)):
        ax = axes[0, i]
        im = ax.imshow(results_np[component], cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add token predictions as text overlay
        for layer in range(preds.shape[0] - 1):  # Exclude final layer for space
            for pos in range(preds.shape[1]):
                if layer < results_np[component].shape[0]:
                    text_color = 'white' if abs(results_np[component][layer, pos]) > 0.5 else 'black'
                    ax.text(pos, layer, f'{preds[layer, pos]}', 
                           ha="center", va="center", color=text_color, fontsize=8)
        
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Layer')
        ax.set_title(f'{title}\n(Restoration of Clean Behavior)')
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels([f"{i}:{t}" for i, t in enumerate(tokens)], rotation=45, ha='right')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Restoration Score')
    
    # Plot logit lens results
    ax = axes[1, 0]
    im = ax.imshow(best_probs, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    for layer in range(preds.shape[0]):
        for pos in range(preds.shape[1]):
            ax.text(pos, layer, f'{preds[layer, pos]}', 
                   ha="center", va="center", color='black', fontsize=8)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    ax.set_title('Logit Lens\n(Top Prediction Probabilities)')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels([f"{i}:{t}" for i, t in enumerate(tokens)], rotation=45, ha='right')
    plt.colorbar(im, ax=ax, label='Probability')
    
    # Plot comparison of patching effects
    ax = axes[1, 1]
    max_effects = {comp: np.max(np.abs(results_np[comp]), axis=0) for comp in components}
    
    x = np.arange(len(tokens))
    width = 0.25
    for i, (comp, title) in enumerate(zip(components, ['Pre', 'Post', 'Hidden'])):
        ax.bar(x + i*width, max_effects[comp], width, label=title, alpha=0.8)
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Max |Restoration Score|')
    ax.set_title('Maximum Patching Effect by Position')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{i}:{t}" for i, t in enumerate(tokens)], rotation=45, ha='right')
    ax.legend()
    
    # Plot layer-wise summary
    ax = axes[1, 2]
    layer_effects = {comp: np.max(np.abs(results_np[comp]), axis=1) for comp in components}
    
    layers = np.arange(results_np['resid_pre'].shape[0])
    for comp, title in zip(components, ['Pre', 'Post', 'Hidden']):
        ax.plot(layers, layer_effects[comp], marker='o', label=title, linewidth=2)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Max |Restoration Score|')
    ax.set_title('Maximum Patching Effect by Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== Patching Results Summary ===")
    for comp in components:
        max_effect = np.max(np.abs(results_np[comp]))
        max_pos = np.unravel_index(np.argmax(np.abs(results_np[comp])), results_np[comp].shape)
        print(f"{comp}: Max effect = {max_effect:.3f} at layer {max_pos[0]}, position {max_pos[1]} ({tokens[max_pos[1]]})")
    
    return fig


# Example usage
if __name__ == "__main__":
    # Run the experiment
    model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    
    clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
    corrupted_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
    correct_answer = " John"
    incorrect_answer = " Mary"
    
    results, hooked_model, clean_cache = comprehensive_activation_patching_experiment(
        model=model,
        tokenizer=tokenizer,
        clean_input=clean_prompt,
        corrupted_input=corrupted_prompt,
        correct_answer=correct_answer,
        incorrect_answer=incorrect_answer
    )

    # Plot comprehensive results
    plot_comprehensive_patching_results(results, hooked_model, clean_prompt)
    
    print("Comprehensive activation patching experiment completed!")
    for component, tensor in results.items():
        print(f"{component} shape: {tensor.shape}")
        print(f"{component} max restoration score: {tensor.max().item():.3f}")
        print(f"{component} min restoration score: {tensor.min().item():.3f}")