import torch
import torch.nn as nn
from transformers import MambaForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Callable, Optional
import functools
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class HookedMamba:
    """
    A wrapper for HuggingFace Mamba models that adds hooks for activation patching.
    Similar to TransformerLens HookedTransformer but for Mamba models.
    """
    
    def __init__(self, model: MambaForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        
        # Storage for hooks and activations
        self.hooks = {}
        self.activation_cache = {}
        self.forward_hooks = []
        
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
    
        # Hook the final norm
        self.model.backbone.norm_f.register_forward_hook(
            self._make_cache_hook("norm_f")
        )
        
        # Hook the language model head
        self.model.lm_head.register_forward_hook(
            self._make_cache_hook("lm_head")
        )
        
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
                    # Pre-hook on the layer itself
                    return layer
                elif component == "resid_post":
                    # Forward hook on the layer itself
                    return layer
                elif component == "norm_output":
                    return layer.norm
                elif component == "mixer_output":
                    return layer.mixer
                # Add other components as needed
        
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
        
        layer_predictions = {}
        layer_probabilities = {}
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
            #print(predictions[:, layer, :].shape, layer_logits.shape)
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

def mamba_activation_patching_experiment(model = None, 
                                         tokenizer = None,
                                         clean_input="After John and Mary went to the store, Mary gave a bottle of milk to",
                                         corrupted_input="After John and Mary went to the store, John gave a bottle of milk to",
                                         correct_answer=" John",
                                         incorrect_answer=" Mary"):
    """
    Complete activation patching experiment for Mamba models.
    Based on the TransformerLens example but adapted for Mamba.
    """
    if model is None:
        model_name = "state-spaces/mamba-130m-hf"  # or your custom model
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
    
    # Define patching hook for Mamba layers
    def mamba_layer_patching_hook(
        activation: torch.Tensor,
        hook: MockHookPoint,
        position: int,
        clean_cache: Dict
    ) -> torch.Tensor:
        """
        Patch activation at specific position with clean activation.
        """
        clean_activation = clean_cache[hook.name]
        
        # Handle different activation shapes
        if activation.dim() == 3:  # [batch, seq, hidden]
            activation[:, position, :] = clean_activation[:, position, :]
        elif activation.dim() == 2:  # [batch, hidden] - for some components
            activation[:, :] = clean_activation[:, :]
        
        return activation
    
    # Perform activation patching
    num_positions = len(clean_tokens[0])
    num_layers = hooked_mamba.cfg.n_layers
    
    # Storage for results
    # NOTE - Patching results has num_layers + 1 rows to include final layer output activations
    patching_results = torch.zeros((num_layers+1, num_positions), device=hooked_mamba.device)
    
    # Components to patch (adjust based on your Mamba model structure)
    components_to_patch = [
        #"layers.{}.mixer_output",  # Mamba mixer output
        #"layers.{}.norm_output",   # Layer norm output
        "layers.{}.resid_pre", 
        #"layers.{}.resid_post",
    ]
    
    print("Starting activation patching...")
    
    for component_template in components_to_patch:
        print(f"Patching component: {component_template}")
        
        for layer in tqdm(range(num_layers), desc=f"Layers"):
            component_name = component_template.format(layer)
            
            for position in range(num_positions):
                # Create temporary hook function
                temp_hook_fn = functools.partial(
                    mamba_layer_patching_hook, 
                    position=position,
                    clean_cache=clean_cache
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
                
                patching_results[layer, position] = normalized_effect
    
    for position in range(num_positions):
        # Create temporary hook function
        temp_hook_fn = functools.partial(
            mamba_layer_patching_hook, 
            position=position,
            clean_cache=clean_cache
        )
        
        # Run with patching hook
        patched_logits = hooked_mamba.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(f"layers.{num_layers-1}.resid_post", temp_hook_fn)]
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
        
        patching_results[num_layers, position] = normalized_effect
        
    return patching_results, hooked_mamba

def visualize_patching_results(patching_results: torch.Tensor, 
                             hooked_mamba: HookedMamba, 
                             clean_input,
                             title: str = "Activation Patching Results"):
    """Visualize activation patching results."""
    
    
    # Convert to numpy
    results_np = patching_results.detach().cpu().numpy()
    
    results_logit_lens = hooked_mamba.logit_lens(clean_input)
    preds = results_logit_lens['predictions'][0].detach().cpu().numpy()
    probs = results_logit_lens['probabilities'][0].detach().cpu().numpy()
    best_probs = np.max(probs, axis=-1)
    # Create heatmap
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    im = ax[0].imshow(results_np, cmap='bwr', aspect='auto', vmin=-1, vmax=1)
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            ax[0].text(j, i, preds[i, j], ha="center", va="center", color="black")
    # Add labels
    ax[0].set_xlabel('Token Position : Token')
    ax[0].xaxis.set_label_position('top')
    ax[0].set_ylabel('Layer')
    ax[0].set_title(title)
    
    # Add token labels
    #tokens = hooked_mamba.to_str_tokens(clean_input)
    tokens = clean_input
    ax[0].set_xticks(range(len(tokens)))
    ax[0].xaxis.tick_top()
    ax[0].set_yticks(range(results_np.shape[0]))
    ax[0].set_xticklabels([f"{i} : {token}" for i, token in enumerate(tokens)])

    im1 = ax[1].imshow(best_probs, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            ax[1].text(j, i, preds[i, j], ha="center", va="center", color="black")
    # Add colorbar
    plt.colorbar(im, ax=ax[0], label='Restoration Score')
    plt.colorbar(im1, ax=ax[1], label='Probability')

    plt.tight_layout()
    plt.show()

    #plt.savefig("mamba_activation_patching_results_pos{}.png".format(corrupted_position))
    plt.savefig("mamba_activation_patching_results.png")
    return fig

# Example usage
if __name__ == "__main__":
    # Run the experiment
    model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    
    clean_prompt="After John and Mary went to the store, Mary gave a bottle of milk to"
    corrupted_prompt="After John and Mary went to the store, John gave a bottle of milk to"
    correct_answer=" John"
    incorrect_answer=" Mary"
    
    results, hooked_model = mamba_activation_patching_experiment(
        model=model,
        tokenizer=tokenizer,
        clean_prompt=clean_prompt,
        corrupted_prompt=corrupted_prompt,
        correct_answer=correct_answer,
        incorrect_answer=incorrect_answer
    )

    # Visualize results
    visualize_patching_results(results, hooked_model, clean_prompt)
    
    print("Activation patching experiment completed!")
    print(f"Results shape: {results.shape}")
    print(f"Max restoration score: {results.max().item():.3f}")
    print(f"Min restoration score: {results.min().item():.3f}")