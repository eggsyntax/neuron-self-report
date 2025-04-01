# neuron_selection/scanner.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from transformers import PreTrainedTokenizerBase
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class NeuronScanner:
    def __init__(
        self, 
        model: HookedTransformer,
        tokenizer: PreTrainedTokenizerBase = None,
        device: str = "mps",
    ):
        """
        Initialize a scanner for finding neurons with diverse activation patterns.
        
        Args:
            model: TransformerLens model to scan
            tokenizer: HuggingFace tokenizer for the model (if None, use model.tokenizer)
            device: Device to run computations on (default: "mps" for M3 Max)
        """
        self.model = model
        self.tokenizer = tokenizer if tokenizer is not None else model.tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Enable caching for activation access
        self.model.use_cache_hook = True
        
        # Model architecture info
        self.n_layers = model.cfg.n_layers
        self.d_model = model.cfg.d_model
        self.d_mlp = model.cfg.d_mlp
        
    def scan_neurons(
        self, 
        texts: List[str], 
        token_pos: Union[int, str] = "last",
        layer_type: str = "mlp_out",
        top_k: int = 10,
        batch_size: int = 8,
    ) -> Dict:
        """
        Scan model for neurons with diverse activation patterns.
        
        Args:
            texts: List of input texts to scan
            token_pos: Token position to extract ("last" or specific index)
            layer_type: Type of layer to scan ("mlp_out" or "resid_post")
            top_k: Number of top neurons to return
            batch_size: Batch size for processing
        
        Returns:
            Dictionary of neuron statistics and rankings
        """
        all_activations = []
        
        # Process inputs in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Scanning neurons"):
            batch_texts = texts[i:i+batch_size]
            batch_activations = self._process_batch(batch_texts, token_pos, layer_type)
            all_activations.append(batch_activations)
            
        # Combine batches
        all_activations = torch.cat(all_activations, dim=0)
        
        # Compute statistics
        means = all_activations.mean(dim=0).cpu().numpy()
        variances = all_activations.var(dim=0).cpu().numpy()
        activation_ranges = (
            all_activations.max(dim=0)[0].cpu().numpy() - 
            all_activations.min(dim=0)[0].cpu().numpy()
        )
        
        # Score neurons based on variance and range
        neuron_scores = {}
        for layer in range(self.n_layers):
            layer_size = all_activations.shape[2]  # Get actual size from activations
            
            for neuron_idx in range(layer_size):
                # Global index
                global_idx = (layer, neuron_idx)
                
                # Get statistics
                mean = means[layer, neuron_idx]
                var = variances[layer, neuron_idx]
                act_range = activation_ranges[layer, neuron_idx]
                
                # Score based on variance and range (balance of both)
                score = var * np.log1p(act_range)
                
                neuron_scores[global_idx] = {
                    "score": score,
                    "mean": mean,
                    "variance": var,
                    "range": act_range,
                }
        
        # Sort by score
        sorted_neurons = sorted(
            neuron_scores.items(), 
            key=lambda x: x[1]["score"], 
            reverse=True
        )
        
        # Select top-k
        top_neurons = sorted_neurons[:top_k]
        
        return {
            "top_neurons": top_neurons,
            "all_neurons": neuron_scores,
            "layer_type": layer_type,
            "n_samples": len(texts),
        }
    
    def _process_batch(
        self, 
        texts: List[str], 
        token_pos: Union[int, str],
        layer_type: str,
    ) -> torch.Tensor:
        """
        Process a batch of texts and extract activations.
        
        Args:
            texts: Batch of input texts
            token_pos: Token position to extract
            layer_type: Type of layer to access
            
        Returns:
            Tensor of activations [batch_size, n_layers, neurons]
        """
        # Tokenize
        tokens = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        # Run model
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens.input_ids,
                attention_mask=tokens.attention_mask,
            )
        
        # Get sequence lengths for "last" token position
        batch_size = len(texts)
        if token_pos == "last":
            seq_lengths = tokens.attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        
        # CORRECTED: We need to check the actual dimension of the layer type
        # Get a sample activation to determine the size
        sample_activation = cache[layer_type, 0]  # First layer
        feature_dim = sample_activation.shape[-1]  # Last dimension is feature size
        
        # Now create the properly sized output tensor
        activations = torch.zeros(
            (batch_size, self.n_layers, feature_dim), 
            device=self.device
        )
        
        for layer in range(self.n_layers):
            # Get the activation tensor [batch, seq_len, neurons]
            layer_acts = cache[layer_type, layer]
            
            for b in range(batch_size):
                if token_pos == "last":
                    # Use the last non-padding token for this specific item in batch
                    t_pos = seq_lengths[b].item()
                else:
                    # Use specified position (clamped to sequence length)
                    t_pos = min(token_pos, layer_acts.shape[1] - 1)
                
                # Extract the activation at the correct position
                activations[b, layer] = layer_acts[b, t_pos]
        
        return activations
    
    def visualize_neuron(
        self, 
        texts: List[str], 
        layer: int, 
        neuron_idx: int,
        token_pos: Union[int, str] = "last",
        layer_type: str = "mlp_out",
        n_samples: int = 100,
    ):
        """
        Visualize activation distribution for a specific neuron.
        
        Args:
            texts: List of input texts
            layer: Layer index
            neuron_idx: Neuron index
            token_pos: Token position to analyze
            layer_type: Type of layer to access
            n_samples: Number of samples to visualize
        """
        if n_samples < len(texts):
            # Sample subset if needed
            sample_indices = np.random.choice(len(texts), n_samples, replace=False)
            sample_texts = [texts[i] for i in sample_indices]
        else:
            sample_texts = texts
        
        # Get activations
        activations = []
        for text in tqdm(sample_texts, desc="Extracting activations"):
            # Tokenize
            tokens = self.tokenizer(
                text, 
                return_tensors="pt",
            ).to(self.device)
            
            # Run model
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens.input_ids)
            
            # Get activation
            if token_pos == "last":
                t_pos = tokens.input_ids.shape[1] - 1
            else:
                t_pos = min(token_pos, tokens.input_ids.shape[1] - 1)
            
            activation = cache[layer_type, layer][0, t_pos, neuron_idx].item()
            activations.append(activation)
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(activations, bins=30, alpha=0.7)
        plt.title(f"Activations for {layer_type} Layer {layer}, Neuron {neuron_idx}")
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Add statistics
        mean = np.mean(activations)
        std = np.std(activations)
        min_val = np.min(activations)
        max_val = np.max(activations)
        
        stats_text = (
            f"Mean: {mean:.4f}\n"
            f"Std Dev: {std:.4f}\n"
            f"Range: [{min_val:.4f}, {max_val:.4f}]"
        )
        
        plt.annotate(
            stats_text, 
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", alpha=0.1)
        )
        
        return plt.gcf()


def test_neuron_scanner():
    """Test the NeuronScanner functionality with a small sample."""
    import os
    from transformer_lens import HookedTransformer
    
    # Use a small test model
    print("Loading model...")
    model_name = "gpt2-small"  # Small model for quick testing
    model = HookedTransformer.from_pretrained(model_name)
    tokenizer = model.tokenizer  # Get tokenizer directly from the model
    
    # Initialize scanner (use 'cpu' if 'mps' isn't available)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    scanner = NeuronScanner(model, tokenizer, device=device)
    
    # Create sample texts
    sample_texts = [
        "The cat sat on the mat.",
        "Machine learning models can be difficult to interpret.",
        "Transformers use attention mechanisms to process sequences.",
        "Neural networks have revolutionized artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    # Test scanning MLP neurons
    print("\nTesting MLP neuron scanning...")
    mlp_results = scanner.scan_neurons(
        sample_texts, 
        token_pos="last",
        layer_type="mlp_out",
        top_k=5
    )
    
    # Print top neurons
    print("\nTop MLP neurons:")
    for i, ((layer, neuron), stats) in enumerate(mlp_results["top_neurons"]):
        print(f"{i+1}. Layer {layer}, Neuron {neuron}: Score={stats['score']:.4f}, "
              f"Variance={stats['variance']:.4f}, Range={stats['range']:.4f}")
    
    # Test scanning residual stream
    print("\nTesting residual stream scanning...")
    resid_results = scanner.scan_neurons(
        sample_texts, 
        token_pos="last",
        layer_type="resid_post",
        top_k=5
    )
    
    # Print top residual stream positions
    print("\nTop residual stream positions:")
    for i, ((layer, neuron), stats) in enumerate(resid_results["top_neurons"]):
        print(f"{i+1}. Layer {layer}, Position {neuron}: Score={stats['score']:.4f}, "
              f"Variance={stats['variance']:.4f}, Range={stats['range']:.4f}")
    
    # Test visualization
    print("\nTesting neuron visualization...")
    top_layer, top_neuron = mlp_results["top_neurons"][0][0]
    fig = scanner.visualize_neuron(
        sample_texts,
        layer=top_layer,
        neuron_idx=top_neuron,
        layer_type="mlp_out"
    )
    
    # Save visualization if matplotlib is in interactive mode
    try:
        plt.savefig("top_neuron_visualization.png")
        print("Visualization saved to top_neuron_visualization.png")
    except Exception as e:
        print(f"Could not save visualization: {e}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    # Run test when the file is executed directly
    test_neuron_scanner()