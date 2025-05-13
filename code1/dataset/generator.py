# dataset/generator.py
import torch
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Union, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformer_lens import HookedTransformer
import os
import json
from tqdm.auto import tqdm

class ActivationDataset(Dataset):
    """Dataset for activation prediction fine-tuning"""
    
    def __init__(
        self,
        inputs: List[str],
        activations: List[float],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        output_tokens: bool = False,
        num_bins: int = 10,
        force_zero_to_nine: bool = False,
    ):
        """
        Initialize dataset for activation prediction.
        
        Args:
            inputs: List of input texts
            activations: List of activation values
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
            output_tokens: Whether outputs should be tokens (True) or continuous values (False)
            num_bins: Number of bins for discretization (only used if output_tokens=True)
        """
        self.inputs = inputs
        self.activations = activations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.output_tokens = output_tokens
        self.num_bins = num_bins
        self.force_zero_to_nine = force_zero_to_nine
        
        if output_tokens:
            # Discretize activations into bins
            self.min_val = min(activations)
            self.max_val = max(activations)
            
            # Create bin edges
            self.bin_edges = np.linspace(
                self.min_val, 
                self.max_val, 
                num_bins + 1
            )
            
            # Convert activations to bin indices (0 to num_bins-1)
            self.discretized = np.digitize(activations, self.bin_edges[1:])
            
            # For token prediction, we want to ensure the values are exactly 0-9
            if force_zero_to_nine and num_bins == 10:
                # Ensure discretized values are in the range 0-9 exactly
                self.discretized = np.clip(self.discretized, 0, 9)
                print(f"Forcing discretized values to exactly 0-9 range, unique values: {np.unique(self.discretized)}")
            
            # Save mapping information
            self.bin_info = {
                "min_val": self.min_val,
                "max_val": self.max_val,
                "bin_edges": self.bin_edges.tolist(),
                "force_zero_to_nine": force_zero_to_nine,
            }
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        """Get a single sample as input_ids, attention_mask, and target"""
        text = self.inputs[idx]
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Remove batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        if self.output_tokens:
            # Return discretized target as token index
            target = torch.tensor(self.discretized[idx], dtype=torch.long)
        else:
            # Return continuous activation value
            target = torch.tensor(self.activations[idx], dtype=torch.float)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target,
        }

    def get_metadata(self):
        """Return dataset metadata"""
        return {
            "num_samples": len(self.inputs),
            "output_type": "tokens" if self.output_tokens else "continuous",
            "num_bins": self.num_bins if self.output_tokens else None,
            "bin_info": self.bin_info if self.output_tokens else None,
            "activation_stats": {
                "min": min(self.activations),
                "max": max(self.activations),
                "mean": np.mean(self.activations),
                "std": np.std(self.activations),
            }
        }

class ActivationDatasetGenerator:
    def __init__(
        self,
        model: HookedTransformer,
        tokenizer: PreTrainedTokenizerBase = None,
        device: str = "mps",
    ):
        """
        Initialize generator for activation datasets.
        
        Args:
            model: TransformerLens model
            tokenizer: HuggingFace tokenizer (if None, use model.tokenizer)
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer if tokenizer is not None else model.tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Enable caching for activation access
        self.model.use_cache_hook = True
    
    def generate_dataset(
        self,
        texts: List[str],
        layer: int,
        neuron_idx: int = None,
        layer_type: str = "mlp_out",
        token_pos: Union[int, str] = "last",
        output_tokens: bool = False,
        num_bins: int = 10,
        max_length: int = 512,
        balance_bins: bool = False,
        target_samples_per_bin: int = 100,
        min_activation: Optional[float] = None,
        max_activation: Optional[float] = None,
        force_zero_to_nine: bool = False,
    ) -> Tuple[ActivationDataset, Dict]:
        """
        Generate dataset for a specific neuron/activation.
        
        Args:
            texts: List of input texts
            layer: Layer index
            neuron_idx: Neuron index (None for residual stream norm)
            layer_type: Type of layer to access
            token_pos: Token position to extract
            output_tokens: Whether to discretize outputs
            num_bins: Number of bins for discretization
            max_length: Maximum sequence length for tokenization
            balance_bins: Whether to balance samples across bins
            target_samples_per_bin: Target number of samples per bin if balancing
            min_activation: Minimum activation to include
            max_activation: Maximum activation to include
            
        Returns:
            ActivationDataset and generation metadata
        """
        # Extract activations for all texts
        all_inputs = []
        all_activations = []
        
        for text in tqdm(texts, desc="Extracting activations"):
            # Tokenize
            tokens = self.tokenizer(
                text, 
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(self.device)
            
            # Determine token position
            if token_pos == "last":
                t_pos = tokens.attention_mask.sum() - 1  # Get last non-padding token
            else:
                t_pos = min(token_pos, tokens.input_ids.shape[1] - 1)
            
            # Run model with cache
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens.input_ids)
            
            # Extract activation
            if layer_type == "mlp_out" and neuron_idx is not None:
                activation = cache[layer_type, layer][0, t_pos, neuron_idx].item()
            elif layer_type == "resid_post" and neuron_idx is not None:
                activation = cache[layer_type, layer][0, t_pos, neuron_idx].item()
            elif layer_type == "resid_post" and neuron_idx is None:
                # Use L2 norm of the residual stream
                activation = torch.norm(
                    cache[layer_type, layer][0, t_pos], p=2
                ).item()
            else:
                raise ValueError(
                    "Invalid combination of layer_type and neuron_idx"
                )
            
            # Filter by activation range if specified
            if (min_activation is None or activation >= min_activation) and \
               (max_activation is None or activation <= max_activation):
                all_inputs.append(text)
                all_activations.append(activation)
        
        # Balance bins if requested
        if balance_bins and output_tokens:
            balanced_inputs, balanced_activations = self._balance_bins(
                all_inputs,
                all_activations,
                num_bins,
                target_samples_per_bin
            )
            all_inputs = balanced_inputs
            all_activations = balanced_activations
        
        # Create dataset
        dataset = ActivationDataset(
            all_inputs,
            all_activations,
            self.tokenizer,
            max_length=max_length,
            output_tokens=output_tokens,
            num_bins=num_bins,
            force_zero_to_nine=force_zero_to_nine,
        )
        
        # Save metadata
        metadata = {
            "layer": layer,
            "neuron_idx": neuron_idx,
            "layer_type": layer_type,
            "token_pos": token_pos if token_pos != "last" else f"last ({max_length-1})",
            "output_tokens": output_tokens,
            "num_bins": num_bins if output_tokens else None,
            "num_samples": len(all_inputs),
            "balanced": balance_bins,
            "target_samples_per_bin": target_samples_per_bin if balance_bins else None,
            "activation_stats": {
                "min": min(all_activations),
                "max": max(all_activations),
                "mean": np.mean(all_activations),
                "std": np.std(all_activations),
                "quartiles": np.percentile(all_activations, [25, 50, 75]).tolist(),
            }
        }
        
        return dataset, metadata
    
    def _balance_bins(
        self,
        inputs: List[str],
        activations: List[float],
        num_bins: int,
        target_samples_per_bin: int,
    ) -> Tuple[List[str], List[float]]:
        """
        Balance dataset across activation bins.
        
        Args:
            inputs: List of input texts
            activations: List of activation values
            num_bins: Number of bins
            target_samples_per_bin: Target samples per bin
            
        Returns:
            Balanced inputs and activations
        """
        # Create bin edges
        min_val = min(activations)
        max_val = max(activations)
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        
        # Assign each sample to a bin
        bin_indices = np.digitize(activations, bin_edges[1:])
        
        # Group samples by bin
        bins = {}
        for i, (text, activation, bin_idx) in enumerate(
            zip(inputs, activations, bin_indices)
        ):
            if bin_idx not in bins:
                bins[bin_idx] = []
            bins[bin_idx].append((text, activation))
        
        # Sample from each bin
        balanced_inputs = []
        balanced_activations = []
        
        for bin_idx, samples in bins.items():
            if len(samples) <= target_samples_per_bin:
                # Take all samples if less than target
                selected = samples
            else:
                # Randomly sample if more than target
                selected = random.sample(samples, target_samples_per_bin)
            
            # Add to balanced lists
            for text, activation in selected:
                balanced_inputs.append(text)
                balanced_activations.append(activation)
        
        return balanced_inputs, balanced_activations
    
    def save_dataset(
        self,
        dataset: ActivationDataset,
        metadata: Dict,
        output_dir: str,
        dataset_name: str,
    ):
        """
        Save dataset and metadata to disk using CSV as the primary storage format.
        
        Args:
            dataset: ActivationDataset to save
            metadata: Dataset metadata
            output_dir: Output directory
            dataset_name: Name for the dataset
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a dataframe of the dataset with all necessary information
        df = pd.DataFrame({
            "text": dataset.inputs,
            "activation": dataset.activations,
        })
        
        # Add discretized column if using tokens
        if dataset.output_tokens:
            df["bin"] = dataset.discretized
        
        # Save data
        df.to_csv(os.path.join(output_dir, f"{dataset_name}.csv"), index=False)
        
        # Enhance metadata to include all info needed to reconstruct the dataset
        enhanced_metadata = metadata.copy()
        enhanced_metadata["dataset_info"] = {
            "max_length": dataset.max_length,
            "output_tokens": dataset.output_tokens,
            "num_bins": dataset.num_bins,
        }
        
        # Add bin information if using tokens
        if dataset.output_tokens:
            enhanced_metadata["dataset_info"]["bin_info"] = dataset.bin_info
        
        # Save enhanced metadata
        with open(os.path.join(output_dir, f"{dataset_name}_metadata.json"), "w") as f:
            json.dump(enhanced_metadata, f, indent=2)
        
        print(f"Dataset saved to {output_dir}/{dataset_name}")

    def load_dataset(
        self,
        input_dir: str,
        dataset_name: str,
    ) -> Tuple[ActivationDataset, Dict]:
        """
        Load dataset and metadata from disk using CSV as the primary data source.
        
        Args:
            input_dir: Input directory
            dataset_name: Name of the dataset
            
        Returns:
            ActivationDataset and metadata
        """
        # Load metadata
        with open(os.path.join(input_dir, f"{dataset_name}_metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # Load dataset from CSV
        df = pd.read_csv(os.path.join(input_dir, f"{dataset_name}.csv"))
        
        # Extract dataset parameters from metadata
        dataset_info = metadata.get("dataset_info", {})
        max_length = dataset_info.get("max_length", 512)
        output_tokens = dataset_info.get("output_tokens", False)
        num_bins = dataset_info.get("num_bins", 10)
        
        # Reconstruct dataset
        dataset = ActivationDataset(
            inputs=df["text"].tolist(),
            activations=df["activation"].tolist(),
            tokenizer=self.tokenizer,
            max_length=max_length,
            output_tokens=output_tokens,
            num_bins=num_bins,
        )
        
        # If using tokens, we need to restore the discretization
        if output_tokens:
            if "bin" in df.columns:
                # If we have the bin column in the CSV, use it directly
                dataset.discretized = df["bin"].to_numpy()
            else:
                # Otherwise, re-discretize based on the bin_info
                bin_info = dataset_info.get("bin_info", {})
                if bin_info:
                    # Recreate bin edges and discretize
                    bin_edges = np.array(bin_info.get("bin_edges", []))
                    if len(bin_edges) > 0:
                        dataset.min_val = bin_info.get("min_val", min(dataset.activations))
                        dataset.max_val = bin_info.get("max_val", max(dataset.activations))
                        dataset.bin_edges = bin_edges
                        dataset.discretized = np.digitize(dataset.activations, dataset.bin_edges[1:])
                        dataset.bin_info = bin_info
        
        return dataset, metadata


def test_dataset_generator():
    """Test the ActivationDatasetGenerator functionality."""
    import os
    import random
    from transformer_lens import HookedTransformer
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Use a small test model
    print("Loading model...")
    model_name = "gpt2-small"  # Small model for quick testing
    model = HookedTransformer.from_pretrained(model_name)
    tokenizer = model.tokenizer  # Get tokenizer directly from the model
    
    # Initialize generator (use 'cpu' if 'mps' isn't available)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    generator = ActivationDatasetGenerator(model, tokenizer, device=device)
    
    # Create sample texts
    sample_texts = [
        "The cat sat on the mat.",
        "Machine learning models can be difficult to interpret.",
        "Transformers use attention mechanisms to process sequences.",
        "Neural networks have revolutionized artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
        "Gradient descent is an optimization algorithm.",
        "The earth revolves around the sun.",
        "The sky is blue because of Rayleigh scattering.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Photosynthesis is how plants convert sunlight into energy.",
    ]
    
    # Test dataset generation for a specific neuron
    print("\nTesting dataset generation for MLP neuron...")
    mlp_dataset, mlp_metadata = generator.generate_dataset(
        texts=sample_texts,
        layer=6,  # Middle layer
        neuron_idx=500,  # Arbitrary neuron
        layer_type="mlp_out",
        token_pos="last",
        output_tokens=True,
        num_bins=5
    )
    
    # Print dataset statistics
    print(f"\nGenerated dataset with {len(mlp_dataset)} samples")
    print(f"Activation range: [{mlp_metadata['activation_stats']['min']:.4f}, "
          f"{mlp_metadata['activation_stats']['max']:.4f}]")
    print(f"Mean activation: {mlp_metadata['activation_stats']['mean']:.4f}")
    
    # Test dataset item retrieval
    print("\nTesting dataset item retrieval:")
    sample_item = mlp_dataset[0]
    print(f"Input IDs shape: {sample_item['input_ids'].shape}")
    print(f"Attention mask shape: {sample_item['attention_mask'].shape}")
    print(f"Label: {sample_item['labels'].item()}")
    
    # Test continuous dataset generation
    print("\nTesting continuous dataset generation...")
    cont_dataset, cont_metadata = generator.generate_dataset(
        texts=sample_texts,
        layer=6,
        neuron_idx=500,
        layer_type="mlp_out",
        token_pos="last",
        output_tokens=False,
    )
    
    # Test dataset saving and loading
    print("\nTesting dataset saving and loading...")
    output_dir = "test_dataset_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset
    dataset_name = "test_mlp_dataset"
    generator.save_dataset(mlp_dataset, mlp_metadata, output_dir, dataset_name)
    
    # Load dataset
    loaded_dataset, loaded_metadata = generator.load_dataset(output_dir, dataset_name)
    
    # Verify loaded dataset
    print("\nVerifying loaded dataset:")
    print(f"Original dataset size: {len(mlp_dataset)}")
    print(f"Loaded dataset size: {len(loaded_dataset)}")
    print(f"Metadata matches: {mlp_metadata['layer'] == loaded_metadata['layer'] and mlp_metadata['neuron_idx'] == loaded_metadata['neuron_idx']}")
    
    # Test residual stream norm dataset
    print("\nTesting residual stream norm dataset generation...")
    resid_dataset, resid_metadata = generator.generate_dataset(
        texts=sample_texts,
        layer=6,
        neuron_idx=None,  # Use norm of residual stream
        layer_type="resid_post",
        token_pos="last",
        output_tokens=False,
    )
    
    print(f"Residual stream dataset size: {len(resid_dataset)}")
    print(f"Activation range: [{resid_metadata['activation_stats']['min']:.4f}, "
          f"{resid_metadata['activation_stats']['max']:.4f}]")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    # Run test when the file is executed directly
    test_dataset_generator()