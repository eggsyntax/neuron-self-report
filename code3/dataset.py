# dataset.py
# ActivationDatasetGenerator: For creating activation datasets

import torch
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple, Union
from datasets import load_dataset # For loading datasets from HuggingFace Hub

class ActivationDatasetGenerator:
    """
    Generates datasets mapping text inputs to neuron activation values from a transformer model.
    """
    def __init__(self, 
                 model: HookedTransformer, 
                 hook_point: str, # e.g., utils.get_act_name("post", layer_index) for MLP neuron output
                 neuron_layer: Optional[int] = None, # Required if hook_point is layer-specific and not fully qualified
                 neuron_index: Optional[int] = None, # Index of the neuron if hook_point refers to a specific neuron activation
                 device: Optional[str] = None):
        """
        Initializes the ActivationDatasetGenerator.

        Args:
            model: The HookedTransformer model from TransformerLens.
            hook_point: The hook point string to extract activations from (e.g., 'blocks.0.mlp.hook_post').
                        Can be obtained using `utils.get_act_name(activation_type, layer_index)`.
            neuron_layer: The layer index of the neuron. Often part of hook_point, but can be specified.
            neuron_index: The index of the neuron within its layer's activations.
                          If None, activations for the entire hook_point are considered (e.g., for a whole layer).
            device: The device to run the model on (e.g., 'cpu', 'cuda', 'mps'). Defaults to model.cfg.device.
        """
        self.model = model
        self.hook_point = hook_point
        self.neuron_layer = neuron_layer # May be redundant if hook_point is specific enough
        self.neuron_index = neuron_index
        
        if device:
            self.device = device
            self.model.to(self.device)
        else:
            self.device = model.cfg.device if hasattr(model, 'cfg') and hasattr(model.cfg, 'device') else 'cpu'
        
        # Validate hook_point and neuron_index
        if self.neuron_index is not None:
            # Try to get a sample activation to check dimensions
            try:
                # A short dummy input.
                if self.model.tokenizer is None:
                    # This should ideally be handled by ensuring the model always has a tokenizer.
                    # Forcing a tokenizer here might have side effects if the model was intentionally tokenizer-free.
                    print("Warning: Model does not have a tokenizer. Pre-validation of neuron_index might fail or be inaccurate if 'dummy text' cannot be tokenized.")

                def validation_hook_fn(act: torch.Tensor, hook: Any): # hook type can be HookPoint or str
                    # Perform the indexing to check bounds. If it fails, IndexError is raised.
                    _ = act[..., self.neuron_index] 
                    # Crucially, return the original activation to not break the ongoing forward pass
                    return act

                self.model.run_with_hooks(
                    "dummy text", # This will use the model's tokenizer
                    fwd_hooks=[(self.hook_point, validation_hook_fn)] 
                )
            except IndexError:
                # This is the expected error if neuron_index is out of bounds
                actual_dim_size = "unknown"
                try: # Try to get the actual dimension size for a better error message
                    temp_act_cache = {}
                    def get_shape_hook(act, hook):
                        temp_act_cache['shape'] = act.shape
                        return act # Return original act
                    # Use a non-empty input that the tokenizer can handle
                    # If tokenizer is None, this will fail, but we've warned above.
                    sample_input_for_shape = self.model.to_tokens("abc") if self.model.tokenizer is not None else torch.tensor([[0,1,2]])
                    self.model.run_with_hooks(sample_input_for_shape.to(self.device), fwd_hooks=[(self.hook_point, get_shape_hook)])
                    if 'shape' in temp_act_cache:
                        actual_dim_size = temp_act_cache['shape'][-1]
                except Exception as shape_e:
                    print(f"Note: Could not determine exact dimension size during IndexError handling: {shape_e}")
                    pass 
                
                err_msg = f"Neuron index {self.neuron_index} is out of bounds for hook_point {self.hook_point}."
                if actual_dim_size != "unknown":
                    err_msg += f" Activation dimension size is {actual_dim_size} (valid indices 0 to {actual_dim_size-1})."
                raise ValueError(err_msg)
            except Exception as e:
                # Other errors during the dummy forward pass
                print(f"Warning: Could not robustly pre-validate neuron_index {self.neuron_index} for hook_point {self.hook_point}. Error during dummy forward pass: {e}")
        
        print(f"ActivationDatasetGenerator initialized for model: {self.model.cfg.model_name}, hook_point: {self.hook_point}, neuron_index: {self.neuron_index}, device: {self.device}")

    # Placeholder for methods to be implemented
    def _get_activations(self, texts: List[str], token_position: Union[str, int] = "last") -> Tuple[List[torch.Tensor], List[List[str]]]:
        """
        Internal method to get activations for a list of texts.
        Returns a list of activation tensors and a list of tokenized texts.
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        all_activations = []
        all_tokens_list = []

        for text in tqdm(texts, desc="Processing texts for activations"):
            tokens = self.model.to_tokens(text, prepend_bos=True).to(self.device) # prepend_bos might be model-dependent
            
            # Cache to store the activation
            activation_cache = {}

            def hook_fn(activation_tensor, hook):
                # Detach and clone to prevent memory issues and ensure it's not part of graph
                activation_cache['activation'] = activation_tensor.detach().clone()

            # Run model with the hook
            with torch.no_grad():
                self.model.run_with_hooks(tokens, fwd_hooks=[(self.hook_point, hook_fn)])
            
            if 'activation' not in activation_cache:
                raise RuntimeError(f"Failed to capture activation from hook_point: {self.hook_point}")

            # Shape of activation_tensor is typically [batch_size, seq_len, d_model] or [batch_size, seq_len, d_mlp] etc.
            # For a specific neuron, it would be [batch_size, seq_len] after indexing
            activation = activation_cache['activation'] # [batch_size, seq_len, activation_dim]

            # Determine the target token index
            if token_position == "last":
                # Find the actual last token, not padding.
                # Assuming tokens is [batch_size, seq_len]
                # For single text processing, batch_size is 1.
                # We need to find the last non-padding token if padding is used.
                # For now, assume no padding or that the model handles it.
                # If prepend_bos=True, tokens[0,0] is BOS.
                # The actual sequence length might be less than tokens.shape[1] if there's padding.
                # For simplicity, let's assume the relevant sequence is the whole sequence for now.
                # TransformerLens models usually don't pad by default in to_tokens for single strings.
                target_idx = tokens.shape[1] - 1 
            elif isinstance(token_position, int):
                target_idx = token_position
                if not (0 <= target_idx < activation.shape[1]): # activation.shape[1] is seq_len
                    raise ValueError(f"token_position index {token_position} out of bounds for sequence length {activation.shape[1]}")
            else:
                raise ValueError(f"Invalid token_position: {token_position}. Must be 'last' or an integer index.")

            # Extract activation for the specific token position
            # activation shape: [batch_size, seq_pos, d_act]
            # We are processing one text at a time, so batch_size = 1
            token_activation = activation[0, target_idx, :] # Shape: [d_act]
            
            if self.neuron_index is not None:
                # If a specific neuron is targeted, index into the activation dimension
                neuron_activation = token_activation[self.neuron_index] # Shape: scalar
            else:
                # If no specific neuron, use the whole activation vector for that token position
                neuron_activation = token_activation # Shape: [d_act]
            
            all_activations.append(neuron_activation.cpu()) # Move to CPU
            all_tokens_list.append(self.model.to_str_tokens(tokens[0])) # Store string tokens

        return all_activations, all_tokens_list

    def generate_dataset_from_texts(self, 
                                    texts: List[str], 
                                    token_position: Union[str, int] = "last",
                                    output_csv_path: Optional[str] = None,
                                    metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Generates a dataset from a list of provided texts.

        Args:
            texts: A list of input strings.
            token_position: The token position to extract activations from ('last' or an integer index).
            output_csv_path: Optional path to save the dataset as a CSV file.
            metadata: Optional dictionary of metadata to save with the dataset.

        Returns:
            A pandas DataFrame with columns ['text', 'tokens', 'activation_value', ... (other metadata)].
        """
        activations, str_tokens_list = self._get_activations(texts, token_position)
        
        # Prepare data for DataFrame
        data = []
        for i, text in enumerate(texts):
            act_val = activations[i].item() if activations[i].ndim == 0 else activations[i].tolist()
            data.append({
                "text": text,
                "tokens": " ".join(str_tokens_list[i]), # Store space-separated string tokens
                "activation_value": act_val 
            })
            
        df = pd.DataFrame(data)

        if metadata:
            for key, value in metadata.items():
                df[key] = value # Add metadata as columns, repeating for all rows

        if output_csv_path:
            # TODO: Handle saving metadata more robustly (e.g., separate JSON or in CSV header)
            df.to_csv(output_csv_path, index=False)
            print(f"Dataset saved to {output_csv_path}")
            if metadata:
                # Save metadata to a separate JSON file as well for clarity
                meta_path = output_csv_path.replace(".csv", "_metadata.json")
                try:
                    import json
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    print(f"Metadata saved to {meta_path}")
                except Exception as e:
                    print(f"Could not save metadata to JSON: {e}")
        
        return df

    def generate_synthetic_dataset(self,
                                   num_samples: int = 2000,
                                   source_dataset_name: str = "wikipedia", # Example, can be changed
                                   source_dataset_config: Optional[str] = "20220301.en", # Example for wikipedia
                                   text_field: str = "text", # Field in the source dataset containing text
                                   min_text_length: int = 10, # Minimum length of text samples
                                   token_position: Union[str, int] = "last",
                                   output_csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generates a dataset using synthetic data from a HuggingFace dataset (e.g., Wikipedia).

        Args:
            num_samples: The number of distinct text samples to generate.
            source_dataset_name: Name of the HuggingFace dataset to use (e.g., "wikipedia", "c4").
            source_dataset_config: Configuration for the HuggingFace dataset (e.g., "20220301.en" for wikipedia).
            text_field: The field in the source dataset that contains the text.
            min_text_length: Minimum character length for a text sample to be considered.
            token_position: The token position to extract activations from.
            output_csv_path: Optional path to save the dataset as a CSV file.

        Returns:
            A pandas DataFrame.
        """
        print(f"Loading synthetic data source: {source_dataset_name} ({source_dataset_config})...")
        try:
            # Load the dataset with streaming to avoid downloading everything if it's huge
            dataset = load_dataset(source_dataset_name, source_dataset_config, split='train', streaming=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {source_dataset_name} ({source_dataset_config}): {e}")

        synthetic_texts = []
        # Iterate through the streamed dataset
        # We need to be careful with streaming as we can't shuffle directly before taking.
        # A common approach is to iterate and collect, then sample if too many, or just take first N.
        # For true diversity, one might need to sample from different parts of a very large dataset.
        
        # Let's try to get more than num_samples initially if possible, then sample, to improve diversity.
        # However, with streaming, we take what we can get sequentially.
        
        print(f"Fetching up to {num_samples} text samples from the source...")
        for example in tqdm(dataset, total=num_samples, desc="Streaming texts"):
            if len(synthetic_texts) >= num_samples:
                break
            text_sample = example.get(text_field, "")
            if isinstance(text_sample, str) and len(text_sample) >= min_text_length:
                # Basic cleaning: replace multiple newlines/spaces
                text_sample = " ".join(text_sample.split())
                synthetic_texts.append(text_sample)
        
        if not synthetic_texts:
            raise ValueError("No suitable text samples found from the source dataset.")
        
        if len(synthetic_texts) > num_samples:
            # If we fetched more, randomly sample down to num_samples
            # This step is less effective with pure streaming if we stop early.
            # Consider fetching a bit more than num_samples then sampling if diversity is key.
            # For now, if we fetched exactly num_samples or less, we use all.
            # If we implement fetching more, then:
            # import random
            # synthetic_texts = random.sample(synthetic_texts, num_samples)
            pass # Current logic takes first num_samples that meet criteria

        print(f"Generated {len(synthetic_texts)} synthetic text samples.")

        metadata = {
            "dataset_type": "synthetic",
            "source_dataset_name": source_dataset_name,
            "source_dataset_config": source_dataset_config,
            "num_samples_requested": num_samples,
            "num_samples_generated": len(synthetic_texts),
            "model_name": self.model.cfg.model_name,
            "hook_point": self.hook_point,
            "neuron_layer": self.neuron_layer,
            "neuron_index": self.neuron_index,
            "token_position": token_position
        }
        
        return self.generate_dataset_from_texts(synthetic_texts, token_position, output_csv_path, metadata)

    def balance_dataset(self, 
                        df: pd.DataFrame, 
                        column_to_balance: str = "activation_value", 
                        num_bins: int = 10) -> pd.DataFrame:
        """
        Balances the dataset based on the distribution of activation values.
        This is a placeholder for a more sophisticated balancing strategy.
        A simple approach is to bin activations and sample equally from bins.

        Args:
            df: The input DataFrame.
            column_to_balance: The name of the column containing activation values.
            num_bins: Number of bins to divide the activation range into.

        Returns:
            A (potentially) downsampled DataFrame, more balanced across activation ranges.
        """
        # This is a simplified balancing strategy. Real balancing can be complex.
        # For "candidate neurons" (0 at least 30%, non-0 at least 30%),
        # a key aspect is ensuring enough zero and non-zero examples.
        
        # If the column is already binned (e.g. for classification), this might not be needed
        # or would need a different approach. This assumes continuous values.
        if df[column_to_balance].dtype == 'object': # e.g. list if neuron_index was None
            print("Warning: Balancing for multi-dimensional activations is not implemented. Returning original DataFrame.")
            return df
        if df[column_to_balance].nunique() <= num_bins : # If few unique values (e.g. already binned)
             print("Warning: Few unique values in column_to_balance, balancing might not be effective or necessary. Returning original DataFrame.")
             return df


        # Create bins
        try:
            df['activation_bin'] = pd.cut(df[column_to_balance], bins=num_bins, labels=False, include_lowest=True)
        except Exception as e:
            print(f"Could not create bins for balancing, possibly due to uniform values. Error: {e}. Returning original DataFrame.")
            return df

        # Determine the smallest bin size (or a target size)
        min_bin_size = df['activation_bin'].value_counts().min()
        
        balanced_df = df.groupby('activation_bin', group_keys=False).apply(lambda x: x.sample(min_bin_size, random_state=42) if len(x) > min_bin_size else x)
        
        balanced_df = balanced_df.drop(columns=['activation_bin'])
        print(f"Dataset balanced. Original size: {len(df)}, Balanced size: {len(balanced_df)}")
        
        return balanced_df

    # TODO: Add methods for saving/loading datasets with metadata more robustly.
    # The current generate_dataset_from_texts and generate_synthetic_dataset handle basic CSV saving.
    # A dedicated load_dataset method would be useful.

# Example Usage (for testing purposes, typically run from the main pipeline script)
if __name__ == '__main__':
    # This example requires a model to be loaded.
    # For actual use, this class would be instantiated and used by neuron_self_report.py
    
    print("Setting up a dummy model for ActivationDatasetGenerator example...")
    try:
        # Ensure MPS is available for Mac M1/M2, otherwise use CPU
        if torch.backends.mps.is_available():
            default_device = "mps"
        elif torch.cuda.is_available():
            default_device = "cuda"
        else:
            default_device = "cpu"
        
        # Using a small model for quick testing
        model_name = "gpt2-small" # or "EleutherAI/pythia-14m" 
        model = HookedTransformer.from_pretrained(model_name, device=default_device)
        model.eval()

        # Example: Target the output of the MLP in layer 0, neuron 10
        target_layer = 0
        target_neuron_index = 10
        # For MLP neurons, the hook point is typically 'blocks.{layer}.mlp.hook_post'
        # from transformer_lens import utils
        # hook_point_name = utils.get_act_name("post", target_layer) # This is for residual stream
        hook_point_name = f"blocks.{target_layer}.mlp.hook_post"


        print(f"Using model: {model_name}, Hook: {hook_point_name}, Neuron: {target_neuron_index}, Device: {default_device}")

        generator = ActivationDatasetGenerator(model, 
                                             hook_point=hook_point_name,
                                             neuron_layer=target_layer, # Optional if hook_point is fully specific
                                             neuron_index=target_neuron_index,
                                             device=default_device)

        # 1. Generate from a list of texts
        sample_texts = [
            "Hello world, this is a test.",
            "TransformerLens is a great library for mechanistic interpretability.",
            "Let's see how this neuron activates.",
            "Another example sentence for testing purposes."
        ]
        df_from_texts = generator.generate_dataset_from_texts(sample_texts, 
                                                              token_position="last", 
                                                              output_csv_path="sample_text_activations.csv")
        print("\nDataset from provided texts:")
        print(df_from_texts.head())

        # Balance the generated dataset
        if not df_from_texts.empty:
            df_balanced_texts = generator.balance_dataset(df_from_texts)
            print("\nBalanced dataset from provided texts:")
            print(df_balanced_texts.head())


        # 2. Generate a synthetic dataset (using a small, fast-loading dataset for example)
        # Using 'glue', 'mrpc' as it's small. Replace with 'wikipedia' or 'c4' for real use.
        # Note: Wikipedia streaming can be slow to find diverse samples quickly.
        print("\nGenerating synthetic dataset (this might take a moment)...")
        try:
            df_synthetic = generator.generate_synthetic_dataset(
                num_samples=50, # Small number for quick example
                source_dataset_name="glue", 
                source_dataset_config="mrpc", # MRPC is small and has 'sentence1', 'sentence2'
                text_field="sentence1", # Use one of its text fields
                min_text_length=5,
                output_csv_path="synthetic_activations.csv"
            )
            print("\nSynthetic dataset:")
            print(df_synthetic.head())

            # Balance the synthetic dataset
            if not df_synthetic.empty:
                df_balanced_synthetic = generator.balance_dataset(df_synthetic)
                print("\nBalanced synthetic dataset:")
                print(df_balanced_synthetic.head())

        except Exception as e:
            print(f"Could not generate synthetic dataset for example: {e}")
            print("This might be due to dataset availability or network issues.")
            print("Skipping synthetic dataset generation example.")


    except Exception as e:
        print(f"An error occurred during the example usage: {e}")
        print("Please ensure you have an internet connection and the required libraries installed.")
        print("You might need to log in to HuggingFace CLI if using gated models/datasets: `huggingface-cli login`")
