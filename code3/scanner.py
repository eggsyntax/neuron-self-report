# scanner.py
# NeuronScanner: For identifying interesting neurons

import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt

class NeuronScanner:
    """
    Scans neurons in a transformer model to identify "interesting" ones
    based on their activation patterns.
    """
    def __init__(self, 
                 model: HookedTransformer,
                 device: Optional[str] = None):
        """
        Initializes the NeuronScanner.

        Args:
            model: The HookedTransformer model from TransformerLens.
            device: The device to run the model on (e.g., 'cpu', 'cuda', 'mps'). Defaults to model.cfg.device.
        """
        self.model = model
        if device:
            self.device = device
            self.model.to(self.device)
        else:
            self.device = model.cfg.device if hasattr(model, 'cfg') and hasattr(model.cfg, 'device') else 'cpu'
        
        self.model.eval() # Ensure model is in eval mode
        print(f"NeuronScanner initialized for model: {self.model.cfg.model_name}, device: {self.device}")

    def _get_all_mlp_neuron_activations_for_texts(
        self, 
        texts: List[str], 
        target_token_position: Union[str, int] = "last",
        layers_to_scan: Optional[List[int]] = None
    ) -> Dict[Tuple[int, int], List[float]]:
        """
        Internal method to get activations for all MLP neurons across specified layers for a list of texts.
        This focuses on MLP 'post' activations, which are typically what we consider neuron firings.

        Args:
            texts: A list of input strings.
            target_token_position: The token position to extract activations from ('last' or an integer index).
            layers_to_scan: A list of layer indices to scan. If None, scans all MLP layers.

        Returns:
            A dictionary where keys are (layer_index, neuron_index) tuples and
            values are lists of activation values for that neuron across the provided texts.
        """
        if layers_to_scan is None:
            layers_to_scan = list(range(self.model.cfg.n_layers))

        # { (layer, neuron_idx): [act1, act2, ...], ... }
        neuron_activations_map: Dict[Tuple[int, int], List[float]] = {}

        for text in tqdm(texts, desc="Scanning texts for neuron activations"):
            tokens = self.model.to_tokens(text, prepend_bos=True).to(self.device)
            
            # Determine the target token index for this text
            if target_token_position == "last":
                current_target_idx = tokens.shape[1] - 1
            elif isinstance(target_token_position, int):
                current_target_idx = target_token_position
                if not (0 <= current_target_idx < tokens.shape[1]):
                    print(f"Warning: token_position index {target_token_position} out of bounds for text '{text}' with seq_len {tokens.shape[1]}. Skipping text.")
                    continue
            else:
                raise ValueError(f"Invalid target_token_position: {target_token_position}")

            hooks = []
            for layer_idx in layers_to_scan:
                # Hook point for MLP neuron activations (after the activation function)
                hook_point = get_act_name("post", layer_idx) # e.g., blocks.{layer}.mlp.hook_post
                hooks.append((hook_point, self._create_hook_fn(neuron_activations_map, layer_idx, current_target_idx)))
            
            with torch.no_grad():
                self.model.run_with_hooks(tokens, fwd_hooks=hooks)
        
        return neuron_activations_map

    def _create_hook_fn(self, neuron_activations_map: Dict[Tuple[int, int], List[float]], layer_idx: int, target_token_idx: int):
        """
        Creates a hook function for a specific layer to capture MLP neuron activations.
        """
        def hook_fn(activation_tensor, hook):
            # activation_tensor shape: [batch_size, seq_len, d_mlp]
            # We process one text at a time, so batch_size = 1
            # Get activations for the target token position
            token_activations = activation_tensor[0, target_token_idx, :].cpu().tolist() # Shape: [d_mlp]
            
            for neuron_idx, act_val in enumerate(token_activations):
                neuron_key = (layer_idx, neuron_idx)
                if neuron_key not in neuron_activations_map:
                    neuron_activations_map[neuron_key] = []
                neuron_activations_map[neuron_key].append(act_val)
        return hook_fn

    def calculate_neuron_statistics(
        self,
        neuron_activations_map: Dict[Tuple[int, int], List[float]]
    ) -> pd.DataFrame:
        """
        Calculates statistics for each neuron based on its collected activations.

        Args:
            neuron_activations_map: Dict from (layer, neuron) to list of activations.

        Returns:
            A pandas DataFrame with columns ['layer', 'neuron_index', 'variance', 
                                           'activation_range', 'mean_activation', 
                                           'median_activation', 'sparsity_score_zeros', 
                                           'sparsity_score_nonzeros'].
        """
        stats_list = []
        for (layer, neuron_idx), activations in tqdm(neuron_activations_map.items(), desc="Calculating neuron stats"):
            if not activations:
                continue
            
            acts_array = np.array(activations)
            variance = np.var(acts_array)
            act_range = np.max(acts_array) - np.min(acts_array)
            mean_act = np.mean(acts_array)
            median_act = np.median(acts_array)
            
            # Sparsity: percentage of times the neuron is zero (or close to zero)
            # and percentage of times it's non-zero.
            # Using a small epsilon for floating point comparisons might be robust,
            # but plan.md specifies "0 at least 30% ... and non-0 at least 30%".
            # So, direct comparison with 0 is intended.
            num_zeros = np.sum(acts_array == 0)
            num_non_zeros = np.sum(acts_array != 0)
            total_samples = len(acts_array)
            
            percent_zeros = (num_zeros / total_samples) * 100 if total_samples > 0 else 0
            percent_non_zeros = (num_non_zeros / total_samples) * 100 if total_samples > 0 else 0
            
            stats_list.append({
                "layer": layer,
                "neuron_index": neuron_idx,
                "variance": variance,
                "activation_range": act_range,
                "mean_activation": mean_act,
                "median_activation": median_act,
                "percent_zeros": percent_zeros,
                "percent_non_zeros": percent_non_zeros,
                "num_samples": total_samples
            })
            
        return pd.DataFrame(stats_list)

    def score_and_select_neurons(
        self,
        neuron_stats_df: pd.DataFrame,
        top_n: int = 5,
        variance_weight: float = 0.5,
        range_weight: float = 0.5
    ) -> pd.DataFrame:
        """
        Scores neurons based on statistics and selects the top N candidates.
        Candidate neurons: 0 at least 30% of the time AND non-0 at least 30% of the time.
        Score = variance_weight * normalized_variance + range_weight * normalized_range.

        Args:
            neuron_stats_df: DataFrame from calculate_neuron_statistics.
            top_n: Number of top neurons to return.
            variance_weight: Weight for variance in the score.
            range_weight: Weight for activation range in the score.

        Returns:
            A DataFrame of the top N candidate neurons with their scores.
        """
        if neuron_stats_df.empty:
            print("Warning: Neuron statistics DataFrame is empty. Cannot score neurons.")
            return pd.DataFrame()

        # Filter for candidate neurons
        candidates_df = neuron_stats_df[
            (neuron_stats_df['percent_zeros'] >= 30) & 
            (neuron_stats_df['percent_non_zeros'] >= 30)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        if candidates_df.empty:
            print("Warning: No candidate neurons found meeting the 30%/30% zero/non-zero criteria.")
            return pd.DataFrame()

        # Normalize variance and range for scoring (min-max normalization)
        # Handle cases where max == min to avoid division by zero
        var_min = candidates_df['variance'].min()
        var_max = candidates_df['variance'].max()
        range_min = candidates_df['activation_range'].min()
        range_max = candidates_df['activation_range'].max()

        if var_max > var_min:
            candidates_df['norm_variance'] = (candidates_df['variance'] - var_min) / (var_max - var_min)
        else:
            candidates_df['norm_variance'] = 0.5 # Assign a neutral value if all variances are the same

        if range_max > range_min:
            candidates_df['norm_range'] = (candidates_df['activation_range'] - range_min) / (range_max - range_min)
        else:
            candidates_df['norm_range'] = 0.5 # Assign a neutral value if all ranges are the same
            
        # Calculate score
        candidates_df['score'] = (variance_weight * candidates_df['norm_variance'] + 
                                  range_weight * candidates_df['norm_range'])
        
        # Sort by score and select top N
        top_neurons = candidates_df.sort_values(by='score', ascending=False).head(top_n)
        
        return top_neurons

    def scan(self, 
             texts_or_dataset_df: Union[List[str], pd.DataFrame],
             text_column: Optional[str] = "text", # if DataFrame is passed
             layers_to_scan: Optional[List[int]] = None,
             target_token_position: Union[str, int] = "last",
             top_n_to_display: int = 5,
             variance_weight: float = 0.5,
             range_weight: float = 0.5
            ) -> pd.DataFrame:
        """
        Main method to perform the neuron scan.
        1. Gets activations for all MLP neurons.
        2. Calculates statistics.
        3. Scores and selects top neurons.
        4. (Optionally) Displays results and allows user selection.

        Args:
            texts_or_dataset_df: Either a list of texts or a DataFrame containing texts.
            text_column: If a DataFrame is passed, the name of the column with text data.
            layers_to_scan: Specific layers to scan. If None, all MLP layers.
            target_token_position: Token position for activation extraction.
            top_n_to_display: How many top neurons to show.
            variance_weight: Weight for variance in scoring.
            range_weight: Weight for range in scoring.

        Returns:
            DataFrame of top N selected neurons with their stats and scores.
        """
        if isinstance(texts_or_dataset_df, pd.DataFrame):
            if text_column not in texts_or_dataset_df.columns:
                raise ValueError(f"Text column '{text_column}' not found in the provided DataFrame.")
            texts = texts_or_dataset_df[text_column].tolist()
        elif isinstance(texts_or_dataset_df, list):
            texts = texts_or_dataset_df
        else:
            raise TypeError("texts_or_dataset_df must be a list of strings or a pandas DataFrame.")

        if not texts:
            print("Warning: No texts provided for scanning.")
            return pd.DataFrame()

        print(f"Starting neuron scan with {len(texts)} text samples...")
        
        neuron_activations_map = self._get_all_mlp_neuron_activations_for_texts(
            texts, target_token_position, layers_to_scan
        )
        
        if not neuron_activations_map:
            print("Warning: No neuron activations were collected. Cannot proceed with scanning.")
            return pd.DataFrame()
            
        neuron_stats_df = self.calculate_neuron_statistics(neuron_activations_map)
        
        if neuron_stats_df.empty:
            print("Warning: Neuron statistics calculation resulted in an empty DataFrame.")
            return pd.DataFrame()

        top_neurons_df = self.score_and_select_neurons(
            neuron_stats_df, top_n_to_display, variance_weight, range_weight
        )

        print("\n--- Top Candidate Neurons ---")
        if not top_neurons_df.empty:
            print(top_neurons_df)
            # Here, we could add interactive selection if running in a suitable environment
            # For now, just return the top N.
        else:
            print("No suitable candidate neurons found after scoring.")
            
        return top_neurons

    def visualize_activation_distribution(
        self, 
        neuron_activations_map: Dict[Tuple[int, int], List[float]],
        layer: int, 
        neuron_index: int,
        bins: int = 50
    ):
        """
        Visualizes the activation distribution for a specific neuron.

        Args:
            neuron_activations_map: Dict from (layer, neuron) to list of activations.
            layer: Layer index of the neuron.
            neuron_index: Index of the neuron.
            bins: Number of bins for the histogram.
        """
        neuron_key = (layer, neuron_index)
        if neuron_key not in neuron_activations_map:
            print(f"Activations for neuron (Layer {layer}, Index {neuron_index}) not found.")
            return

        activations = neuron_activations_map[neuron_key]
        if not activations:
            print(f"No activations recorded for neuron (Layer {layer}, Index {neuron_index}).")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(activations, bins=bins, edgecolor='black')
        plt.title(f"Activation Distribution for Neuron (Layer {layer}, Index {neuron_index})")
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.5)
        plt.show() # In a script, this might need plt.savefig() instead or be handled by main pipeline

# Example Usage (for testing purposes)
if __name__ == '__main__':
    print("Setting up a dummy model for NeuronScanner example...")
    try:
        if torch.backends.mps.is_available():
            default_device = "mps"
        elif torch.cuda.is_available():
            default_device = "cuda"
        else:
            default_device = "cpu"

        model_name = "gpt2-small"
        model = HookedTransformer.from_pretrained(model_name, device=default_device)

        scanner = NeuronScanner(model, device=default_device)

        sample_texts_for_scan = [
            "The quick brown fox jumps over the lazy dog.",
            "Exploring the inner workings of neural networks is fascinating.",
            "TransformerLens provides powerful tools for such explorations.",
            "Let's find some interesting neurons in this model.",
            "Activation patterns can reveal much about a neuron's function.",
            "Sparsity and variance are key metrics to consider.",
            "This is another sentence to increase our sample size.",
            "The cat sat on the mat, purring softly.",
            "Computational linguistics combines computer science and language.",
            "Understanding AI is becoming increasingly important."
        ] * 5 # Repeat to get more samples for stats

        # To make it faster for example, scan only a few layers
        # layers_to_scan_example = [0, 1] 
        layers_to_scan_example = list(range(model.cfg.n_layers // 2)) # Scan first half of layers

        print(f"\nScanning neurons across layers: {layers_to_scan_example} (or all if None)...")
        
        # --- Option 1: Get all activations first, then process ---
        # This is more memory intensive but allows reusing activations for visualization
        all_mlp_activations = scanner._get_all_mlp_neuron_activations_for_texts(
            sample_texts_for_scan, 
            layers_to_scan=layers_to_scan_example,
            target_token_position="last"
        )
        
        if all_mlp_activations:
            stats_df = scanner.calculate_neuron_statistics(all_mlp_activations)
            top_neurons = pd.DataFrame() # Initialize top_neurons
            if not stats_df.empty:
                top_neurons = scanner.score_and_select_neurons(stats_df, top_n=5)
                print("\n--- Top Neurons (Calculated Step-by-Step) ---")
                if not top_neurons.empty:
                    print(top_neurons)
                    # Visualize distribution of the top neuron found
                    top_l = top_neurons.iloc[0]['layer']
                    top_idx = top_neurons.iloc[0]['neuron_index']
                    print(f"\nVisualizing activation distribution for top neuron: Layer {top_l}, Index {top_idx}")
                    scanner.visualize_activation_distribution(all_mlp_activations, top_l, top_idx)
                else:
                    print("No top neurons found from step-by-step calculation.")
            else:
                print("Statistics DataFrame is empty.")
        else:
            print("No MLP activations collected.")

        # --- Option 2: Use the main scan() method ---
        print("\n--- Running main scan() method ---")
        top_neurons_main_scan = scanner.scan(
            sample_texts_for_scan,
            layers_to_scan=layers_to_scan_example,
            target_token_position="last",
            top_n_to_display=3
        )
        # The scan method already prints the top neurons.
        # If you wanted to visualize from here, you'd need to re-run _get_all_mlp_neuron_activations_for_texts
        # or modify scan() to return the full activation map.
        # For now, the visualization is shown in Option 1.

    except Exception as e:
        print(f"An error occurred during the NeuronScanner example usage: {e}")
        import traceback
        traceback.print_exc()
        print("Ensure required libraries are installed and model can be loaded.")
