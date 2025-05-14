# scanner.py
# NeuronScanner: For identifying interesting neurons

import os
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
        self.model = model
        if device:
            self.device = device
            self.model.to(self.device)
        else:
            self.device = model.cfg.device if hasattr(model, 'cfg') and hasattr(model.cfg, 'device') else 'cpu'
        
        self.model.eval() 
        print(f"NeuronScanner initialized for model: {self.model.cfg.model_name}, device: {self.device}")
        self.config = {} # Initialize config, can be updated by configure_output

    def _get_all_mlp_neuron_activations_for_texts(
        self, 
        texts: List[str], 
        target_token_position: Union[str, int] = "last",
        layers_to_scan: Optional[List[int]] = None
    ) -> Dict[Tuple[int, int], List[float]]:
        if layers_to_scan is None:
            layers_to_scan = list(range(self.model.cfg.n_layers))
        neuron_activations_map: Dict[Tuple[int, int], List[float]] = {}
        for text in tqdm(texts, desc="Scanning texts for neuron activations"):
            tokens = self.model.to_tokens(text, prepend_bos=True).to(self.device)
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
                hook_point = get_act_name("post", layer_idx) 
                hooks.append((hook_point, self._create_hook_fn(neuron_activations_map, layer_idx, current_target_idx)))
            with torch.no_grad():
                self.model.run_with_hooks(tokens, fwd_hooks=hooks)
        return neuron_activations_map

    def _create_hook_fn(self, neuron_activations_map: Dict[Tuple[int, int], List[float]], layer_idx: int, target_token_idx: int):
        def hook_fn(activation_tensor, hook):
            token_activations = activation_tensor[0, target_token_idx, :].cpu().tolist() 
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
        stats_list = []
        for (layer, neuron_idx), activations in tqdm(neuron_activations_map.items(), desc="Calculating neuron stats"):
            if not activations: continue
            acts_array = np.array(activations)
            stats_list.append({
                "layer": layer, "neuron_index": neuron_idx,
                "variance": np.var(acts_array),
                "activation_range": np.max(acts_array) - np.min(acts_array),
                "mean_activation": np.mean(acts_array),
                "median_activation": np.median(acts_array),
                "percent_zeros": (np.sum(acts_array == 0) / len(acts_array)) * 100 if len(acts_array) > 0 else 0,
                "percent_non_zeros": (np.sum(acts_array != 0) / len(acts_array)) * 100 if len(acts_array) > 0 else 0,
                "num_samples": len(acts_array)
            })
        return pd.DataFrame(stats_list)

    def score_and_select_neurons(
        self, neuron_stats_df: pd.DataFrame, top_n: int = 5,
        variance_weight: float = 0.5, range_weight: float = 0.5
    ) -> pd.DataFrame:
        if neuron_stats_df.empty:
            print("Warning: Neuron statistics DataFrame is empty. Cannot score neurons.")
            return pd.DataFrame()
        candidates_df = neuron_stats_df[
            (neuron_stats_df['percent_zeros'] >= 30) & 
            (neuron_stats_df['percent_non_zeros'] >= 30)
        ].copy()
        if candidates_df.empty:
            print("Warning: No candidate neurons found meeting the 30%/30% zero/non-zero criteria.")
            return pd.DataFrame()
        var_min, var_max = candidates_df['variance'].min(), candidates_df['variance'].max()
        range_min, range_max = candidates_df['activation_range'].min(), candidates_df['activation_range'].max()
        candidates_df['norm_variance'] = (candidates_df['variance'] - var_min) / (var_max - var_min) if var_max > var_min else 0.5
        candidates_df['norm_range'] = (candidates_df['activation_range'] - range_min) / (range_max - range_min) if range_max > range_min else 0.5
        candidates_df['score'] = (variance_weight * candidates_df['norm_variance'] + 
                                  range_weight * candidates_df['norm_range'])
        return candidates_df.sort_values(by='score', ascending=False).head(top_n)

    def scan(self, 
             texts_or_dataset_df: Union[List[str], pd.DataFrame],
             text_column: Optional[str] = "text", 
             layers_to_scan: Optional[List[int]] = None,
             target_token_position: Union[str, int] = "last",
             top_n_to_display: int = 5,
             variance_weight: float = 0.5, range_weight: float = 0.5
            ) -> pd.DataFrame:
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
        else:
            print("No suitable candidate neurons found after scoring.")
        return top_neurons_df

    def visualize_activation_distribution(
        self, neuron_activations_map: Dict[Tuple[int, int], List[float]],
        layer: int, neuron_index: int, bins: int = 50
    ):
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
        
        output_dir = self.config.get("output_dir", "output/scanner_visualizations")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        plot_filename = f"activation_dist_L{layer}N{neuron_index}.png"
        plot_save_path = os.path.join(output_dir, plot_filename)
        try:
            plt.savefig(plot_save_path)
            print(f"Activation distribution plot saved to {plot_save_path}")
        except Exception as e:
            print(f"Error saving activation distribution plot: {e}")
        plt.close()

    def configure_output(self, config: Dict[str, Any]):
        self.config.update(config) # Merge provided config with existing

# Example Usage (for testing purposes)
if __name__ == '__main__':
    print("Setting up a dummy model for NeuronScanner example...")
    try:
        if torch.backends.mps.is_available(): default_device = "mps"
        elif torch.cuda.is_available(): default_device = "cuda"
        else: default_device = "cpu"
        model_name = "gpt2-small"
        model = HookedTransformer.from_pretrained(model_name, device=default_device)
        scanner = NeuronScanner(model, device=default_device)
        scanner.configure_output({"output_dir": "output/scanner_example_output"})
        sample_texts_for_scan = [
            "The quick brown fox jumps over the lazy dog.",
            "Exploring the inner workings of neural networks is fascinating."
        ] * 20 
        layers_to_scan_example = list(range(model.cfg.n_layers // 2)) 
        print(f"\nScanning neurons across layers: {layers_to_scan_example} (or all if None)...")
        
        all_mlp_activations = scanner._get_all_mlp_neuron_activations_for_texts(
            sample_texts_for_scan, 
            layers_to_scan=layers_to_scan_example,
            target_token_position="last"
        )
        
        if all_mlp_activations:
            stats_df = scanner.calculate_neuron_statistics(all_mlp_activations)
            top_neurons = pd.DataFrame() 
            if not stats_df.empty:
                top_neurons = scanner.score_and_select_neurons(stats_df, top_n=5)
            
            print("\n--- Top Neurons (Calculated Step-by-Step) ---")
            if not top_neurons.empty: # top_neurons is now guaranteed to be a DataFrame
                print(top_neurons)
                top_l = top_neurons.iloc[0]['layer']
                top_idx = top_neurons.iloc[0]['neuron_index']
                print(f"\nVisualizing activation distribution for top neuron: Layer {top_l}, Index {top_idx}")
                scanner.visualize_activation_distribution(all_mlp_activations, top_l, top_idx)
            else: # This covers cases where stats_df was empty or score_and_select_neurons returned empty
                print("No top neurons found from step-by-step calculation.")
        else:
            print("No MLP activations collected.")

        print("\n--- Running main scan() method ---")
        top_neurons_main_scan = scanner.scan(
            sample_texts_for_scan,
            layers_to_scan=layers_to_scan_example,
            target_token_position="last",
            top_n_to_display=3
        )
    except Exception as e:
        print(f"An error occurred during the NeuronScanner example usage: {e}")
        import traceback
        traceback.print_exc()
        print("Ensure required libraries are installed and model can be loaded.")
