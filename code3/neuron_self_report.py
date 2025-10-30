# neuron_self_report.py
# Main Pipeline: Orchestrating the entire workflow

import argparse
import json
import os
import shutil
import time # time was not used, but good to keep if planned
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, Optional, List # Added List just in case, though not in errors

from transformer_lens import HookedTransformer
from torch.utils.data import TensorDataset, DataLoader # DataLoader was not used directly, but good for future
from sklearn.model_selection import train_test_split

# Import project modules
from dataset import ActivationDatasetGenerator
from scanner import NeuronScanner
from architecture import ActivationPredictor
from trainer import PredictorTrainer

# DEFAULT_CONFIG dictionary has been removed. config.json is the single source of truth.

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a JSON configuration file.
    Raises FileNotFoundError or json.JSONDecodeError if issues occur.
    """
    try:
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        print(f"Successfully loaded configuration from {config_path}")
        return user_config
    except FileNotFoundError:
        print(f"Error: Config file {config_path} not found.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Config file {config_path} is not valid JSON. Details: {e}")
        raise

def setup_output_directory(config: Dict[str, Any]) -> str:
    """Sets up the output directory, archiving previous run if it exists."""
    output_dir = config['output_dir']
    output_dir_parent = os.path.dirname(os.path.abspath(output_dir)) # Get parent of the output_dir
    
    if os.path.exists(output_dir):
        # Archive previous output
        archive_parent_dir = os.path.join(output_dir_parent, "previous-outputs") # Relative to output_dir's parent
        if not os.path.exists(archive_parent_dir):
            os.makedirs(archive_parent_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir_name = f"{os.path.basename(os.path.normpath(output_dir))}_{timestamp}"
        archive_path = os.path.join(archive_parent_dir, archive_dir_name)
        
        print(f"Archiving existing output directory '{output_dir}' to '{archive_path}'...")
        try:
            shutil.move(output_dir, archive_path)
        except Exception as e:
            print(f"Could not archive existing output directory: {e}. Overwriting might occur or errors.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True) 
    
    config_save_path = os.path.join(output_dir, "run_config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Current run configuration saved to {config_save_path}")
    
    return output_dir

def determine_device(config_device: Optional[str]) -> str:
    if config_device:
        if config_device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS device requested but not available. Falling back to CPU.")
            return "cpu"
        if config_device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA device requested but not available. Falling back to CPU.")
            return "cpu"
        return config_device
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def main_pipeline(config_path: str):
    config = load_config(config_path)
    
    device_str = determine_device(config.get("device"))
    config["device"] = device_str
    print(f"Using device: {device_str}")

    seed = config.get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device_str == "cuda": torch.cuda.manual_seed_all(seed)

    output_dir = setup_output_directory(config)
    if config.get("wandb_run_name") is None:
        config["wandb_run_name"] = f"{config['model_name']}_L{config['neuron_layer']}N{config['neuron_index']}_{datetime.now().strftime('%Y%m%d-%H%M')}"

    print(f"Loading base model: {config['model_name']}...")
    base_model = HookedTransformer.from_pretrained(config['model_name'], device=device_str)
    base_model.eval()

    from transformer_lens.utils import get_act_name as tl_get_act_name 
    from datasets import load_dataset

    if config.get("run_neuron_scanner", False):
        print("\n--- Running Neuron Scanner ---")
        scanner = NeuronScanner(base_model, device=device_str)
        scanner.configure_output(config) 

        print(f"Loading texts for scanner from: {config['scanner_texts_source_dataset_name']}")
        scanner_texts = []
        try:
            scan_dataset_args = [config['scanner_texts_source_dataset_name']]
            if config.get('scanner_texts_source_dataset_config'): 
                scan_dataset_args.append(config['scanner_texts_source_dataset_config'])
            
            scan_hf_dataset = load_dataset(*scan_dataset_args, split='train', streaming=True)
            
            for i, example in enumerate(scan_hf_dataset):
                if len(scanner_texts) >= config['scanner_num_texts']: break
                text_sample = example.get(config['scanner_texts_text_field'], "")
                if isinstance(text_sample, str) and len(text_sample) >= config['scanner_texts_min_length']:
                    scanner_texts.append(" ".join(text_sample.split())) 
            
            if not scanner_texts: raise ValueError("No suitable texts found for scanner from the source.")
            print(f"Collected {len(scanner_texts)} texts for scanning.")
        except Exception as e:
            print(f"Error loading texts for scanner: {e}. Skipping scanner and using config values.")
            if config.get("hook_point") is None: 
                 if not (0 <= config['neuron_layer'] < base_model.cfg.n_layers):
                    raise ValueError(f"Invalid neuron_layer {config['neuron_layer']} for model with {base_model.cfg.n_layers} layers.")
                 config['hook_point'] = tl_get_act_name("post", config['neuron_layer'])
        else: 
            top_neurons_df = scanner.scan(
                texts_or_dataset_df=scanner_texts,
                layers_to_scan=config.get('scanner_layers_to_scan'),
                target_token_position=config.get('scanner_target_token_position', 'last'),
                top_n_to_display=config.get('scanner_top_n_to_display', 5),
                variance_weight=config.get('scanner_variance_weight', 0.5),
                range_weight=config.get('scanner_range_weight', 0.5)
            )
            if top_neurons_df is not None and not top_neurons_df.empty:
                if config.get("scanner_auto_select_top_neuron", True):
                    selected_neuron = top_neurons_df.iloc[0]
                    original_layer, original_index = config['neuron_layer'], config['neuron_index']
                    config['neuron_layer'], config['neuron_index'] = int(selected_neuron['layer']), int(selected_neuron['neuron_index'])
                    if original_layer != config['neuron_layer'] or original_index != config['neuron_index']:
                        print(f"Scanner auto-selected Neuron: Layer {config['neuron_layer']}, Index {config['neuron_index']}")
                        print(f"Updated target from (L{original_layer}, N{original_index}) to (L{config['neuron_layer']}, N{config['neuron_index']}) based on scanner.")
                    else:
                        print(f"Scanner confirmed config Neuron: Layer {config['neuron_layer']}, Index {config['neuron_index']}")
                else:
                    print("Manual neuron selection from scanner results is not yet implemented. Using first result or config.")
                    selected_neuron = top_neurons_df.iloc[0] 
                    config['neuron_layer'], config['neuron_index'] = int(selected_neuron['layer']), int(selected_neuron['neuron_index'])
                if not (0 <= config['neuron_layer'] < base_model.cfg.n_layers): 
                    raise ValueError(f"Invalid neuron_layer {config['neuron_layer']} from scanner for model with {base_model.cfg.n_layers} layers.")
                config['hook_point'] = tl_get_act_name("post", config['neuron_layer'])
            else: 
                print("Neuron scanner did not find any suitable candidate neurons. Using values from config.")
                if config.get("hook_point") is None: 
                    if not (0 <= config['neuron_layer'] < base_model.cfg.n_layers):
                        raise ValueError(f"Invalid neuron_layer {config['neuron_layer']} for model with {base_model.cfg.n_layers} layers.")
                    config['hook_point'] = tl_get_act_name("post", config['neuron_layer'])
    
    if config.get("hook_point") is None:
        if not (0 <= config['neuron_layer'] < base_model.cfg.n_layers):
            raise ValueError(f"Invalid neuron_layer {config['neuron_layer']} for model with {base_model.cfg.n_layers} layers.")
        config['hook_point'] = tl_get_act_name("post", config['neuron_layer'])
    
    print(f"Proceeding with Target Layer: {config['neuron_layer']}, Neuron Index: {config['neuron_index']}, Hook: {config['hook_point']}")

    print("Initializing ActivationDatasetGenerator...")
    dataset_generator = ActivationDatasetGenerator(
        model=base_model, hook_point=config['hook_point'],
        neuron_layer=config['neuron_layer'], neuron_index=config['neuron_index'], device=device_str
    )
    activation_df: Optional[pd.DataFrame] = None
    if config['data_source'] == 'synthetic':
        print("Generating synthetic dataset...")
        activation_df = dataset_generator.generate_synthetic_dataset(
            num_samples=config['num_samples'], source_dataset_name=config['synthetic_source_dataset_name'],
            source_dataset_config=config['synthetic_source_dataset_config'], text_field=config['synthetic_text_field'],
            min_text_length=config['synthetic_min_text_length'], token_position=config['token_position'], 
            output_csv_path=os.path.join(output_dir, "synthetic_activation_dataset.csv")
        )
    elif config['data_source'] == 'provided':
        if config['dataset_path'] is None or not os.path.exists(config['dataset_path']):
            raise FileNotFoundError(f"Provided dataset_path '{config['dataset_path']}' not found.")
        print(f"Loading provided dataset from: {config['dataset_path']}")
        provided_texts_df = pd.read_csv(config['dataset_path'])
        if 'text' not in provided_texts_df.columns: raise ValueError("Provided dataset CSV must contain a 'text' column.")
        texts_for_activation = provided_texts_df['text'].tolist()[:config['num_samples']] 
        activation_df = dataset_generator.generate_dataset_from_texts(
            texts_for_activation, token_position=config['token_position'],
            output_csv_path=os.path.join(output_dir, "provided_texts_activation_dataset.csv"),
            metadata={"source": "provided_csv", "original_path": config['dataset_path']}
        )
    else: raise ValueError(f"Invalid data_source: {config['data_source']}")
    if activation_df is None or activation_df.empty: raise ValueError("Dataset generation failed or resulted in an empty dataset.")
    if config.get("balance_dataset", False): 
        print("Balancing dataset...")
        activation_df = dataset_generator.balance_dataset(activation_df, num_bins=config['balance_dataset_bins'])
        activation_df.to_csv(os.path.join(output_dir, "balanced_activation_dataset.csv"), index=False)
    
    input_texts = activation_df['text'].tolist()
    if not hasattr(base_model, 'tokenizer') or base_model.tokenizer is None:
        raise AttributeError("Base model does not have a tokenizer. Cannot prepare dataset for PyTorch.")
    
    tokenizer = base_model.tokenizer
    max_len = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length is not None else 512
    
    current_pad_token_id = tokenizer.pad_token_id
    if current_pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            current_pad_token_id = tokenizer.eos_token_id
            print(f"Set tokenizer pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
        else: raise ValueError("Tokenizer does not have a pad_token_id and eos_token_id is also missing. Cannot proceed with padding.")
    if not isinstance(current_pad_token_id, int):
        raise TypeError(f"pad_token_id must be an integer, but found {current_pad_token_id} of type {type(current_pad_token_id)}")

    print("Tokenizing texts for PyTorch dataset...")
    all_input_ids = []
    for text in tqdm(input_texts, desc="Tokenizing for training"): 
        tokens = base_model.to_tokens(text, prepend_bos=True).squeeze(0) 
        if len(tokens) > max_len: tokens = tokens[:max_len]
        elif len(tokens) < max_len:
            padding_val = int(current_pad_token_id) 
            padding = torch.full((max_len - len(tokens),), padding_val, dtype=torch.long, device=tokens.device) 
            tokens = torch.cat((tokens, padding), dim=0)
        all_input_ids.append(tokens)
    input_ids_tensor = torch.stack(all_input_ids)
    
    # Target processing based on output_type
    output_type = config['output_type']
    # Ensure activation_values is a NumPy array of floats for consistent processing
    try:
        activation_values = activation_df['activation_value'].values.astype(np.float64)
    except Exception as e:
        print(f"Error converting activation_values to float64 numpy array: {e}")
        # Potentially inspect activation_df['activation_value'].dtype or sample values
        raise TypeError("Could not convert 'activation_value' column to a numeric NumPy array.")


    if output_type == 'regression':
        # Normalize targets for regression
        act_mean = np.mean(activation_values)
        act_std = np.std(activation_values)
        if act_std == 0: # Avoid division by zero if all activations are the same
            print("Warning: Activation values have zero standard deviation. Using unnormalized targets.")
            normalized_activations = activation_values 
        else:
            normalized_activations = (activation_values - act_mean) / act_std
        targets_tensor = torch.tensor(normalized_activations, dtype=torch.float32)
        config['target_normalization_mean'] = float(act_mean) # Store for potential denormalization
        config['target_normalization_std'] = float(act_std) if act_std > 0 else 1.0
        print(f"Regression targets normalized (mean={act_mean:.4f}, std={act_std:.4f})")
    elif output_type == 'classification':
        num_classes = config.get('num_classes_classification', 5)
        # Ensure bins cover the full range, including min and max
        min_val, max_val = activation_values.min(), activation_values.max() # Now should work on np.array
        if min_val == max_val: # Handle case where all activations are the same
             bins = np.linspace(min_val - 0.5, max_val + 0.5, num_classes + 1)
        else:
             bins = np.linspace(min_val, max_val + 1e-6, num_classes + 1)
        
        target_labels = pd.cut(activation_values, bins=bins, labels=False, include_lowest=True, right=True)
        # Handle potential NaNs if a value is outside bins (shouldn't happen with include_lowest and epsilon)
        target_labels = np.nan_to_num(target_labels, nan=0).astype(int) # Replace NaN with class 0
        targets_tensor = torch.tensor(target_labels, dtype=torch.long)
        print(f"Classification targets created with {num_classes} classes. Bins: {bins}")
    elif output_type == 'token_binary':
        # Ensure token_on_id and token_off_id are set in config
        if config.get("token_on_id") is None:
            config["token_on_id"] = tokenizer.encode(" on", add_special_tokens=False)[0] # Common to add space
            config["token_off_id"] = tokenizer.encode(" off", add_special_tokens=False)[0]
            print(f"Set token_on_id: {config['token_on_id']}, token_off_id: {config['token_off_id']}")

        target_labels = [config["token_on_id"] if act > 0 else config["token_off_id"] for act in activation_values]
        targets_tensor = torch.tensor(target_labels, dtype=torch.long)
    elif output_type == 'token_digit':
        # Ensure token_digit_ids are set
        if not config.get("token_digit_ids"):
            config["token_digit_ids"] = [tokenizer.encode(f" {i}", add_special_tokens=False)[0] for i in range(10)]
            print(f"Set token_digit_ids: {config['token_digit_ids']}")
        
        token_map = config["token_digit_ids"]
        if len(token_map) != 10: raise ValueError("token_digit_ids must contain 10 token IDs.")

        mean_act, std_act = np.mean(activation_values), np.std(activation_values)
        min_val_range = mean_act - 2 * std_act
        max_val_range = mean_act + 2 * std_act
        
        # Create 10 bins for the range [min_val_range, max_val_range]
        # np.digitize will assign values to bins 1 through 10 (if 9 bin edges)
        # or 0 through 9 if we use it carefully.
        # We want to map to indices 0-9 for token_map.
        if max_val_range == min_val_range: # Handle case of zero std_dev
            bins = np.linspace(min_val_range - 0.5, max_val_range + 0.5, 10) # 9 edges for 10 bins
        else:
            bins = np.linspace(min_val_range, max_val_range, 10) # 9 edges for 10 bins (0-8)
        
        # np.digitize returns indices from 1. We subtract 1 for 0-based indexing.
        # Clip values to be within the defined range before digitizing to avoid out-of-bounds.
        clipped_activations = np.clip(activation_values, min_val_range, max_val_range)
        
        # `bins` for np.digitize should be the edges. For N bins, N-1 edges.
        # If we have 10 target tokens (0-9), we need 9 bin edges to divide the space into 10 regions.
        # The Nth token corresponds to values >= bins[N-1].
        # The 0th token corresponds to values < bins[0].
        # So, if bins has 9 edges, digitize returns 0-9 for values within range.
        # Let's use 9 edges (10 bins).
        if max_val_range == min_val_range:
             bin_edges = np.linspace(min_val_range - 0.5, max_val_range + 0.5, 10 -1)
        else:
             bin_edges = np.linspace(min_val_range, max_val_range, 10 -1)


        target_indices = np.digitize(clipped_activations, bins=bin_edges, right=False) # right=False: bin[i-1] <= x < bin[i]
        target_indices = np.clip(target_indices, 0, 9) # Ensure indices are within 0-9

        target_labels = [token_map[idx] for idx in target_indices]
        targets_tensor = torch.tensor(target_labels, dtype=torch.long)
        print(f"Token_digit targets created. Range: [{min_val_range:.2f}, {max_val_range:.2f}], Bins used for digitize: {bin_edges}")
    else:
        raise ValueError(f"Unhandled output_type for target tensor creation: {output_type}")

    train_idx, val_idx = train_test_split(range(len(input_ids_tensor)), test_size=0.25, random_state=seed)
    
    train_inputs, val_inputs = input_ids_tensor[train_idx], input_ids_tensor[val_idx]
    train_targets, val_targets = targets_tensor[train_idx], targets_tensor[val_idx]
    
    train_torch_dataset = TensorDataset(train_inputs, train_targets)
    val_torch_dataset = TensorDataset(val_inputs, val_targets)
    print(f"Created PyTorch datasets. Train size: {len(train_torch_dataset)}, Val size: {len(val_torch_dataset)}")

    print("Initializing ActivationPredictor model for training...")
    
    if output_type in ['regression', 'classification'] and config.get('feature_extraction_hook_point') is None:
        last_layer_idx = base_model.cfg.n_layers - 1
        config['feature_extraction_hook_point'] = tl_get_act_name("resid_post", last_layer_idx)
        print(f"Defaulted feature_extraction_hook_point to: {config['feature_extraction_hook_point']}")

    base_model_output_dim_for_head = base_model.cfg.d_model 
    feature_extraction_hook_point_val = config.get('feature_extraction_hook_point')
    if isinstance(feature_extraction_hook_point_val, str): 
        if 'mlp.hook_post' in feature_extraction_hook_point_val:
            base_model_output_dim_for_head = base_model.cfg.d_mlp
        elif 'attn_out' in feature_extraction_hook_point_val:
            base_model_output_dim_for_head = base_model.cfg.d_model 
    config['base_model_output_dim_for_head'] = base_model_output_dim_for_head

    predictor_model = ActivationPredictor(
        base_model=base_model, prediction_head_type=output_type,
        base_model_output_dim=base_model_output_dim_for_head if output_type in ['regression', 'classification'] else None,
        num_classes=config.get('num_classes_classification') if output_type == 'classification' else None,
        device=device_str
    )

    print("Initializing PredictorTrainer...")
    trainer = PredictorTrainer(
        model=predictor_model, config=config, 
        train_dataset=train_torch_dataset, val_dataset=val_torch_dataset, device=device_str
    )
    
    print("Starting training...")
    trainer.train()

    print("Pipeline finished. Results and artifacts are in:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuron Self-Report Pipeline")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to the JSON configuration file for the pipeline.")
    args = parser.parse_args()
    main_pipeline(args.config)
