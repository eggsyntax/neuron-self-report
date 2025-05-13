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

# Define sensible defaults for configuration parameters (as per plan.md)
DEFAULT_CONFIG = {
    "model_name": "gpt2-small",
    "neuron_layer": 0, # Example, will be determined by scanner or user
    "neuron_index": 0, # Example, will be determined by scanner or user
    "hook_point": None, # To be constructed, e.g., f"blocks.{layer}.mlp.hook_post"
    "dataset_path": None, # Path to a user-provided dataset (optional)
    "learning_rate": 1e-4,
    "batch_size": 32,
    "output_type": "regression", # 'regression', 'classification', 'token_binary', 'token_digit'
    "data_source": "synthetic", # 'synthetic' or 'provided'
    "token_position": "last", # 'last' or an integer index for activation extraction for dataset
    "num_samples": 2000, # Total number of samples to use/generate
    "output_dir": "output/", # Default output directory within code3/
    "epochs": 10,
    "unfreeze_strategy": "head_only", # 'head_only', 'all_layers', 'layers_after_target', 'selective_components'
    "components_to_unfreeze": [], # For 'selective_components' strategy
    "device": None, # Auto-detect: 'mps', 'cuda', 'cpu'
    "use_wandb": False,
    "wandb_project": "neuron-self-report",
    "wandb_run_name": None, # Auto-generate if None
    "early_stopping_patience": 3,
    "random_seed": 42, # For reproducibility
    # Scanner specific defaults
    "scanner_layers_to_scan": None, # None for all layers, or list of layer indices
    "scanner_top_n_to_display": 5,
    "scanner_variance_weight": 0.5,
    "scanner_range_weight": 0.5,
    # ActivationPredictor specific
    "feature_extraction_hook_point": None, # Hook point for features fed to the predictor head
                                           # e.g. f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
    "target_token_position_for_features": "last", # Token position for features for the head
    "num_classes_classification": 5, # If output_type is 'classification'
    # Synthetic dataset generation
    "synthetic_source_dataset_name": "wikipedia",
    "synthetic_source_dataset_config": "20220301.en",
    "synthetic_text_field": "text",
    "synthetic_min_text_length": 20,
    "balance_dataset_bins": 10, # For ActivationDatasetGenerator.balance_dataset
}

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a JSON configuration file and merges with defaults."""
    try:
        with open(config_path, 'r') as f:
            user_config = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using default configuration.")
        return DEFAULT_CONFIG.copy()
    except json.JSONDecodeError:
        print(f"Error: Config file {config_path} is not valid JSON. Using default configuration.")
        return DEFAULT_CONFIG.copy()
        
    config = DEFAULT_CONFIG.copy()
    config.update(user_config) # User config overrides defaults
    return config

def setup_output_directory(config: Dict[str, Any]) -> str:
    """Sets up the output directory, archiving previous run if it exists."""
    output_dir = config['output_dir']
    
    if os.path.exists(output_dir):
        # Archive previous output
        archive_parent_dir = "previous-outputs"
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
            # If shutil.move fails (e.g. output_dir is a file), we might want to handle it.
            # For now, we'll let os.makedirs fail if output_dir is a file.
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True) # exist_ok=True in case archiving failed but dir still exists
    
    # Save the current config to the output directory
    config_save_path = os.path.join(output_dir, "run_config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Current run configuration saved to {config_save_path}")
    
    return output_dir


def determine_device(config_device: Optional[str]) -> str:
    """Determines the computation device."""
    if config_device:
        if config_device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS device requested but not available. Falling back to CPU.")
            return "cpu"
        if config_device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA device requested but not available. Falling back to CPU.")
            return "cpu"
        return config_device
    # Auto-detect
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def main_pipeline(config_path: str):
    """
    Orchestrates the end-to-end neuron self-report pipeline.
    """
    config = load_config(config_path)
    
    # Setup device
    device_str = determine_device(config.get("device"))
    config["device"] = device_str # Update config with the determined device
    print(f"Using device: {device_str}")

    # Set random seed for reproducibility
    seed = config.get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device_str == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Setup output directory
    output_dir = setup_output_directory(config)
    if config.get("wandb_run_name") is None: # Auto-generate W&B run name if not provided
        config["wandb_run_name"] = f"{config['model_name']}_L{config['neuron_layer']}N{config['neuron_index']}_{datetime.now().strftime('%Y%m%d-%H%M')}"


    # 1. Load Model
    print(f"Loading base model: {config['model_name']}...")
    base_model = HookedTransformer.from_pretrained(config['model_name'], device=device_str)
    base_model.eval() # Start in eval mode

    # 2. Neuron Selection (if not fully specified in config)
    # This step might involve using NeuronScanner or taking direct config values.
    # For now, assume neuron_layer and neuron_index are either in config or need to be found.
    
    # Construct hook_point for dataset generation if not explicitly given for that
    # This is the neuron whose activation we want to predict.
    if config.get("hook_point") is None:
        # Default to MLP post-activation hook for the target neuron
        from transformer_lens.utils import get_act_name as tl_get_act_name
        # Ensure neuron_layer is valid
        if not (0 <= config['neuron_layer'] < base_model.cfg.n_layers):
             raise ValueError(f"Invalid neuron_layer {config['neuron_layer']} for model with {base_model.cfg.n_layers} layers.")
        config['hook_point'] = tl_get_act_name("post", config['neuron_layer']) # For MLP activations
        print(f"Constructed hook_point for target neuron: {config['hook_point']}")


    # TODO: Implement NeuronScanner integration if neuron_index is not specified or needs confirmation.
    # For now, we assume neuron_layer and neuron_index are sufficiently specified in config.
    # If scanner is used:
    # scanner = NeuronScanner(base_model, device=device_str)
    # scan_texts = ... (e.g., from a small generic dataset or first few from synthetic source)
    # top_neurons_df = scanner.scan(scan_texts, ...)
    # selected_neuron = ... (user input or take top one)
    # config['neuron_layer'] = selected_neuron['layer']
    # config['neuron_index'] = selected_neuron['neuron_index']
    # config['hook_point'] = tl_get_act_name("post", config['neuron_layer'])
    print(f"Targeting Layer: {config['neuron_layer']}, Neuron Index: {config['neuron_index']} at Hook: {config['hook_point']}")


    # 3. Dataset Generation
    print("Initializing ActivationDatasetGenerator...")
    dataset_generator = ActivationDatasetGenerator(
        model=base_model,
        hook_point=config['hook_point'],
        neuron_layer=config['neuron_layer'], # Can be redundant if hook_point is specific
        neuron_index=config['neuron_index'],
        device=device_str
    )

    activation_df: Optional[pd.DataFrame] = None
    if config['data_source'] == 'synthetic':
        print("Generating synthetic dataset...")
        activation_df = dataset_generator.generate_synthetic_dataset(
            num_samples=config['num_samples'],
            source_dataset_name=config['synthetic_source_dataset_name'],
            source_dataset_config=config['synthetic_source_dataset_config'],
            text_field=config['synthetic_text_field'],
            min_text_length=config['synthetic_min_text_length'],
            token_position=config['token_position'], # For extracting activations for the dataset labels
            output_csv_path=os.path.join(output_dir, "synthetic_activation_dataset.csv")
        )
    elif config['data_source'] == 'provided':
        if config['dataset_path'] is None or not os.path.exists(config['dataset_path']):
            raise FileNotFoundError(f"Provided dataset_path '{config['dataset_path']}' not found.")
        print(f"Loading provided dataset from: {config['dataset_path']}")
        # Assuming the provided dataset is a CSV with a 'text' column
        # We still need to process it to get activations.
        provided_texts_df = pd.read_csv(config['dataset_path'])
        if 'text' not in provided_texts_df.columns:
            raise ValueError("Provided dataset CSV must contain a 'text' column.")
        
        texts_for_activation = provided_texts_df['text'].tolist()[:config['num_samples']] # Limit samples
        
        activation_df = dataset_generator.generate_dataset_from_texts(
            texts_for_activation,
            token_position=config['token_position'],
            output_csv_path=os.path.join(output_dir, "provided_texts_activation_dataset.csv"),
            metadata={"source": "provided_csv", "original_path": config['dataset_path']}
        )
    else:
        raise ValueError(f"Invalid data_source: {config['data_source']}")

    if activation_df is None or activation_df.empty:
        raise ValueError("Dataset generation failed or resulted in an empty dataset.")

    # Balance dataset if specified (simple binning for now)
    if config.get("balance_dataset", False): # Add "balance_dataset": true to config to enable
        print("Balancing dataset...")
        activation_df = dataset_generator.balance_dataset(activation_df, num_bins=config['balance_dataset_bins'])
        activation_df.to_csv(os.path.join(output_dir, "balanced_activation_dataset.csv"), index=False)

    # Prepare PyTorch Datasets
    # Inputs are tokenized text, targets are activation values (or classes/token_ids for other output_types)
    
    # TODO: Handle different target types for classification/token prediction
    # For 'token_binary'/'token_digit', targets need to be mapped to specific token IDs.
    # For 'classification', targets need to be class indices (binned activations).
    # This logic should be part of dataset preparation or ActivationDatasetGenerator.
    # For now, assume 'activation_value' is directly usable or will be processed.
    
    # Example: For regression, targets are 'activation_value'
    # For other types, this target processing needs to be more sophisticated.
    # E.g., for token_binary: targets = (activation_df['activation_value'] > 0).astype(int)
    # then map 0 to tokenizer.encode('off')[0] and 1 to tokenizer.encode('on')[0]
    # This mapping needs access to the model's tokenizer.

    # For now, let's assume a simple case for regression and prepare for that.
    # The actual input to the ActivationPredictor model will be token IDs.
    # We need to tokenize the 'text' column from activation_df.
    
    input_texts = activation_df['text'].tolist()
    # Max length for padding/truncation - should be based on model's max sequence length
    if not hasattr(base_model, 'tokenizer') or base_model.tokenizer is None:
        raise AttributeError("Base model does not have a tokenizer. Cannot prepare dataset for PyTorch.")
    
    # Now we know base_model.tokenizer is not None
    tokenizer = base_model.tokenizer
    max_len = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length is not None else 512
    
    # Tokenize all texts
    # Using padding=True, truncation=True, return_tensors="pt"
    # Note: TransformerLens to_tokens doesn't do batch tokenization with padding easily.
    # Using HuggingFace tokenizer directly for this part.

    # Ensure tokenizer has pad_token_id if it's missing (e.g. for GPT-2)
    current_pad_token_id = tokenizer.pad_token_id
    if current_pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            current_pad_token_id = tokenizer.eos_token_id
            print(f"Set tokenizer pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
        else:
            # Fallback: if no EOS token ID either, this is problematic.
            # For many models, 0 is a common choice if unk or pad is not set, but can be risky.
            # Let's raise an error if no suitable pad token ID can be found.
            raise ValueError("Tokenizer does not have a pad_token_id and eos_token_id is also missing. Cannot proceed with padding.")
    
    if not isinstance(current_pad_token_id, int):
        raise TypeError(f"pad_token_id must be an integer, but found {current_pad_token_id} of type {type(current_pad_token_id)}")


    print("Tokenizing texts for PyTorch dataset...")
    # Manually pad/truncate if using model.to_tokens in a loop, or use tokenizer directly
    # For simplicity, let's assume we can get input_ids for the model.
    # The ActivationPredictor expects input_ids.
    # The dataset should yield (input_ids_tensor, target_tensor)
    
    # This part needs careful implementation based on how ActivationDatasetGenerator stores data
    # and how ActivationPredictor expects its input.
    # Let's assume activation_df has 'text' and 'activation_value'.
    # We need to convert 'text' to input_ids for the model.
    
    # Simplified: Re-tokenize for the training data. This is not ideal if texts are long.
    # Better: Store tokenized versions in ActivationDatasetGenerator or ensure it can provide them.
    # For now, we'll re-tokenize.
    
    all_input_ids = []
    for text in tqdm(input_texts, desc="Tokenizing for training"): # tqdm was undefined, now imported
        tokens = base_model.to_tokens(text, prepend_bos=True).squeeze(0) # Remove batch dim
        # Manual padding/truncation to a fixed length (e.g., max_len)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        elif len(tokens) < max_len:
            # Ensure current_pad_token_id is an int for torch.full
            padding_val = int(current_pad_token_id) 
            padding = torch.full((max_len - len(tokens),), padding_val, dtype=torch.long, device=tokens.device) # Ensure padding is on the same device
            tokens = torch.cat((tokens, padding), dim=0)
        all_input_ids.append(tokens)
    
    input_ids_tensor = torch.stack(all_input_ids)
    
    # Targets:
    # This needs to be adapted based on config['output_type']
    if config['output_type'] == 'regression':
        targets_tensor = torch.tensor(activation_df['activation_value'].values, dtype=torch.float32)
    elif config['output_type'] == 'classification':
        # Assume 'activation_value' needs to be binned into class indices
        # This binning logic should ideally be in ActivationDatasetGenerator or a helper
        # For now, placeholder:
        # Example: pd.cut(activation_df['activation_value'], bins=config['num_classes_classification'], labels=False)
        raise NotImplementedError("Classification target processing not fully implemented in pipeline.")
    elif config['output_type'] in ['token_binary', 'token_digit']:
        # Targets should be the token IDs for 'on'/'off' or '0'-'9'.
        # This requires mapping activation_value to these token IDs.
        raise NotImplementedError("Token-based target processing not fully implemented in pipeline.")
    else:
        raise ValueError(f"Unhandled output_type for target tensor creation: {config['output_type']}")

    # Split data
    train_idx, val_idx = train_test_split(range(len(input_ids_tensor)), test_size=0.25, random_state=seed)
    
    train_inputs = input_ids_tensor[train_idx]
    val_inputs = input_ids_tensor[val_idx]
    train_targets = targets_tensor[train_idx]
    val_targets = targets_tensor[val_idx]
    
    train_torch_dataset = TensorDataset(train_inputs, train_targets)
    val_torch_dataset = TensorDataset(val_inputs, val_targets)
    print(f"Created PyTorch datasets. Train size: {len(train_torch_dataset)}, Val size: {len(val_torch_dataset)}")

    # 4. Create Prediction Model (ActivationPredictor)
    print("Initializing ActivationPredictor model for training...")
    # Determine base_model_output_dim for regression/classification heads
    # This depends on the feature_extraction_hook_point
    # If it's 'blocks.L.hook_resid_post', dim is d_model. If 'blocks.L.mlp.hook_post', dim is d_mlp.
    # This needs to be robust. For now, assume d_model if not specified.
    # A helper function in TransformerLens (like get_shape_for_hook_point) would be ideal.
    
    # Simplified: Assume feature_extraction_hook_point gives features of d_model size
    # This should be configured more precisely.
    # If config['feature_extraction_hook_point'] is like '...mlp.hook_post', then output_dim is model.cfg.d_mlp
    # If it's like '...hook_resid_post', then output_dim is model.cfg.d_model
    # For now, let's require it in config or make a simple guess.
    
    # Default feature extraction hook point if not set for regression/classification
    if config['output_type'] in ['regression', 'classification'] and config.get('feature_extraction_hook_point') is None:
        # Default to last layer's residual stream output
        last_layer_idx = base_model.cfg.n_layers - 1
        config['feature_extraction_hook_point'] = tl_get_act_name("resid_post", last_layer_idx)
        print(f"Defaulted feature_extraction_hook_point to: {config['feature_extraction_hook_point']}")

    # Determine base_model_output_dim based on the hook point
    # This is a simplification. A robust way would be to run a dummy input and check shape.
    base_model_output_dim_for_head = base_model.cfg.d_model # Default
    
    # Correctly check feature_extraction_hook_point before membership test
    feature_extraction_hook_point_val = config.get('feature_extraction_hook_point')
    if isinstance(feature_extraction_hook_point_val, str): # Ensure it's a string before 'in'
        if 'mlp.hook_post' in feature_extraction_hook_point_val:
            base_model_output_dim_for_head = base_model.cfg.d_mlp
        elif 'attn_out' in feature_extraction_hook_point_val:
            base_model_output_dim_for_head = base_model.cfg.d_model # Attn out adds to residual stream
    
    # Update config with this determined/guessed dimension if it was used
    config['base_model_output_dim_for_head'] = base_model_output_dim_for_head


    predictor_model = ActivationPredictor(
        base_model=base_model,
        prediction_head_type=config['output_type'],
        base_model_output_dim=base_model_output_dim_for_head if config['output_type'] in ['regression', 'classification'] else None,
        num_classes=config.get('num_classes_classification') if config['output_type'] == 'classification' else None,
        device=device_str
    )

    # 5. Train the Model
    print("Initializing PredictorTrainer...")
    trainer = PredictorTrainer(
        model=predictor_model,
        config=config, # Pass the full config dictionary
        train_dataset=train_torch_dataset,
        val_dataset=val_torch_dataset,
        device=device_str
    )
    
    print("Starting training...")
    trainer.train()

    # 6. Evaluate and Visualize Results (some is done in trainer, more can be added here)
    print("Pipeline finished. Results and artifacts are in:", output_dir)
    # TODO: Add final evaluation on a test set if available.
    # TODO: Generate and save final visualizations/reports.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuron Self-Report Pipeline")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to the JSON configuration file for the pipeline.")
    args = parser.parse_args()
    
    main_pipeline(args.config)
