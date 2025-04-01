#!/usr/bin/env python
# inspect_model.py - Utility for examining saved ActivationPredictor models

"""
Model Inspection Utility

This script provides functionality for inspecting saved ActivationPredictor models
(.pt files) and their associated configurations.

Example usage:
    python inspect_model.py --model-dir ./output/models/neuron_l6_n500_20250331_123456/best_model
    python inspect_model.py --model-dir ./output/models --list-all
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
from datetime import datetime

# Get the project root directory
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Try to import the architecture module
try:
    from architecture.architecture import ActivationPredictor
    from transformer_lens import HookedTransformer
    MODEL_IMPORTS_AVAILABLE = True
except ImportError:
    print("Warning: Could not import ActivationPredictor. Limited functionality available.")
    MODEL_IMPORTS_AVAILABLE = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect saved ActivationPredictor models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing the model or parent directory containing multiple models")
    parser.add_argument("--list-all", action="store_true", default=False,
                        help="List all models in the parent directory")
    parser.add_argument("--show-weights", action="store_true", default=False,
                        help="Show weight statistics (min, max, mean, std)")
    parser.add_argument("--plot-weights", action="store_true", default=False,
                        help="Generate histograms of weight distributions")
    parser.add_argument("--load-model", action="store_true", default=False,
                        help="Load the full model (requires transformer_lens)")
    parser.add_argument("--base-model", type=str, default="gpt2-small",
                        help="Base transformer model to use when loading full model")

    return parser.parse_args()

def list_models(parent_dir: str) -> List[str]:
    """List all model directories in the parent directory."""
    model_dirs = []

    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)

        # Check if it's a directory
        if os.path.isdir(item_path):
            # Check if it contains config.json (model directory)
            config_path = os.path.join(item_path, "config.json")
            if os.path.exists(config_path):
                model_dirs.append(item_path)

            # Check if it could be a model variant (best_model, final_model)
            variant_config = os.path.join(item_path, "config.json")
            if os.path.exists(variant_config):
                model_dirs.append(item_path)

            # Check subdirectories (models might be in nested directories)
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    subconfig_path = os.path.join(subitem_path, "config.json")
                    if os.path.exists(subconfig_path):
                        model_dirs.append(subitem_path)

    return model_dirs

def inspect_model_config(model_dir: str) -> Dict:
    """Inspect model configuration."""
    config_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return {}

    with open(config_path, "r") as f:
        config = json.load(f)

    return config

def inspect_weights(model_dir: str, plot: bool = False) -> Dict:
    """Inspect model weights."""
    head_path = os.path.join(model_dir, "head.pt")

    if not os.path.exists(head_path):
        print(f"Error: Model weights file not found at {head_path}")
        return {}

    # Load model weights
    weights = torch.load(head_path, map_location=torch.device('cpu'))

    # Analyze weights
    weight_stats = {}

    for name, param in weights.items():
        if param.dim() >= 1:  # Skip scalar parameters
            param_np = param.numpy()

            # Calculate statistics
            weight_stats[name] = {
                "shape": list(param.shape),
                "min": float(np.min(param_np)),
                "max": float(np.max(param_np)),
                "mean": float(np.mean(param_np)),
                "std": float(np.std(param_np)),
                "norm": float(np.linalg.norm(param_np)),
                "zero_fraction": float(np.mean(param_np == 0)),
            }

            # Generate histogram plot if requested
            if plot:
                plt.figure(figsize=(10, 6))
                plt.hist(param_np.flatten(), bins=50, alpha=0.7)
                plt.title(f"Weight Distribution: {name}")
                plt.xlabel("Weight Value")
                plt.ylabel("Frequency")
                plt.grid(alpha=0.3)

                # Add statistics annotation
                stats_text = (
                    f"Shape: {param.shape}\n"
                    f"Min: {weight_stats[name]['min']:.4f}\n"
                    f"Max: {weight_stats[name]['max']:.4f}\n"
                    f"Mean: {weight_stats[name]['mean']:.4f}\n"
                    f"Std: {weight_stats[name]['std']:.4f}"
                )

                plt.annotate(
                    stats_text,
                    xy=(0.95, 0.95),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                    bbox=dict(boxstyle="round", alpha=0.1)
                )

                # Create plots directory
                plots_dir = os.path.join(model_dir, "weight_plots")
                os.makedirs(plots_dir, exist_ok=True)

                # Save plot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(os.path.join(plots_dir, f"{name.replace('.', '_')}_{timestamp}.png"))
                plt.close()

    return weight_stats

def load_full_model(model_dir: str, base_model_name: str) -> Optional[object]:
    """Load the full ActivationPredictor model."""
    if not MODEL_IMPORTS_AVAILABLE:
        print("Error: Cannot load full model. Required modules not available.")
        return None

    try:
        # Load base model
        print(f"Loading base model: {base_model_name}")
        base_model = HookedTransformer.from_pretrained(base_model_name)

        # Load activation predictor
        print(f"Loading activation predictor from {model_dir}")
        predictor = ActivationPredictor.load(
            path=model_dir,
            base_model=base_model,
            device="cpu"  # Use CPU for inspection
        )

        return predictor
    except Exception as e:
        print(f"Error loading full model: {e}")
        return None

def print_model_summary(model) -> None:
    """Print a summary of the loaded model."""
    if not model:
        return

    print("\nModel Summary:")
    print("-" * 40)
    print(f"Target Layer: {model.target_layer}")
    print(f"Target Neuron: {model.target_neuron}")
    print(f"Layer Type: {model.layer_type}")
    print(f"Token Position: {model.token_pos}")
    print(f"Head Type: {model.head_type}")
    print(f"Feature Layer: {model.feature_layer}")

    print("\nHead Architecture:")
    if model.head_type == "classification":
        print(f"  Classification Head with {model.head.num_classes} classes")
    else:
        print(f"  Regression Head")

    print(f"  Input Dimension: {model.head.input_dim}")

    if hasattr(model.head, "hidden") and model.head.hidden is not None:
        print(f"  Hidden Layer: {model.head.hidden.in_features} -> {model.head.hidden.out_features}")

    print(f"  Dropout: {model.head.dropout.p}")

    # Try to print bin edges for classification
    if model.head_type == "classification" and model.bin_edges is not None:
        bin_edges = model.bin_edges
        print("\nClassification Bins:")
        for i in range(len(bin_edges) - 1):
            print(f"  Class {i}: [{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f})")

    # Print normalization parameters for regression
    if model.head_type == "regression":
        if model.activation_mean is not None and model.activation_std is not None:
            print("\nRegression Normalization:")
            print(f"  Mean: {model.activation_mean:.4f}")
            print(f"  Std: {model.activation_std:.4f}")

def main():
    """Main function."""
    args = parse_arguments()

    if args.list_all:
        # List all models in the parent directory
        print(f"Listing all models in {args.model_dir}:")
        model_dirs = list_models(args.model_dir)

        if not model_dirs:
            print("No models found.")
            return

        print(f"Found {len(model_dirs)} models:")
        for i, model_dir in enumerate(model_dirs):
            print(f"{i+1}. {model_dir}")

            # Try to read the config to get basic info
            config = inspect_model_config(model_dir)
            if config:
                head_type = config.get("head_type", "Unknown")
                layer = config.get("target_layer", "Unknown")
                neuron = config.get("target_neuron", "Unknown")
                print(f"   Type: {head_type}, Layer: {layer}, Neuron: {neuron}")

            print()
    else:
        # Inspect a single model
        if not os.path.exists(args.model_dir):
            print(f"Error: Model directory not found: {args.model_dir}")
            return

        print(f"Inspecting model at {args.model_dir}")

        # Load model configuration
        config = inspect_model_config(args.model_dir)
        if config:
            print("\nModel Configuration:")
            print("-" * 40)

            # Print key configuration elements
            if "head_type" in config:
                print(f"Head Type: {config['head_type']}")
            if "target_layer" in config:
                print(f"Target Layer: {config['target_layer']}")
            if "target_neuron" in config:
                print(f"Target Neuron: {config['target_neuron']}")
            if "layer_type" in config:
                print(f"Layer Type: {config['layer_type']}")
            if "token_pos" in config:
                print(f"Token Position: {config['token_pos']}")
            if "feature_layer" in config:
                print(f"Feature Layer: {config['feature_layer']}")

            # Print head configuration if available
            if "head_config" in config:
                print("\nHead Configuration:")
                for key, value in config["head_config"].items():
                    print(f"{key}: {value}")

            # Print bin edges for classification
            if "bin_edges" in config and config["bin_edges"]:
                print("\nClassification Bins:")
                bin_edges = config["bin_edges"]
                for i in range(len(bin_edges) - 1):
                    print(f"Class {i}: [{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f})")

            # Print normalization parameters for regression
            if "activation_mean" in config and "activation_std" in config:
                if config["activation_mean"] is not None and config["activation_std"] is not None:
                    print("\nRegression Normalization:")
                    print(f"Mean: {config['activation_mean']:.4f}")
                    print(f"Std: {config['activation_std']:.4f}")

        # Inspect weights if requested
        if args.show_weights:
            print("\nWeight Statistics:")
            print("-" * 40)

            weight_stats = inspect_weights(args.model_dir, plot=args.plot_weights)

            if weight_stats:
                for name, stats in weight_stats.items():
                    print(f"{name}:")
                    print(f"  Shape: {stats['shape']}")
                    print(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
                    print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                    print(f"  Norm: {stats['norm']:.4f}")
                    print(f"  Zero Fraction: {stats['zero_fraction']:.4f}")
                    print()

                if args.plot_weights:
                    plots_dir = os.path.join(args.model_dir, "weight_plots")
                    print(f"Weight distribution plots saved to {plots_dir}")

        # Load full model if requested
        if args.load_model and MODEL_IMPORTS_AVAILABLE:
            full_model = load_full_model(args.model_dir, args.base_model)

            if full_model:
                print_model_summary(full_model)

                # Check if the model can make predictions
                print("\nModel is ready for inference.")

                # Ask if user wants to test predictions on sample texts
                response = input("\nDo you want to test the model with sample texts? (y/n): ")
                if response.lower() == 'y':
                    # Define a few sample texts
                    sample_texts = [
                        "The quick brown fox jumps over the lazy dog.",
                        "Neural networks consist of layers of interconnected nodes.",
                        "Language models can generate coherent text passages.",
                        "Scientists study complex patterns in data to discover insights.",
                    ]

                    print("\nMaking predictions on sample texts:")
                    predictions = full_model.predict(
                        sample_texts,
                        return_activations=True,
                        return_uncertainties=True,
                    )

                    if len(predictions) == 3:
                        preds, actuals, uncertainties = predictions

                        for i, (text, pred, actual, uncertainty) in enumerate(
                            zip(sample_texts, preds, actuals, uncertainties)
                        ):
                            print(f"\nSample {i+1}:")
                            print(f"Text: {text}")
                            print(f"Predicted: {pred:.6f}")
                            print(f"Actual: {actual:.6f}")
                            print(f"Error: {abs(pred-actual):.6f}")
                            print(f"Uncertainty: {uncertainty:.6f}")
                    elif len(predictions) == 2:
                        preds, actuals = predictions

                        for i, (text, pred, actual) in enumerate(
                            zip(sample_texts, preds, actuals)
                        ):
                            print(f"\nSample {i+1}:")
                            print(f"Text: {text}")
                            print(f"Predicted: {pred:.6f}")
                            print(f"Actual: {actual:.6f}")
                            print(f"Error: {abs(pred-actual):.6f}")
                    else:
                        preds = predictions

                        for i, (text, pred) in enumerate(zip(sample_texts, preds)):
                            print(f"\nSample {i+1}:")
                            print(f"Text: {text}")
                            print(f"Predicted: {pred:.6f}")

if __name__ == "__main__":
    main()

