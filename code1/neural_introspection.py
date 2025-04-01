#!/usr/bin/env python
# neural_introspection.py - End-to-end neural network interpretability pipeline

"""
Neural Introspection Pipeline

This script demonstrates the end-to-end pipeline for training a transformer model
to accurately predict (introspect) its own internal neuron activations.

The pipeline consists of four main stages:
1. Neuron Selection - Find neurons with interesting activation patterns
2. Dataset Generation - Create a dataset of text inputs and activation values
3. Model Training - Fine-tune a model to predict neuron activations
4. Evaluation - Assess and visualize prediction accuracy

Example usage:
    python neural_introspection.py --model gpt2-small --layer 6 --output-dir ./results

For more options, run:
    python neural_introspection.py --help
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from transformer_lens import HookedTransformer
from typing import List, Dict, Optional, Union, Tuple

# Import our modules
from neuron_selection.scanner import NeuronScanner
from dataset.generator import ActivationDatasetGenerator, ActivationDataset
from architecture.architecture import ActivationPredictor
from training.trainer import PredictorTrainer, load_activation_dataset_from_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("neural_introspection")

def parse_arguments():
    """
    Parse command line arguments.
    
    This function defines all command-line parameters with sensible defaults
    for the neural introspection pipeline.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Neural Introspection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument("--model", type=str, default="gpt2-small",
                        help="Name of the base transformer model")
    parser.add_argument("--layer", type=int, default=None,
                        help="Specific layer to analyze (if None, scan for most interesting)")
    parser.add_argument("--neuron", type=int, default=None,
                        help="Specific neuron to analyze (if None, scan for most interesting)")
    parser.add_argument("--layer-type", type=str, default="mlp_out", 
                        choices=["mlp_out", "resid_post"],
                        help="Type of layer to extract activations from")
    parser.add_argument("--token-pos", type=str, default="last",
                        help="Token position to analyze ('last' or specific index)")
    
    # Input data parameters
    parser.add_argument("--texts", type=str, default=None,
                        help="Path to file with sample texts (one per line)")
    parser.add_argument("--num-samples", type=int, default=750,
                        help="Number of texts to generate if --texts not provided")
    
    # Dataset parameters
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory for output files")
    parser.add_argument("--force-overwrite", action="store_true", default=False,
                        help="Force overwrite of existing output directory (no archiving)")
    
    # Training parameters
    parser.add_argument("--regression", action="store_true", default=True,
                        help="Use regression head instead of classification")
    parser.add_argument("--num-bins", type=int, default=10,
                        help="Number of bins for classification (ignored for regression)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Validation set size (fraction)")
    parser.add_argument("--test-split", type=float, default=0.15,
                        help="Test set size (fraction)")
    parser.add_argument("--early-stopping", action="store_true", default=True,
                        help="Enable early stopping during training")
    parser.add_argument("--feature-layer", type=int, default=-1,
                        help="Layer to extract features from for prediction (-1 for last layer)")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (None for auto-detection)")
    
    return parser.parse_args()

def load_or_generate_texts(texts_path: Optional[str], num_samples: int) -> List[str]:
    """
    Load texts from file or generate sample texts.
    
    Args:
        texts_path: Path to file with sample texts (one per line)
        num_samples: Number of texts to generate if texts_path is None
        
    Returns:
        List of text samples
    """
    if texts_path and os.path.exists(texts_path):
        logger.info(f"Loading texts from {texts_path}")
        with open(texts_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        # If we need more samples than the file contains, repeat as needed
        if len(texts) < num_samples:
            logger.warning(f"Only {len(texts)} texts found in file, repeating to reach {num_samples}")
            texts = (texts * ((num_samples // len(texts)) + 1))[:num_samples]
        
        # If we have more than requested, select a random subset
        elif len(texts) > num_samples:
            logger.warning(f"Found {len(texts)} texts, selecting {num_samples} randomly")
            texts = np.random.choice(texts, num_samples, replace=False).tolist()
            
        return texts
    else:
        # Generate synthetic texts 
        logger.info(f"No text file provided or file not found. Generating {num_samples} synthetic texts")
        
        # These are placeholder texts that exercise different linguistic patterns
        templates = [
            "The quick brown fox jumps over the lazy dog.",
            "In machine learning, models are trained on data to make predictions.",
            "Neural networks consist of layers of interconnected nodes.",
            "Transformers use attention mechanisms to process sequences.",
            "Language models can generate coherent text passages.",
            "The sun rises in the east and sets in the west.",
            "Scientists study complex patterns in data to discover insights.",
            "Deep learning has transformed computer vision tasks.",
            "Artificial intelligence systems continue to advance rapidly.",
            "Researchers work to make AI more interpretable and transparent.",
            "The human brain contains approximately 86 billion neurons.",
            "Climate change poses significant challenges for future generations.",
            "Quantum computing leverages principles of quantum mechanics.",
            "Renewable energy sources include solar, wind, and hydroelectric power.",
            "The internet has revolutionized how people access information.",
            "Forests are crucial ecosystems that support biodiversity.",
            "Medical research continues to develop new treatments for diseases.",
            "Philosophy examines fundamental questions about existence and knowledge.",
            "Mathematics provides tools for modeling physical phenomena.",
            "Ocean currents distribute heat around the planet."
        ]
        
        texts = []
        for _ in range(num_samples):
            # Select a random template and add some variation
            template = np.random.choice(templates)
            # Add a random suffix to create some variation
            suffix = np.random.choice([
                " This is fascinating.",
                " Many people find this interesting.",
                " This has important implications.",
                " This is a key concept to understand.",
                " This idea continues to evolve.",
                " Experts continue to debate this topic.",
                " Research in this area is ongoing.",
                " This represents a significant advancement.",
                " The implications are far-reaching.",
                " This concept has practical applications.",
                " Continued progress is expected in this field.",
                " The underlying principles are widely applicable.",
                " This approach has gained widespread adoption.",
                " Future developments may change our understanding.",
                " Historical context helps explain this phenomenon."
            ])
            texts.append(template + suffix)
            
        return texts

def select_neuron(
    model: HookedTransformer,
    texts: List[str],
    layer: Optional[int] = None,
    neuron: Optional[int] = None,
    layer_type: str = "mlp_out",
    token_pos: Union[int, str] = "last",
    device: str = "cpu",
    output_dir: str = "output"
) -> Tuple[int, int, Dict]:
    """
    Select an interesting neuron for analysis.
    
    This function either uses the specified layer/neuron or scans
    the model to find neurons with interesting activation patterns.
    
    Args:
        model: HookedTransformer model to analyze
        texts: List of input texts
        layer: Specific layer to use (if None, scan for most interesting)
        neuron: Specific neuron to use (if None, scan for most interesting)
        layer_type: Type of layer to extract activations from
        token_pos: Token position to analyze
        device: Device to run on
        output_dir: Directory for saving visualizations
        
    Returns:
        Tuple of (layer_index, neuron_index, neuron_stats)
    """
    if layer is not None and neuron is not None:
        logger.info(f"Using specified neuron: Layer {layer}, Neuron {neuron}")
        return layer, neuron, {"specified": True}
    
    logger.info(f"Scanning for interesting neurons using {len(texts)} sample texts")
    
    # Create neuron scanner
    scanner = NeuronScanner(model, device=device)
    
    # Analyze a subset of texts for efficiency
    scan_texts = texts[:min(100, len(texts))]
    
    # Scan for interesting neurons
    scan_results = scanner.scan_neurons(
        texts=scan_texts,
        token_pos=token_pos,
        layer_type=layer_type,
        top_k=10,
    )
    
    # If layer specified but neuron not specified, find most interesting neuron in that layer
    if layer is not None:
        # Filter neurons from the specified layer
        layer_neurons = [
            (layer_idx, neuron_idx, stats)
            for (layer_idx, neuron_idx), stats in scan_results["all_neurons"].items()
            if layer_idx == layer
        ]
        
        if not layer_neurons:
            logger.warning(f"No neurons found in layer {layer}, using top neuron instead")
            top_layer, top_neuron = scan_results["top_neurons"][0][0]
            top_stats = scan_results["top_neurons"][0][1]
        else:
            # Sort by score and take the top
            layer_neurons.sort(key=lambda x: x[2]["score"], reverse=True)
            top_layer, top_neuron, top_stats = layer_neurons[0]
    else:
        # Use the top neuron across all layers
        top_layer, top_neuron = scan_results["top_neurons"][0][0]
        top_stats = scan_results["top_neurons"][0][1]
    
    logger.info(f"Selected neuron: Layer {top_layer}, Neuron {top_neuron}")
    logger.info(f"  Score: {top_stats['score']:.4f}, Variance: {top_stats['variance']:.4f}, "
                f"Range: {top_stats['range']:.4f}")
    
    # Visualize the selected neuron
    fig = scanner.visualize_neuron(
        texts=scan_texts,
        layer=top_layer,
        neuron_idx=top_neuron,
        token_pos=token_pos,
        layer_type=layer_type,
    )
    
    # Save the visualization
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f"selected_neuron_l{top_layer}_n{top_neuron}.png"))
    plt.close()
    
    logger.info(f"Neuron visualization saved to {figures_dir}")
    
    return top_layer, top_neuron, top_stats

def generate_dataset(
    model: HookedTransformer,
    texts: List[str],
    layer: int,
    neuron: int,
    layer_type: str = "mlp_out",
    token_pos: Union[int, str] = "last",
    use_regression: bool = True,
    num_bins: int = 10,
    output_dir: str = "output",
    device: str = "cpu",
) -> Tuple[str, str]:
    """
    Generate a dataset for the selected neuron.
    
    This function extracts activation values for the specified neuron
    across a set of input texts, creating a dataset for training.
    
    Args:
        model: HookedTransformer model
        texts: List of input texts
        layer: Layer index
        neuron: Neuron index
        layer_type: Type of layer to extract activations from
        token_pos: Token position to analyze
        use_regression: Whether to create a regression dataset (vs. classification)
        num_bins: Number of bins for classification
        output_dir: Directory for output files
        device: Device to run on
        
    Returns:
        Tuple of (dataset_csv_path, dataset_metadata_path)
    """
    logger.info(f"Generating {'regression' if use_regression else 'classification'} dataset "
                f"for Layer {layer}, Neuron {neuron}")
    
    # Create dataset generator
    generator = ActivationDatasetGenerator(model, device=device)
    
    # Generate dataset
    dataset, metadata = generator.generate_dataset(
        texts=texts,
        layer=layer,
        neuron_idx=neuron,
        layer_type=layer_type,
        token_pos=token_pos,
        output_tokens=not use_regression,  # False for regression, True for classification
        num_bins=num_bins,
        balance_bins=not use_regression,  # Balance bins for classification
    )
    
    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"neuron_l{layer}_n{neuron}_{timestamp}"
    dataset_dir = os.path.join(output_dir, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save dataset
    generator.save_dataset(dataset, metadata, dataset_dir, dataset_name)
    
    logger.info(f"Dataset saved to {dataset_dir}/{dataset_name}")
    logger.info(f"  Samples: {len(dataset)}")
    logger.info(f"  Activation range: [{metadata['activation_stats']['min']:.4f}, "
                f"{metadata['activation_stats']['max']:.4f}]")
    logger.info(f"  Mean: {metadata['activation_stats']['mean']:.4f}, "
                f"Std: {metadata['activation_stats']['std']:.4f}")
    
    # Generate a histogram of activation values
    plt.figure(figsize=(10, 6))
    plt.hist(dataset.activations, bins=20, alpha=0.7)
    plt.title(f"Activation Distribution - Layer {layer}, Neuron {neuron}")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    
    # Add statistics
    stats_text = (
        f"Mean: {metadata['activation_stats']['mean']:.4f}\n"
        f"Std Dev: {metadata['activation_stats']['std']:.4f}\n"
        f"Min: {metadata['activation_stats']['min']:.4f}\n"
        f"Max: {metadata['activation_stats']['max']:.4f}"
    )
    
    plt.annotate(
        stats_text, 
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", alpha=0.1)
    )
    
    plt.savefig(os.path.join(dataset_dir, f"{dataset_name}_distribution.png"))
    plt.close()
    
    return (
        os.path.join(dataset_dir, f"{dataset_name}.csv"),
        os.path.join(dataset_dir, f"{dataset_name}_metadata.json")
    )

def train_model(
    model: HookedTransformer,
    dataset_csv: str,
    dataset_metadata: str,
    layer: int,
    neuron: int,
    layer_type: str = "mlp_out",
    token_pos: Union[int, str] = "last",
    feature_layer: int = -1,
    use_regression: bool = True,
    batch_size: int = 16,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    val_split: float = 0.15,
    test_split: float = 0.15,
    early_stopping: bool = True,
    output_dir: str = "output",
    device: str = "cpu",
) -> Tuple[ActivationPredictor, PredictorTrainer]:
    """
    Train a model to predict neuron activations.
    
    This function creates and trains an ActivationPredictor model
    that uses the transformer's own representations to predict
    the activation value of the target neuron.
    
    Args:
        model: HookedTransformer model
        dataset_csv: Path to dataset CSV file
        dataset_metadata: Path to dataset metadata file
        layer: Layer index of target neuron
        neuron: Neuron index of target neuron
        layer_type: Type of layer to extract activations from
        token_pos: Token position to analyze
        feature_layer: Layer to extract features from for prediction
        use_regression: Whether to use regression head
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        val_split: Validation set size (fraction)
        test_split: Test set size (fraction)
        early_stopping: Whether to use early stopping
        output_dir: Directory for output files
        device: Device to run on
        
    Returns:
        Tuple of (predictor, trainer)
    """
    logger.info(f"Training {model.cfg.model_name} to predict Layer {layer}, Neuron {neuron} activations")
    logger.info(f"  Using {'regression' if use_regression else 'classification'} head")
    logger.info(f"  Features from layer: {feature_layer}")
    
    # Load dataset
    dataset = load_activation_dataset_from_files(
        dataset_csv,
        dataset_metadata,
        model.tokenizer,
    )
    
    # Get activation statistics for scaling
    if hasattr(dataset, "bin_info") and dataset.bin_info is not None:
        bin_edges = np.array(dataset.bin_info["bin_edges"])
    else:
        bin_edges = None
    
    # Get activation mean and std for regression scaling
    activations = dataset.activations
    activation_mean = np.mean(activations)
    activation_std = np.std(activations)
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, "models", f"neuron_l{layer}_n{neuron}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create predictor model
    head_type = "regression" if use_regression else "classification"
    num_classes = dataset.num_bins if not use_regression else None
    
    # Configure the prediction head with a hidden layer
    head_config = {
        "hidden_dim": 64,
        "dropout": 0.1,
        "activation": "gelu"
    }
    
    predictor = ActivationPredictor(
        base_model=model,
        head_type=head_type,
        num_classes=num_classes,
        target_layer=layer,
        target_neuron=neuron,
        layer_type=layer_type,
        token_pos=token_pos,
        feature_layer=feature_layer,
        head_config=head_config,
        device=device,
        bin_edges=bin_edges,
        activation_mean=activation_mean,
        activation_std=activation_std,
    )
    
    # Create trainer
    trainer = PredictorTrainer(
        predictor=predictor,
        dataset=dataset,
        output_dir=model_dir,
        device=device,
    )
    
    # Prepare data
    trainer.prepare_data(
        val_split=val_split,
        test_split=test_split,
        batch_size=batch_size,
    )
    
    # Train model with more detailed configuration
    metrics = trainer.train(
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,  # L2 regularization
        lr_scheduler="cosine",  # Use cosine learning rate schedule
        lr_scheduler_kwargs={"eta_min": 1e-6},
        early_stopping=early_stopping,
        patience=5 if early_stopping else 0,
        grad_clip=1.0,  # Gradient clipping
        freeze_base_model=True,  # Only train the prediction head
    )
    
    logger.info(f"Training completed in {metrics['train_time']:.1f} seconds")
    logger.info(f"  Final training loss: {metrics['train_loss'][-1]:.4f}")
    if len(metrics.get('val_loss', [])) > 0:
        logger.info(f"  Final validation loss: {metrics['val_loss'][-1]:.4f}")
    if 'best_val_loss' in metrics and metrics['best_val_loss'] < float('inf'):
        logger.info(f"  Best validation loss: {metrics['best_val_loss']:.4f} (epoch {metrics['best_epoch']})")
    
    # Load the best model if early stopping was used
    if early_stopping and metrics.get('best_epoch', -1) > 0:
        best_model_path = os.path.join(model_dir, "best_model")
        predictor = ActivationPredictor.load(
            path=best_model_path,
            base_model=model,
            device=device,
        )
        logger.info(f"Loaded best model from epoch {metrics['best_epoch']}")
    
    return predictor, trainer

def evaluate_model(
    predictor: ActivationPredictor,
    trainer: PredictorTrainer,
    texts: List[str],
    output_dir: str = "output",
) -> Dict:
    """
    Evaluate model performance and generate visualizations.
    
    This function evaluates the trained model on the test set,
    generates visualizations of prediction accuracy, and tests
    the model on a few new examples.
    
    Args:
        predictor: Trained ActivationPredictor model
        trainer: PredictorTrainer used for training
        texts: List of input texts for custom evaluation
        output_dir: Directory for output files
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info("Evaluating model performance")
    
    # Evaluate on test set
    test_results = trainer.evaluate_and_visualize_predictions(
        split="test",
        num_samples=100,
    )
    
    logger.info(f"Test set evaluation:")
    logger.info(f"  Correlation: {test_results['correlation']:.4f}")
    logger.info(f"  MSE: {test_results['mse']:.4f}")
    
    # Generate a comprehensive prediction visualization
    plt.figure(figsize=(12, 8))
    
    # Main scatter plot
    plt.subplot(2, 2, 1)
    xs = test_results["actuals"]
    ys = test_results["predictions"]
    
    # Create scatter plot
    plt.scatter(xs, ys, alpha=0.6, c='blue')
    
    # Add diagonal reference line
    min_val = min(min(xs), min(ys))
    max_val = max(max(xs), max(ys))
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    
    plt.title(f"Predicted vs. Actual Activations")
    plt.xlabel("Actual Activation")
    plt.ylabel("Predicted Activation")
    plt.grid(alpha=0.3)
    
    # Add error histogram
    plt.subplot(2, 2, 2)
    errors = np.array(ys) - np.array(xs)
    plt.hist(errors, bins=20, alpha=0.7, color='red')
    plt.title("Prediction Errors")
    plt.xlabel("Error (Predicted - Actual)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    
    # Add activation histogram
    plt.subplot(2, 2, 3)
    plt.hist(xs, bins=20, alpha=0.7, color='green', label='Actual')
    plt.hist(ys, bins=20, alpha=0.5, color='blue', label='Predicted')
    plt.title("Activation Distributions")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add error vs. actual plot to check for bias
    plt.subplot(2, 2, 4)
    plt.scatter(xs, errors, alpha=0.6, c='purple')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title("Error vs. Actual Activation")
    plt.xlabel("Actual Activation")
    plt.ylabel("Error")
    plt.grid(alpha=0.3)
    
    # Add overall title and metrics
    plt.suptitle(
        f"Neuron Introspection Analysis (Layer {predictor.target_layer}, Neuron {predictor.target_neuron})",
        fontsize=16
    )
    
    # Add metrics text
    metrics_text = (
        f"Correlation: {test_results['correlation']:.4f}\n"
        f"MSE: {test_results['mse']:.4f}\n"
        f"Mean Error: {np.mean(errors):.4f}\n"
        f"Error Std: {np.std(errors):.4f}"
    )
    
    plt.figtext(
        0.02, 0.02,
        metrics_text,
        horizontalalignment='left',
        verticalalignment='bottom',
        bbox=dict(boxstyle="round", alpha=0.1)
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_path = os.path.join(
        figures_dir, 
        f"introspection_analysis_l{predictor.target_layer}_n{predictor.target_neuron}_{timestamp}.png"
    )
    plt.savefig(figure_path)
    plt.close()
    
    logger.info(f"Introspection analysis visualization saved to {figure_path}")
    
    # Test on a few new examples
    if texts:
        logger.info("Testing on new examples:")
        
        # Select a few diverse examples
        if len(texts) > 10:
            # Select evenly spaced examples to get a diverse sample
            indices = np.linspace(0, len(texts)-1, 5, dtype=int)
            test_sample = [texts[i] for i in indices]
        else:
            test_sample = texts[:5]
        
        # Get predictions and actual values
        predictions, activations = predictor.predict(
            test_sample,
            return_activations=True,
        )
        
        # Display results
        for i, (text, pred, actual) in enumerate(zip(test_sample, predictions, activations)):
            logger.info(f"Example {i+1}:")
            # Truncate long texts for display
            display_text = text if len(text) <= 70 else text[:67] + "..."
            logger.info(f"  Text: {display_text}")
            logger.info(f"  Predicted: {pred:.6f}, Actual: {actual:.6f}, Error: {abs(pred-actual):.6f}")
    
    # Generate a detailed report
    report_path = os.path.join(output_dir, f"introspection_report_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write(f"Neural Introspection Report\n")
        f.write(f"=========================\n\n")
        
        f.write(f"Target Neuron Information\n")
        f.write(f"------------------------\n")
        f.write(f"Model: {predictor.base_model.cfg.model_name}\n")
        f.write(f"Layer: {predictor.target_layer}\n")
        f.write(f"Neuron: {predictor.target_neuron}\n")
        f.write(f"Layer Type: {predictor.layer_type}\n")
        f.write(f"Feature Layer: {predictor.feature_layer}\n")
        f.write(f"Head Type: {predictor.head_type}\n\n")
        
        f.write(f"Performance Metrics\n")
        f.write(f"------------------\n")
        f.write(f"Correlation: {test_results['correlation']:.6f}\n")
        f.write(f"MSE: {test_results['mse']:.6f}\n")
        f.write(f"Mean Error: {np.mean(errors):.6f}\n")
        f.write(f"Error Std: {np.std(errors):.6f}\n\n")
        
        # Add sample predictions
        f.write(f"Sample Predictions (Test Set)\n")
        f.write(f"----------------------------\n")
        
        # Include 10 random samples from the test set
        sample_indices = np.random.choice(len(test_results["actuals"]), 
                                         min(10, len(test_results["actuals"])), 
                                         replace=False)
        
        for i, idx in enumerate(sample_indices):
            actual = test_results["actuals"][idx]
            predicted = test_results["predictions"][idx]
            error = predicted - actual
            text = test_results["texts"][idx]
            
            # Truncate long texts for display
            display_text = text if len(text) <= 70 else text[:67] + "..."
            
            f.write(f"Sample {i+1}:\n")
            f.write(f"  Text: {display_text}\n")
            f.write(f"  Actual: {actual:.6f}\n")
            f.write(f"  Predicted: {predicted:.6f}\n")
            f.write(f"  Error: {error:.6f}\n\n")
    
    logger.info(f"Introspection report saved to {report_path}")
    
    # Return combined results
    return {
        "test_results": test_results,
        "layer": predictor.target_layer,
        "neuron": predictor.target_neuron,
        "correlation": test_results["correlation"],
        "mse": test_results["mse"],
        "report_path": report_path,
        "figure_path": figure_path,
    }

def archive_existing_output(output_dir: str, force_overwrite: bool = False) -> bool:
    """
    Archive existing output directory if it exists.
    
    This function checks if the output directory exists and contains any content.
    If it does, it moves the entire contents to an archive directory with a timestamp.
    
    Args:
        output_dir: Path to the output directory
        force_overwrite: Whether to overwrite without archiving
        
    Returns:
        Boolean indicating whether archiving was performed
    """
    # If directory doesn't exist, nothing to archive
    if not os.path.exists(output_dir):
        return False
    
    # Check if directory has any content
    dir_contents = os.listdir(output_dir)
    if not dir_contents:
        return False  # Directory exists but is empty
    
    # If force overwrite is enabled, just return
    if force_overwrite:
        print(f"Warning: Overwriting existing output in {output_dir} (--force-overwrite enabled)")
        return False
    
    # Create archive directory if it doesn't exist
    archive_root = os.path.join(os.path.dirname(output_dir), "output-archive")
    os.makedirs(archive_root, exist_ok=True)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(archive_root, f"{os.path.basename(output_dir)}_{timestamp}")
    
    # Move all contents to archive directory
    import shutil
    
    # First create the archive directory
    os.makedirs(archive_dir, exist_ok=True)
    
    # Move each item individually
    for item in dir_contents:
        source = os.path.join(output_dir, item)
        destination = os.path.join(archive_dir, item)
        
        try:
            # Handle both files and directories
            if os.path.isdir(source):
                shutil.copytree(source, destination)
                shutil.rmtree(source)
            else:
                shutil.copy2(source, destination)
                os.remove(source)
        except Exception as e:
            print(f"Warning: Failed to archive {item}: {e}")
    
    print(f"Archived existing output to {archive_dir}")
    return True

def main():
    """
    Main pipeline function.
    
    This function orchestrates the entire neural introspection pipeline:
    1. Parse arguments and set up the environment
    2. Select an interesting neuron
    3. Generate a dataset for that neuron
    4. Train a model to predict activations
    5. Evaluate and visualize results
    """
    # Parse arguments
    args = parse_arguments()
    
    # Archive existing output if necessary
    archive_existing_output(args.output_dir, args.force_overwrite)
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"introspection_log_{timestamp}.log")
    
    # Add file handler to logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Neural introspection pipeline started")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Determine device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model = HookedTransformer.from_pretrained(args.model)
    model.to(device)
    model.eval()
    
    # Enable activation caching for efficient access
    model.use_cache_hook = True
    
    # Log model architecture
    logger.info(f"Model architecture: {args.model}")
    logger.info(f"  Layers: {model.cfg.n_layers}")
    logger.info(f"  Hidden size: {model.cfg.d_model}")
    logger.info(f"  MLP intermediate size: {model.cfg.d_mlp}")
    
    # Load or generate sample texts
    texts = load_or_generate_texts(args.texts, args.num_samples)
    logger.info(f"Loaded {len(texts)} sample texts")
    
    # Select neuron to analyze
    layer, neuron, stats = select_neuron(
        model=model,
        texts=texts,
        layer=args.layer,
        neuron=args.neuron,
        layer_type=args.layer_type,
        token_pos=args.token_pos,
        device=device,
        output_dir=args.output_dir
    )
    
    # Generate dataset
    dataset_csv, dataset_metadata = generate_dataset(
        model=model,
        texts=texts,
        layer=layer,
        neuron=neuron,
        layer_type=args.layer_type,
        token_pos=args.token_pos,
        use_regression=args.regression,
        num_bins=args.num_bins,
        output_dir=args.output_dir,
        device=device,
    )
    
    # Train model
    predictor, trainer = train_model(
        model=model,
        dataset_csv=dataset_csv,
        dataset_metadata=dataset_metadata,
        layer=layer,
        neuron=neuron,
        layer_type=args.layer_type,
        token_pos=args.token_pos,
        feature_layer=args.feature_layer,
        use_regression=args.regression,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        test_split=args.test_split,
        early_stopping=args.early_stopping,
        output_dir=args.output_dir,
        device=device,
    )
    
    # Evaluate model
    results = evaluate_model(
        predictor=predictor,
        trainer=trainer,
        texts=texts,
        output_dir=args.output_dir,
    )
    
    # Save final results summary
    summary_path = os.path.join(args.output_dir, f"introspection_summary_{timestamp}.json")
    
    # Create a summary dictionary with key results
    summary = {
        "experiment_time": timestamp,
        "model": args.model,
        "target": {
            "layer": layer,
            "neuron": neuron,
            "layer_type": args.layer_type,
            "token_position": args.token_pos,
        },
        "dataset": {
            "size": args.num_samples,
            "csv_path": dataset_csv,
            "metadata_path": dataset_metadata,
        },
        "training": {
            "head_type": "regression" if args.regression else "classification",
            "feature_layer": args.feature_layer,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "early_stopping": args.early_stopping,
        },
        "results": {
            "correlation": results["correlation"],
            "mse": results["mse"],
            "report_path": results["report_path"],
            "figure_path": results["figure_path"],
        }
    }
    
    # Save summary as JSON
    with open(summary_path, "w") as f:
        import json
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {summary_path}")
    logger.info("Neural introspection pipeline completed successfully!")
    
    # Print a concise summary to the console
    print("\n" + "="*60)
    print(f"NEURAL INTROSPECTION PIPELINE COMPLETED")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Target: Layer {layer}, Neuron {neuron}")
    print(f"Results:")
    print(f"  Correlation: {results['correlation']:.4f}")
    print(f"  MSE: {results['mse']:.4f}")
    print(f"Output directory: {args.output_dir}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()