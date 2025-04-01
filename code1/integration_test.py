# integration_test.py

import torch
import numpy as np
from transformer_lens import HookedTransformer
from neuron_selection.scanner import NeuronScanner
from dataset.generator import ActivationDatasetGenerator
from architecture.architecture import ActivationPredictor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import argparse
import time

def run_integration_test(
    model_name: str = "gpt2-small",
    device: str = "mps",
    top_k: int = 5,
    sample_texts_path: str = None,
    n_sample_texts: int = 100,
    output_dir: str = "integration_test_output",
    dataset_name: str = "test_dataset",
    num_bins: int = 5,
    evaluation_only: bool = False,
):
    """
    Run an integration test of the three main modules.
    
    Args:
        model_name: Name of the transformer model to use
        device: Device to run on
        top_k: Number of top neurons to consider
        sample_texts_path: Path to sample texts file (one text per line)
        n_sample_texts: Number of sample texts to use if generating
        output_dir: Directory to save outputs
        dataset_name: Name for the generated dataset
        num_bins: Number of bins for discretization
        evaluation_only: If True, only run evaluation (no training)
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load model
    print(f"Loading model {model_name}...")
    model = HookedTransformer.from_pretrained(model_name)
    model.to(device)
    tokenizer = model.tokenizer
    
    # Step 2: Load or generate sample texts
    if sample_texts_path and os.path.exists(sample_texts_path):
        print(f"Loading sample texts from {sample_texts_path}...")
        with open(sample_texts_path, "r") as f:
            sample_texts = [line.strip() for line in f if line.strip()]
            sample_texts = sample_texts[:n_sample_texts]  # Limit to n_sample_texts
    else:
        print(f"Creating {n_sample_texts} sample texts...")
        # Use a predefined set of diverse example texts
        example_texts = [
            "The neural network processed the input data efficiently.",
            "Researchers discovered a new activation pattern in deep layers.",
            "Information flows through transformer models via attention mechanisms.",
            "The gradient descent algorithm optimized the weights rapidly.",
            "Attention heads specialize in different linguistic features.",
            "Transformer models excel at capturing long-range dependencies.",
            "Tokenization affects how models process text input.",
            "The embedding layer maps tokens to high-dimensional vectors.",
            "Residual connections help information flow through deep networks.",
            "Layer normalization stabilizes the training of deep networks.",
            "The model achieved state-of-the-art performance on multiple benchmarks.",
            "Fine-tuning adapts pretrained models to specific downstream tasks.",
            "The feed-forward network transforms token representations.",
            "Self-attention allows tokens to attend to other positions in the sequence.",
            "Position embeddings provide spatial information to the model.",
        ]
        # Repeat to reach desired count
        sample_texts = []
        while len(sample_texts) < n_sample_texts:
            sample_texts.extend(example_texts[:min(len(example_texts), n_sample_texts-len(sample_texts))])
        
        print(f"Sample text count: {len(sample_texts)}")
        print(f"Sample text example: {sample_texts[0]}")
    
    # Step 3: Scan for interesting neurons
    print("\nScanning for interesting neurons...")
    scanner = NeuronScanner(model, tokenizer, device=device)
    scan_results = scanner.scan_neurons(
        texts=sample_texts, 
        token_pos="last",
        layer_type="mlp_out",
        top_k=top_k,
    )
    
    # Print top neurons
    print("\nTop neurons by activation diversity:")
    for i, ((layer, neuron), stats) in enumerate(scan_results["top_neurons"]):
        print(f"{i+1}. Layer {layer}, Neuron {neuron}: "
              f"Score={stats['score']:.4f}, "
              f"Variance={stats['variance']:.4f}, "
              f"Range={stats['range']:.4f}")
    
    # Step 4: Visualize top neuron
    top_layer, top_neuron = scan_results["top_neurons"][0][0]
    print(f"\nVisualizing top neuron (Layer {top_layer}, Neuron {top_neuron})...")
    fig = scanner.visualize_neuron(
        texts=sample_texts,
        layer=top_layer,
        neuron_idx=top_neuron,
        layer_type="mlp_out"
    )
    
    fig.savefig(os.path.join(output_dir, "top_neuron_visualization.png"))
    plt.close(fig)
    
    # Step 5: Generate dataset for top neuron
    print("\nGenerating dataset for top neuron...")
    generator = ActivationDatasetGenerator(model, tokenizer, device=device)
    
    # Generate a classification dataset
    classification_dataset, class_metadata = generator.generate_dataset(
        texts=sample_texts,
        layer=top_layer,
        neuron_idx=top_neuron,
        layer_type="mlp_out",
        token_pos="last",
        output_tokens=True,
        num_bins=num_bins,
        balance_bins=True,
        target_samples_per_bin=20,
    )
    
    # Save dataset
    generator.save_dataset(
        classification_dataset, 
        class_metadata, 
        output_dir, 
        f"{dataset_name}_classification"
    )
    
    # Generate a regression dataset
    regression_dataset, reg_metadata = generator.generate_dataset(
        texts=sample_texts,
        layer=top_layer,
        neuron_idx=top_neuron,
        layer_type="mlp_out",
        token_pos="last",
        output_tokens=False,  # Regression
    )
    
    # Save dataset
    generator.save_dataset(
        regression_dataset, 
        reg_metadata, 
        output_dir, 
        f"{dataset_name}_regression"
    )
    
    # Step 6: Create and evaluate predictors
    print("\nCreating activation predictors...")
    
    # For classification
    print("\nTesting classification predictor...")
    # Get bin edges from the dataset
    bin_edges = np.array(classification_dataset.bin_info["bin_edges"])
    
    class_predictor = ActivationPredictor(
        base_model=model,
        head_type="classification",
        num_classes=num_bins,
        target_layer=top_layer,
        target_neuron=top_neuron,
        layer_type="mlp_out",
        token_pos="last",
        head_config={"hidden_dim": 32},
        device=device,
        bin_edges=bin_edges,
    )
    
    # Split data for evaluation
    all_texts = classification_dataset.inputs
    all_bins = classification_dataset.discretized
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_bins, test_size=0.2, random_state=42
    )
    
    # Evaluate on test set
    print(f"\nEvaluating classification predictor on {len(X_test)} test samples...")
    
    # Get raw predictions (logits)
    predictions, actual_activations = class_predictor.predict(
        texts=X_test,
        batch_size=8,
        return_activations=True,
        return_raw=True,
    )
    
    # Convert to class predictions
    pred_classes = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(pred_classes == y_test)
    print(f"Classification accuracy (before training): {accuracy:.4f}")
    
    # Report on test set
    class_report = class_predictor.report(X_test[:10], confidence_threshold=0.6)
    print(f"Mean error: {class_report['mean_error']:.4f}")
    print(f"Mean confidence: {class_report['mean_confidence']:.4f}")
    
    # For regression
    print("\nTesting regression predictor...")
    reg_predictor = ActivationPredictor(
        base_model=model,
        head_type="regression",
        target_layer=top_layer,
        target_neuron=top_neuron,
        layer_type="mlp_out",
        token_pos="last",
        head_config={"hidden_dim": 32},
        device=device,
        activation_mean=np.mean(regression_dataset.activations),
        activation_std=np.std(regression_dataset.activations),
    )
    
    # Split regression data
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        regression_dataset.inputs, 
        regression_dataset.activations, 
        test_size=0.2, 
        random_state=42
    )
    
    # Evaluate regression model
    print(f"\nEvaluating regression predictor on {len(X_reg_test)} test samples...")
    reg_predictions, reg_actual = reg_predictor.predict(
        texts=X_reg_test,
        batch_size=8,
        return_activations=True,
    )
    
    # Calculate MSE
    mse = np.mean((reg_predictions - reg_actual) ** 2)
    print(f"Regression MSE (before training): {mse:.4f}")
    
    # Report on test set
    reg_report = reg_predictor.report(X_reg_test[:10])
    print(f"Regression mean error: {reg_report['mean_error']:.4f}")
    print(f"Regression mean confidence: {reg_report['mean_confidence']:.4f}")
    
    # Step 7: Save predictors for future use
    class_predictor.save(os.path.join(output_dir, "classification_predictor"))
    reg_predictor.save(os.path.join(output_dir, "regression_predictor"))
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\nIntegration test completed successfully!")
    print(f"Total runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"Outputs saved to {output_dir}")
    
    return {
        "top_neurons": scan_results["top_neurons"],
        "classification_dataset": classification_dataset,
        "regression_dataset": regression_dataset,
        "class_predictor": class_predictor,
        "reg_predictor": reg_predictor,
        "class_accuracy": accuracy,
        "reg_mse": mse,
        "runtime": elapsed,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run integration test for neuron introspection modules")
    parser.add_argument("--model", type=str, default="gpt2-small", help="Model name to use")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (defaults to mps if available, else cpu)")
    parser.add_argument("--samples", type=int, default=100, help="Number of sample texts to use")
    parser.add_argument("--texts", type=str, default=None, help="Path to sample texts file")
    parser.add_argument("--output", type=str, default="integration_test_output", help="Output directory")
    parser.add_argument("--bins", type=int, default=5, help="Number of bins for discretization")
    
    args = parser.parse_args()
    
    # Set device automatically if not specified
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Run integration test
    results = run_integration_test(
        model_name=args.model,
        device=device,
        n_sample_texts=args.samples,
        sample_texts_path=args.texts,
        output_dir=args.output,
        num_bins=args.bins,
    )