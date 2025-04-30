# Mechanistic Interpretability Experiments - Jupyter Notebook Plan

This plan outlines two Jupyter notebooks for analyzing trained activation prediction models through mechanistic interpretability techniques.

## Overview

We'll create two Jupyter notebooks:

1. `linear_weights_analysis.ipynb`: Compare the linear weights used for prediction with the MLP layer weights
2. `activation_patching.ipynb`: Patch neuron activations to assess their impact on predictions

## Notebook 1: Linear Weights Analysis

### Setup and Loading (3-4 cells)
```python
# Imports
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from transformer_lens import HookedTransformer
```

```python
# Function to load trained model head
def load_head(model_path):
    """Load the saved prediction head and its configuration"""
    config_path = os.path.join(model_path, "config.json")
    head_path = os.path.join(model_path, "head.pt")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    head_weights = torch.load(head_path, map_location="cpu")
    
    return head_weights, config
```

```python
# Path to a trained model - update this path to a valid model directory
model_path = "../output/models/neuron_l8_n481_20250422_173314"

# Load the trained model
head_weights, config = load_head(model_path)

# Display the config to understand what we're analyzing
print("Model configuration:")
for key, value in config.items():
    if key != "head_config":  # Skip printing the full head config for brevity
        print(f"  {key}: {value}")

# Print head type details
print(f"\nHead type: {config['head_type']}")
if "head_config" in config and "hidden_dim" in config["head_config"]:
    print(f"Hidden dimension: {config['head_config']['hidden_dim']}")
```

```python
# Load the base transformer model
base_model_name = config["base_model_name"]
base_model = HookedTransformer.from_pretrained(base_model_name)

# Extract target neuron information
target_layer = config["target_layer"]
target_neuron = config["target_neuron"]
layer_type = config["layer_type"]
print(f"Analyzing neuron {target_neuron} in layer {target_layer}")
print(f"Base model: {base_model_name}")
print(f"Hidden dimension: {base_model.cfg.d_model}")
print(f"MLP dimension: {base_model.cfg.d_mlp}")
```

### Extract and Examine Weights (5-6 cells)
```python
# Examine the head weights
print("Head weights keys:")
for key in head_weights.keys():
    print(f"  {key}: {head_weights[key].shape}")
```

```python
# Extract linear weights from the head based on head type
def extract_linear_weights(head_weights, config, base_model=None):
    """Extract the linear prediction tensor from the head"""
    head_type = config["head_type"]
    
    if head_type == "regression":
        # For regression head, extract the final linear layer weights
        if "hidden.weight" in head_weights:
            # Has hidden layer - need to compose the transformation
            hidden_w = head_weights["hidden.weight"]
            hidden_b = head_weights["hidden.bias"]
            output_w = head_weights["output.weight"]
            output_b = head_weights["output.bias"]
            
            # Note: This is an approximation, as we're ignoring the non-linearity
            # We'll use only the linear component for direct comparison
            linear_weights = output_w.squeeze()
            print("Using output layer weights from regression head with hidden layer")
        else:
            # Direct linear projection
            linear_weights = head_weights["output.weight"].squeeze()
            print("Using output layer weights from regression head without hidden layer")
    
    elif head_type == "classification":
        # For classification head, combine weights based on bin edges
        output_weights = head_weights["output.weight"]
        bin_edges = config.get("bin_edges")
        
        if bin_edges:
            # Calculate bin centers
            bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
            
            # Weight each class weight by its bin center to get a linear approximation
            combined_weights = torch.zeros_like(output_weights[0])
            for i, center in enumerate(bin_centers):
                if i < output_weights.shape[0]:
                    combined_weights += output_weights[i] * center
            
            linear_weights = combined_weights
            print(f"Using weighted combination of classification weights with {len(bin_centers)} bin centers")
        else:
            # If no bin edges, use the first class weight as an approximation
            linear_weights = output_weights[0]
            print("Warning: Using first class weight as approximation (no bin edges found)")
    
    elif head_type == "token":
        # For token head, weights come from the base model's embedding
        if base_model is None:
            raise ValueError("Base model is required for token head analysis")
            
        # Get digit tokens (0-9) from the head config or use defaults
        digit_tokens = config.get("head_config", {}).get("digit_tokens", list(range(48, 58)))  # ASCII 0-9
        
        if not digit_tokens:
            # Fallback to ASCII digits if no tokens found
            digit_tokens = list(range(48, 58))
            
        # Get embedding vectors for digit tokens
        digit_embeddings = base_model.embed.weight[digit_tokens]
        
        # Weight each embedding by its value (0-9)
        combined_weights = torch.zeros_like(digit_embeddings[0])
        for i, embedding in enumerate(digit_embeddings):
            combined_weights += embedding * (i / 9.0)  # Normalize to 0-1 range
        
        linear_weights = combined_weights
        print(f"Using weighted combination of {len(digit_tokens)} token embeddings")
    
    else:
        raise ValueError(f"Unknown head type: {head_type}")
    
    return linear_weights
```

```python
# Extract the linear prediction weights
linear_weights = extract_linear_weights(head_weights, config, base_model)
print(f"Linear weights shape: {linear_weights.shape}")
```

```python
# Extract MLP input weights for the target neuron
def get_mlp_in_weights(model, layer, neuron=None):
    """Get weights from residual stream into MLP layer"""
    # Try different model architectures
    try:
        # TransformerLens standard format
        if hasattr(model.blocks[layer].mlp, "W_in"):
            mlp_in = model.blocks[layer].mlp.W_in.weight
        # GPT-2 style
        elif hasattr(model.blocks[layer].mlp, "c_fc"):
            mlp_in = model.blocks[layer].mlp.c_fc.weight
        # Fallback for other architectures
        else:
            # Try to find any weight matrix in the MLP module
            for name, module in model.blocks[layer].mlp.named_modules():
                if isinstance(module, torch.nn.Linear) and module.weight.shape[0] > model.cfg.d_model:
                    mlp_in = module.weight
                    break
            else:
                raise ValueError("Couldn't find MLP input weights")
                
        if neuron is not None:
            return mlp_in[neuron]  # Row corresponding to this neuron
        return mlp_in
    except Exception as e:
        print(f"Error accessing MLP input weights: {e}")
        print("Model structure:")
        for name, param in model.named_parameters():
            if f"blocks.{layer}" in name and "mlp" in name and "weight" in name:
                print(f"  {name}: {param.shape}")
        raise
```

```python
# Extract MLP output weights for the target neuron
def get_mlp_out_weights(model, layer, neuron=None):
    """Get weights from MLP neuron to output"""
    # Try different model architectures
    try:
        # TransformerLens standard format
        if hasattr(model.blocks[layer].mlp, "W_out"):
            mlp_out = model.blocks[layer].mlp.W_out.weight
        # GPT-2 style
        elif hasattr(model.blocks[layer].mlp, "c_proj"):
            mlp_out = model.blocks[layer].mlp.c_proj.weight
        # Fallback for other architectures
        else:
            # Try to find any weight matrix in the MLP module with appropriate shape
            for name, module in model.blocks[layer].mlp.named_modules():
                if isinstance(module, torch.nn.Linear) and module.weight.shape[1] > model.cfg.d_model:
                    mlp_out = module.weight
                    break
            else:
                raise ValueError("Couldn't find MLP output weights")
                
        if neuron is not None:
            # For output weights, the neuron corresponds to a column
            return mlp_out[:, neuron]  
        return mlp_out
    except Exception as e:
        print(f"Error accessing MLP output weights: {e}")
        print("Model structure:")
        for name, param in model.named_parameters():
            if f"blocks.{layer}" in name and "mlp" in name and "weight" in name:
                print(f"  {name}: {param.shape}")
        raise
```

```python
# Inspect the block structure to understand how to access weights
try:
    print(f"MLP module structure for layer {target_layer}:")
    for name, module in base_model.blocks[target_layer].mlp.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"  {name}: {module}")
            print(f"    input_dim: {module.in_features}, output_dim: {module.out_features}")
except Exception as e:
    print(f"Error inspecting MLP structure: {e}")
    print("Trying to list all parameters:")
    for name, param in base_model.named_parameters():
        if f"blocks.{target_layer}" in name and "mlp" in name:
            print(f"  {name}: {param.shape}")

# Extract MLP weights for the target neuron
try:
    mlp_in_weights = get_mlp_in_weights(base_model, target_layer, target_neuron)
    print(f"MLP input weights shape: {mlp_in_weights.shape}")
    
    mlp_out_weights = get_mlp_out_weights(base_model, target_layer, target_neuron)
    print(f"MLP output weights shape: {mlp_out_weights.shape}")
except Exception as e:
    print(f"Error extracting MLP weights: {e}")
    # Try getting all weights to see what's available
    print("Available parameters:")
    for name, param in base_model.named_parameters():
        if f"blocks.{target_layer}" in name:
            print(f"  {name}: {param.shape}")
```

### Compute Similarity and Visualize (3-4 cells)
```python
# Compute cosine similarity
def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors"""
    a_flat = a.flatten().to(torch.float32)
    b_flat = b.flatten().to(torch.float32)
    
    # Normalize
    a_norm = torch.nn.functional.normalize(a_flat.unsqueeze(0)).squeeze(0)
    b_norm = torch.nn.functional.normalize(b_flat.unsqueeze(0)).squeeze(0)
    
    # Compute similarity
    return torch.dot(a_norm, b_norm).item()
```

```python
# Compare linear weights with MLP input weights
sim_with_input = cosine_similarity(linear_weights, mlp_in_weights)
print(f"Cosine similarity with MLP input weights: {sim_with_input:.4f}")

# Compare linear weights with MLP output weights
sim_with_output = cosine_similarity(linear_weights, mlp_out_weights)
print(f"Cosine similarity with MLP output weights: {sim_with_output:.4f}")
```

```python
# Create a bar chart comparing similarities
plt.figure(figsize=(10, 6))
bars = plt.bar(['Similarity with MLP Input', 'Similarity with MLP Output'], 
        [sim_with_input, sim_with_output],
        color=['blue', 'green'])

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom')

plt.ylim(-1, 1)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title(f'Cosine Similarity between Linear Weights and MLP Weights\nNeuron {target_neuron} in Layer {target_layer}')
plt.ylabel('Cosine Similarity')
plt.grid(axis='y', alpha=0.3)
plt.savefig('weight_similarity_comparison.png')
plt.show()
```

### Statistical Analysis and Visualization (4-5 cells)
```python
# Analyze weight distributions
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(linear_weights.detach().cpu().numpy(), bins=30, alpha=0.7, color='blue')
plt.title('Linear Prediction Weights Distribution')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(mlp_in_weights.detach().cpu().numpy(), bins=30, alpha=0.7, color='green')
plt.title('MLP Input Weights Distribution')
plt.xlabel('Weight Value')

plt.subplot(1, 3, 3)
plt.hist(mlp_out_weights.detach().cpu().numpy(), bins=30, alpha=0.7, color='orange')
plt.title('MLP Output Weights Distribution')
plt.xlabel('Weight Value')

plt.tight_layout()
plt.savefig('weight_distributions.png')
plt.show()
```

```python
# Compute statistics for each weight vector
def compute_weight_stats(weights, name):
    """Compute basic statistics for weight tensor"""
    w = weights.detach().cpu().numpy()
    return {
        "name": name,
        "mean": float(np.mean(w)),
        "std": float(np.std(w)),
        "min": float(np.min(w)),
        "max": float(np.max(w)),
        "l2_norm": float(np.linalg.norm(w)),
        "sparsity": float(np.mean(np.abs(w) < 1e-5))
    }

# Compute and display statistics
stats = [
    compute_weight_stats(linear_weights, "Linear Prediction Weights"),
    compute_weight_stats(mlp_in_weights, "MLP Input Weights"),
    compute_weight_stats(mlp_out_weights, "MLP Output Weights")
]

print("Weight Statistics:")
for s in stats:
    print(f"\n{s['name']}:")
    for k, v in s.items():
        if k != 'name':
            print(f"  {k}: {v:.6f}")
```

```python
# If the weights are high-dimensional, try visualizing with PCA
from sklearn.decomposition import PCA

# Prepare weight vectors
weight_vectors = [
    ("Linear Weights", linear_weights.detach().cpu().numpy()),
    ("MLP Input Weights", mlp_in_weights.detach().cpu().numpy()),
    ("MLP Output Weights", mlp_out_weights.detach().cpu().numpy())
]

# Stack all vectors for PCA
all_weights = np.vstack([w[1].reshape(1, -1) for w in weight_vectors])

# Perform PCA
pca = PCA(n_components=2)
weights_pca = pca.fit_transform(all_weights)

# Plot in 2D space
plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'orange']
for i, (name, _) in enumerate(weight_vectors):
    plt.scatter(weights_pca[i, 0], weights_pca[i, 1], label=name, s=150, color=colors[i])

plt.title('PCA Projection of Weight Vectors')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('weight_vectors_pca.png')
plt.show()
```

```python
# Generate random vectors for comparison
def random_similarity_baseline(vector, num_samples=1000):
    """Compare a vector with random vectors to establish a baseline"""
    vector = vector.flatten().detach().cpu().numpy()
    similarities = []
    
    for _ in range(num_samples):
        # Generate random vector with same dimension
        random_vec = np.random.randn(vector.shape[0])
        # Normalize both vectors
        vec_norm = vector / np.linalg.norm(vector)
        rand_norm = random_vec / np.linalg.norm(random_vec)
        # Compute cosine similarity
        sim = np.dot(vec_norm, rand_norm)
        similarities.append(sim)
    
    return {
        "mean": np.mean(similarities),
        "std": np.std(similarities),
        "p5": np.percentile(similarities, 5),
        "p95": np.percentile(similarities, 95),
        "min": np.min(similarities),
        "max": np.max(similarities)
    }

# Calculate random baselines
random_baseline = random_similarity_baseline(linear_weights)

print("Random similarity baseline:")
for k, v in random_baseline.items():
    print(f"  {k}: {v:.6f}")

# Calculate how many standard deviations our similarities are from random
z_score_input = (sim_with_input - random_baseline["mean"]) / random_baseline["std"]
z_score_output = (sim_with_output - random_baseline["mean"]) / random_baseline["std"]

print(f"\nStatistical significance:")
print(f"  Input similarity z-score: {z_score_input:.2f}")
print(f"  Output similarity z-score: {z_score_output:.2f}")
```

### Summary and Interpretation (1-2 cells)
```python
# Create a summary table
summary = {
    "model_path": model_path,
    "head_type": config["head_type"],
    "target_layer": target_layer,
    "target_neuron": target_neuron,
    "sim_with_input": sim_with_input,
    "sim_with_output": sim_with_output,
    "random_baseline_mean": random_baseline["mean"],
    "z_score_input": z_score_input,
    "z_score_output": z_score_output,
    "timestamp": str(np.datetime64('now'))
}

print("Analysis Summary:")
for k, v in summary.items():
    print(f"  {k}: {v}")

# Save results
os.makedirs("../mech_interp/results", exist_ok=True)
results_path = f"../mech_interp/results/linear_weights_analysis_{target_layer}_{target_neuron}.json"
with open(results_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {results_path}")
```

## Notebook 2: Activation Patching

### Setup and Loading (3-4 cells)
```python
# Similar imports and loading code as the first notebook
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from transformer_lens import HookedTransformer

# Load the trained model and configuration
# (Same code as the first notebook)

# Sample test inputs
test_inputs = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can be difficult to interpret.",
    "Transformers use attention mechanisms to process sequences.",
    "Neural networks have revolutionized artificial intelligence.",
    "Researchers work to make AI systems more transparent."
]

# Tokenize inputs
tokenized_inputs = [
    base_model.tokenizer(text, return_tensors="pt")
    for text in test_inputs
]
```

### Helper Functions for Patching and Prediction (4-5 cells)
```python
# Function to make predictions with the trained head
def predict_with_head(input_ids, attention_mask, head_weights, config, base_model):
    """Run the base model and predict with the trained head"""
    # Run the base model to get activations
    with torch.no_grad():
        # Get output features at the appropriate layer
        feature_layer = config.get("feature_layer", -1)
        outputs = base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Get the feature representation (residual stream)
        if feature_layer < 0:
            feature_layer = base_model.cfg.n_layers + feature_layer
        
        # Extract features for the specified position
        token_pos = config.get("token_pos", "last")
        if token_pos == "last":
            pos = int((attention_mask.sum(-1) - 1)[0]) if attention_mask is not None else -1
        else:
            pos = int(token_pos)
        
        features = outputs.hidden_states[feature_layer][0, pos]
        
        # Apply the head
        head_type = config["head_type"]
        
        if head_type == "regression":
            # Apply dropout, hidden layer (if present), and output layer
            if "hidden.weight" in head_weights:
                x = torch.nn.functional.dropout(features, p=config["head_config"].get("dropout", 0.1))
                x = torch.nn.functional.linear(x, head_weights["hidden.weight"], head_weights["hidden.bias"])
                x = torch.nn.functional.gelu(x)
                x = torch.nn.functional.dropout(x, p=config["head_config"].get("dropout", 0.1))
                pred = torch.nn.functional.linear(x, head_weights["output.weight"], head_weights["output.bias"])
            else:
                x = torch.nn.functional.dropout(features, p=config["head_config"].get("dropout", 0.1))
                pred = torch.nn.functional.linear(x, head_weights["output.weight"], head_weights["output.bias"])
            
            pred = pred.squeeze()
            
        elif head_type == "classification":
            # Apply classification head
            if "hidden.weight" in head_weights:
                x = torch.nn.functional.dropout(features, p=config["head_config"].get("dropout", 0.1))
                x = torch.nn.functional.linear(x, head_weights["hidden.weight"], head_weights["hidden.bias"])
                x = torch.nn.functional.gelu(x)
                x = torch.nn.functional.dropout(x, p=config["head_config"].get("dropout", 0.1))
                logits = torch.nn.functional.linear(x, head_weights["output.weight"], head_weights["output.bias"])
            else:
                x = torch.nn.functional.dropout(features, p=config["head_config"].get("dropout", 0.1))
                logits = torch.nn.functional.linear(x, head_weights["output.weight"], head_weights["output.bias"])
            
            # Convert to continuous prediction if bin edges are available
            bin_edges = config.get("bin_edges")
            if bin_edges:
                bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Weight by bin centers
                pred = 0
                for i, center in enumerate(bin_centers):
                    if i < probs.shape[-1]:
                        pred += probs[i] * center
            else:
                pred = torch.argmax(logits).item()
                
        elif head_type == "token":
            # Token prediction is handled differently
            # This implementation needs to be customized based on your token head design
            # Basic skeleton shown here
            logits = outputs.logits[0, pos]
            digit_tokens = config.get("head_config", {}).get("digit_tokens", list(range(48, 58)))
            
            # Extract logits for digit tokens
            digit_logits = torch.stack([logits[idx] for idx in digit_tokens])
            probs = torch.nn.functional.softmax(digit_logits, dim=0)
            
            # Weight by digit values (0-9)
            pred = 0
            for i, p in enumerate(probs):
                pred += i * p.item()
            
        return pred
```

```python
# Function to extract the neuron activation
def get_neuron_activation(input_ids, attention_mask, base_model, layer, neuron, layer_type="mlp_out"):
    """Extract the activation of a specific neuron"""
    with torch.no_grad():
        _, cache = base_model.run_with_cache(
            input_ids, 
            attention_mask=attention_mask
        )
        
        # Get last token position
        pos = int((attention_mask.sum(-1) - 1)[0]) if attention_mask is not None else -1
        
        # Extract activation
        activation = cache[layer_type, layer][0, pos, neuron].item()
        
        return activation
```

```python
# Function to create a patching hook
def create_patching_hook(neuron_idx, new_value, patch_type="set"):
    """Create a hook function for patching a neuron activation"""
    def hook_fn(activation, hook):
        # Get shape info
        batch_size = activation.shape[0]
        seq_len = activation.shape[1]
        
        # Create a copy to avoid modifying the original
        patched = activation.clone()
        
        # Apply patching to the target neuron for all positions or just the last token
        # We'll patch only the last token to match how we're extracting features
        if hook.n_pos is not None:  # If we have position information
            for i in range(batch_size):
                pos = hook.n_pos[i]
                if patch_type == "set":
                    # Set to constant value
                    patched[i, pos, neuron_idx] = new_value
                elif patch_type == "scale":
                    # Scale by factor
                    patched[i, pos, neuron_idx] = activation[i, pos, neuron_idx] * new_value
                elif patch_type == "zero":
                    # Set to zero
                    patched[i, pos, neuron_idx] = 0.0
        else:  # If we don't have position info, assume last token
            # This is simpler but maybe less precise
            if patch_type == "set":
                patched[:, -1, neuron_idx] = new_value
            elif patch_type == "scale":
                patched[:, -1, neuron_idx] = activation[:, -1, neuron_idx] * new_value
            elif patch_type == "zero":
                patched[:, -1, neuron_idx] = 0.0
                
        return patched
    
    return hook_fn
```

### Baseline Measurement (2-3 cells)
```python
# Measure baseline neuron activations and predictions
baseline_results = []

for i, inputs in enumerate(tokenized_inputs):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"] if "attention_mask" in inputs else None
    
    # Get neuron activation
    activation = get_neuron_activation(
        input_ids, 
        attention_mask, 
        base_model, 
        target_layer, 
        target_neuron, 
        layer_type
    )
    
    # Get prediction
    prediction = predict_with_head(
        input_ids, 
        attention_mask, 
        head_weights, 
        config, 
        base_model
    )
    
    baseline_results.append({
        "input": test_inputs[i],
        "activation": activation,
        "prediction": prediction
    })
    
    print(f"Input {i+1}:")
    print(f"  Text: {test_inputs[i][:50]}...")
    print(f"  Activation: {activation:.4f}")
    print(f"  Prediction: {prediction:.4f}")

# Calculate baseline statistics
baseline_activations = [r["activation"] for r in baseline_results]
baseline_predictions = [r["prediction"] for r in baseline_results]

print(f"\nBaseline Statistics:")
print(f"  Mean Activation: {np.mean(baseline_activations):.4f}")
print(f"  Mean Prediction: {np.mean(baseline_predictions):.4f}")
print(f"  Correlation: {np.corrcoef(baseline_activations, baseline_predictions)[0,1]:.4f}")
```

### Patching Experiments (4-5 cells)
```python
# Run patching experiment with a constant value
def run_patching_experiment(patch_value, patch_type="set"):
    """Run experiment with patched neuron activation"""
    patched_results = []
    
    # Create hook function
    hook_fn = create_patching_hook(target_neuron, patch_value, patch_type)
    
    for i, inputs in enumerate(tokenized_inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"] if "attention_mask" in inputs else None
        
        # Get last token position for hook context
        last_pos = int((attention_mask.sum(-1) - 1)[0]) if attention_mask is not None else -1
        
        # Create a hook context object to pass to the hook
        class HookContext:
            def __init__(self, positions):
                self.n_pos = positions
                
        hook_context = HookContext([last_pos])  # Pass position info to hook
        
        # Run with patching hook
        hook_name = f"{layer_type}.{target_layer}"
        with base_model.hooks([(hook_name, hook_fn)]):
            # Get patched activation
            patched_activation = get_neuron_activation(
                input_ids, 
                attention_mask, 
                base_model, 
                target_layer, 
                target_neuron, 
                layer_type
            )
            
            # Get prediction with patched activation
            patched_prediction = predict_with_head(
                input_ids, 
                attention_mask, 
                head_weights, 
                config, 
                base_model
            )
        
        # Calculate changes from baseline
        baseline = baseline_results[i]
        act_change = patched_activation - baseline["activation"]
        pred_change = patched_prediction - baseline["prediction"]
        
        patched_results.append({
            "input": test_inputs[i],
            "patched_activation": patched_activation,
            "patched_prediction": patched_prediction,
            "activation_change": act_change,
            "prediction_change": pred_change
        })
    
    return patched_results
```

```python
# Run constant-value patching experiment
constant_value = 0.0  # Try with zero
constant_results = run_patching_experiment(constant_value, "set")

# Print results
print(f"Patching results (value={constant_value}):")
for i, result in enumerate(constant_results):
    print(f"Input {i+1}:")
    print(f"  Baseline Activation: {baseline_results[i]['activation']:.4f}")
    print(f"  Patched Activation: {result['patched_activation']:.4f}")
    print(f"  Activation Change: {result['activation_change']:.4f}")
    print(f"  Prediction Change: {result['prediction_change']:.4f}")
```

```python
# Run scaling experiment with multiple factors
scale_factors = [0.0, 0.5, 1.0, 1.5, 2.0]
scaling_results = []

for factor in scale_factors:
    results = run_patching_experiment(factor, "scale")
    
    # Aggregate results
    avg_act_change = np.mean([r["activation_change"] for r in results])
    avg_pred_change = np.mean([r["prediction_change"] for r in results])
    
    scaling_results.append({
        "factor": factor,
        "results": results,
        "avg_act_change": avg_act_change,
        "avg_pred_change": avg_pred_change
    })
    
    print(f"Scale factor {factor}:")
    print(f"  Avg Activation Change: {avg_act_change:.4f}")
    print(f"  Avg Prediction Change: {avg_pred_change:.4f}")
```

### Visualize Results (5-6 cells)
```python
# Visualize scaling experiment results
plt.figure(figsize=(12, 6))

# Plot activation changes
act_changes = [r["avg_act_change"] for r in scaling_results]
pred_changes = [r["avg_pred_change"] for r in scaling_results]
factors = [r["factor"] for r in scaling_results]

plt.subplot(1, 2, 1)
plt.plot(factors, act_changes, 'b-o', label='Activation Change')
plt.plot(factors, pred_changes, 'r-o', label='Prediction Change')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=1.0, color='k', linestyle='--', alpha=0.3)

plt.title(f'Effect of Scaling Neuron {target_neuron}')
plt.xlabel('Scaling Factor')
plt.ylabel('Change from Baseline')
plt.legend()
plt.grid(alpha=0.3)

# Plot changes in predicted vs. actual space
plt.subplot(1, 2, 2)
# Get baseline means
baseline_act_mean = np.mean(baseline_activations)
baseline_pred_mean = np.mean(baseline_predictions)

# Calculate projected values after scaling
scaled_acts = [baseline_act_mean + change for change in act_changes]
scaled_preds = [baseline_pred_mean + change for change in pred_changes]

# Plot with connecting lines in activation-prediction space
plt.plot(scaled_acts, scaled_preds, 'g-o')
plt.plot([baseline_act_mean], [baseline_pred_mean], 'ko', markersize=10, label='Baseline (factor=1.0)')

# Add labels for scaling factors
for i, factor in enumerate(factors):
    plt.annotate(f"{factor:.2f}", 
                 (scaled_acts[i], scaled_preds[i]),
                 xytext=(5, 5), textcoords='offset points')

plt.title('Activation-Prediction Space')
plt.xlabel('Average Activation')
plt.ylabel('Average Prediction')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('scaling_experiment.png')
plt.show()
```

```python
# Check if prediction changes are proportional to activation changes
pred_vs_act = []
for result in scaling_results:
    for sample in result["results"]:
        pred_vs_act.append({
            "input": sample["input"],
            "act_change": sample["activation_change"],
            "pred_change": sample["prediction_change"],
            "factor": result["factor"]
        })

# Plot prediction change vs activation change
plt.figure(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(scale_factors)))

# Create a plot with all individual points
for i, factor in enumerate(scale_factors):
    # Get points for this scaling factor
    points = [p for p in pred_vs_act if p["factor"] == factor]
    plt.scatter(
        [p["act_change"] for p in points],
        [p["pred_change"] for p in points],
        label=f'Scale={factor}',
        color=colors[i],
        alpha=0.7,
        s=50
    )

plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.title('Prediction Change vs Activation Change\nAll Inputs and Scaling Factors')
plt.xlabel('Activation Change')
plt.ylabel('Prediction Change')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pred_vs_act_change_all.png')
plt.show()
```

```python
# Fit a linear model to quantify the relationship
from sklearn.linear_model import LinearRegression

# Prepare data
X = np.array([p["act_change"] for p in pred_vs_act]).reshape(-1, 1)
y = np.array([p["pred_change"] for p in pred_vs_act])

# Fit model
model = LinearRegression()
model.fit(X, y)

# Calculate R² score
r2 = model.score(X, y)
slope = model.coef_[0]
intercept = model.intercept_

# Plot the data with regression line
plt.figure(figsize=(10, 7))
plt.scatter(X, y, alpha=0.6)
plt.plot(
    [X.min(), X.max()], 
    [model.predict([[X.min()]])[0], model.predict([[X.max()]])[0]], 
    'r-', linewidth=2
)

plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.title(f'Linear Relationship between Activation and Prediction Changes\nSlope: {slope:.4f}, R²: {r2:.4f}')
plt.xlabel('Activation Change')
plt.ylabel('Prediction Change')
plt.grid(alpha=0.3)

# Add equation on the plot
equation = f"y = {slope:.4f}x + {intercept:.4f}"
plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', 
             backgroundcolor='white', fontsize=12)

plt.savefig('pred_vs_act_regression.png')
plt.show()

# Print out the relationship summary
print(f"Linear Relationship Summary:")
print(f"  Slope: {slope:.6f}")
print(f"  Intercept: {intercept:.6f}")
print(f"  R² coefficient: {r2:.6f}")
print(f"  Equation: Prediction Change = {slope:.4f} × Activation Change + {intercept:.4f}")
```

```python
# Per-input analysis to see which inputs are most affected
per_input_sensitivity = {}

# Organize data by input
for inp in test_inputs:
    points = [p for p in pred_vs_act if p["input"] == inp]
    
    if len(points) >= 2:  # Need at least 2 points for regression
        X_inp = np.array([p["act_change"] for p in points]).reshape(-1, 1)
        y_inp = np.array([p["pred_change"] for p in points])
        
        # Fit individual model
        inp_model = LinearRegression()
        inp_model.fit(X_inp, y_inp)
        
        # Store results
        per_input_sensitivity[inp] = {
            "slope": float(inp_model.coef_[0]),
            "r2": inp_model.score(X_inp, y_inp),
            "points": len(points),
            "X": X_inp.flatten().tolist(),
            "y": y_inp.tolist()
        }

# Sort inputs by sensitivity (slope)
sorted_inputs = sorted(per_input_sensitivity.items(), 
                      key=lambda x: abs(x[1]["slope"]), 
                      reverse=True)

# Print results
print("Per-input sensitivity analysis:")
print("-" * 80)
print(f"{'Input':<50} | {'Slope':>10} | {'R²':>10} | {'Points':>6}")
print("-" * 80)
for inp, data in sorted_inputs:
    # Truncate input text
    short_inp = inp[:47] + "..." if len(inp) > 47 else inp
    print(f"{short_inp:<50} | {data['slope']:>10.4f} | {data['r2']:>10.4f} | {data['points']:>6}")

# Visualize individual input models
plt.figure(figsize=(15, 12))
n_inputs = len(sorted_inputs)
rows = (n_inputs + 1) // 2  # Calculate rows needed

for i, (inp, data) in enumerate(sorted_inputs[:min(n_inputs, 8)]):  # Show at most 8 inputs
    plt.subplot(rows, 2, i+1)
    
    # Plot points
    plt.scatter(data["X"], data["y"], alpha=0.7)
    
    # Plot regression line
    x_range = np.array([min(data["X"]), max(data["X"])])
    slope = data["slope"]
    intercept = 0  # Assuming zero intercept for simplicity
    plt.plot(x_range, slope * x_range + intercept, 'r-')
    
    # Add info
    short_title = inp[:30] + "..." if len(inp) > 30 else inp
    plt.title(f"{short_title}\nSlope: {slope:.4f}, R²: {data['r2']:.4f}")
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(alpha=0.3)
    
plt.tight_layout()
plt.savefig('per_input_sensitivity.png')
plt.show()
```

```python
# Fit a linear model to quantify the relationship
from sklearn.linear_model import LinearRegression

# Prepare data
X = np.array([p["act_change"] for p in pred_vs_act]).reshape(-1, 1)
y = np.array([p["pred_change"] for p in pred_vs_act])

# Fit model
model = LinearRegression()
model.fit(X, y)

# Calculate R² score
r2 = model.score(X, y)
slope = model.coef_[0]
intercept = model.intercept_

# Plot the data with regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7)
plt.plot(
    [X.min(), X.max()], 
    [model.predict([[X.min()]])[0], model.predict([[X.max()]])[0]], 
    'r-', linewidth=2
)

plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.title(f'Linear Relationship between Activation and Prediction Changes\nSlope: {slope:.4f}, R²: {r2:.4f}')
plt.xlabel('Activation Change')
plt.ylabel('Prediction Change')
plt.grid(alpha=0.3)
plt.savefig('pred_vs_act_regression.png')
plt.show()

# Save results
summary = {
    "model_path": model_path,
    "head_type": config["head_type"],
    "target_layer": target_layer,
    "target_neuron": target_neuron,
    "baseline_correlation": np.corrcoef(baseline_activations, baseline_predictions)[0,1],
    "slope": float(slope),
    "intercept": float(intercept),
    "r_squared": float(r2),
    "scale_factors": scale_factors,
    "avg_act_changes": act_changes,
    "avg_pred_changes": pred_changes,
    "timestamp": str(np.datetime64('now'))
}

os.makedirs("../mech_interp/results", exist_ok=True)
results_path = f"../mech_interp/results/activation_patching_{target_layer}_{target_neuron}.json"
with open(results_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Results saved to {results_path}")
```

## Implementation Notes

1. **Enhanced Robustness**:
   - Added better error handling for different model architectures
   - Improved support for all head types (regression, classification, token)
   - Added statistical significance testing with random baselines
   - Added position-aware patching to precisely target the right tokens

2. **Added Statistical Analysis**:
   - Added weight distribution analysis and detailed statistics
   - Included PCA visualization to see relationships between vectors
   - Added random baseline comparison to establish statistical significance
   - Added linear regression to quantify patching relationships
   - Added per-input sensitivity analysis to identify which inputs are most affected

3. **Better Visualization**:
   - More informative plots with proper annotations
   - Statistical metrics shown directly on visualizations
   - Results saved in structured JSON format for further analysis
   - Added multi-panel plots to show relationships from different perspectives

## Next Steps

After implementing these notebooks, you can:

1. Run the analysis on multiple models with different training configurations
2. Compare results across different neurons and layers
3. Extend with additional mechanistic interpretability techniques
4. Use the findings to guide future model training approaches