# Neural Network Introspection System

## Introduction

The Neural Network Introspection System enables transformer models to "self-report" their internal activation values. The system fine-tunes a model to accurately predict the activation strength of a specific neuron or residual stream value from the model's own representations, creating a form of introspective capability.

This project demonstrates a novel approach to neural network interpretability: rather than using external probes to analyze the network, we use the network's own representations to predict specific activations, allowing us to study how the model can understand its own internal states.

Core features:
- Identification of interesting neurons based on activation patterns
- Generation of datasets mapping text inputs to activation values
- Fine-tuning models to predict internal activations
- Comprehensive monitoring of training dynamics
- Analysis of activation distribution changes during training
- Visualization of gradient flow and activation patterns

## User Information

### Installation

1. Clone this repository:
```bash
git clone https://github.com/username/neural-introspection.git
cd neural-introspection
```

2. Install required packages:
```bash
pip install torch transformers numpy matplotlib pandas tqdm scikit-learn transformer_lens
```

3. Verify installation by running a simple test:
```bash
python neural_introspection.py --model gpt2-small --help
```

### Basic Usage

The system provides an end-to-end pipeline for neural introspection:

```bash
# Basic usage with automatic neuron selection
python neural_introspection.py --model gpt2-small --output-dir ./results

# Using a specific neuron
python neural_introspection.py --model gpt2-small --layer 6 --neuron 500 --output-dir ./results

# With increased dataset size
python neural_introspection.py --model gpt2-small --num-samples 1000 --output-dir ./results
```

### Command Line Options

#### Model Selection
- `--model` (default: `gpt2-small`): Base transformer model to use
  - This specifies which pre-trained model to analyze and fine-tune
  - Any model supported by TransformerLens can be used
  
- `--layer` (default: `None`): Target layer to analyze
  - If not specified, the system will scan for the most interesting layer
  - For GPT2-small, valid values are 0-11
  
- `--neuron` (default: `None`): Target neuron to analyze
  - If not specified, the system will scan for the most interesting neuron
  - The range depends on the model architecture (e.g., 0-3071 for GPT2-small MLPs)
  
- `--layer-type` (default: `mlp_out`): Type of layer to analyze
  - `mlp_out`: Output of MLP layers (post-activation)
  - `resid_post`: Values in the residual stream after a layer
  
- `--token-pos` (default: `last`): Token position to analyze
  - `last`: The last token in each sequence
  - Integer: A specific token position (e.g., 0 for the first token)

#### Dataset Parameters
- `--num-samples` (default: `200`): Number of text samples to generate/use
  - Larger datasets generally lead to better results but slower training
  
- `--texts` (default: `None`): Path to file with custom text samples
  - File should contain one text sample per line
  - If not provided, the system generates synthetic samples
  
- `--output-dir` (default: `output`): Directory for outputs
  - The system creates subdirectories for datasets, models, and visualizations
  
- `--force-overwrite` (default: `False`): Overwrite existing output
  - By default, existing output is archived with a timestamp

#### Training Parameters
- `--head-type` (default: `regression`): Type of prediction head
  - `regression`: Predict continuous activation values (standard approach)
  - `classification`: Use classification with binned activation values
  - `token`: Train the model to output digit tokens (0-9) representing activation strength
  
- `--num-bins` (default: `10`): Number of bins for classification
  - Only used when `--head-type` is set to `classification`
  - More bins provide finer granularity but require more data
  
- `--batch-size` (default: `16`): Batch size for training
  - Larger batches can improve stability but use more memory
  
- `--epochs` (default: `20`): Number of training epochs
  - Early stopping may terminate training before this limit
  
- `--learning-rate` (default: `1e-4`): Training learning rate
  - Lower values may improve stability; higher values train faster
  
- `--val-split` (default: `0.15`): Validation set size (fraction)
  - Portion of data used for validation during training
  
- `--test-split` (default: `0.15`): Test set size (fraction)
  - Portion of data used for final evaluation
  
- `--early-stopping` (default: `True`): Enable early stopping
  - Stops training when validation loss stops improving
  
- `--feature-layer` (default: `-1`): Layer to extract features from
  - `-1`: Use the final layer
  - Can be any layer index (usually a later layer is better)

#### Unfreezing Parameters
- `--unfreeze` (default: `none`): Unfreezing strategy
  - `none`: Freeze entire model (only train prediction head)
  - `all`: Unfreeze entire model while keeping head frozen (forces model to adapt)
  - `after_target`: Unfreeze layers after target neuron's layer
  - `from_layer`: Unfreeze from specific layer onwards
  - `selective`: Unfreeze specific component types
  - Note: When any unfreezing strategy other than `none` is used, prediction heads remain frozen
  
- `--unfreeze-from` (default: `None`): Layer to start unfreezing from
  - Only used with `--unfreeze from_layer`
  
- `--unfreeze-components` (default: `""`): Components to unfreeze
  - Comma-separated list: `attention,mlp,embeddings`
  - Only used with `--unfreeze selective`

#### Monitoring Parameters
- `--track-gradients` (default: `False`): Track gradient flow
  - Monitors how gradients propagate through the network
  
- `--track-activations` (default: `False`): Track activation distributions
  - Monitors how neuron behavior changes during training
  
- `--activation-interval` (default: `5`): Activation tracking frequency
  - How often to record activation distributions (in epochs)
  
- `--gradient-interval` (default: `50`): Gradient tracking frequency
  - How often to record gradient statistics (in steps)

#### Hardware Parameters
- `--device` (default: `None`): Device to use
  - Auto-detects MPS (Apple Silicon), CUDA, or CPU if not specified

#### Details on monitoring:

**Activation Distribution Monitoring** tracks how a target neuron's behavior changes throughout the training process, providing direct evidence of whether the model is modifying its internal representations. The system captures activation values before training begins and at regular intervals during training, generating visualizations that compare the initial and final distributions alongside statistical measures of the changes (like KL divergence and distribution shifts). This feature is particularly valuable for testing hypotheses about neural self-modification, regularization effects, and understanding how fine-tuning alters specific neurons' behavior patterns.

When examining the activation distribution visualizations, pay attention to changes in the distribution shape, mean, and variance. A narrowing distribution suggests the neuron is becoming more predictable or specialized, while shifts in the mean indicate the neuron is systematically changing its firing threshold. The comparison plot highlights before/after differences, while the evolution plot shows progressive changes across training epochs. The KL divergence metric quantifies the overall distributional shift - higher values indicate more dramatic changes to the neuron's behavior. These insights help determine whether training is merely adapting to the neuron's existing behavior or actively modifying how the neuron itself operates.

**Gradient Flow Analysis** tracks how gradients propagate through different layers of the neural network during training, revealing which components are actively learning and how information flows backward through the model. By monitoring gradient statistics (mean magnitude, norm, and maximum values) at regular intervals, it generates visualizations showing gradient behavior across different layers over time. This helps identify issues like vanishing or exploding gradients and understand which parts of the network contribute most to learning.

When interpreting the gradient visualization plots, focus on the relative differences between layers rather than absolute values. Layers with consistently higher gradient magnitudes are learning more actively, while declining gradients over time may indicate approaching convergence. The gradient norm by layer chart shows overall gradient health, while the layer-wise gradient plots reveal if certain components (like attention mechanisms or specific layers) are dominating the learning process. Sharp spikes or unusual patterns often indicate potential issues that may require adjustment to hyperparameters or architecture.

### Cookbook

#### 1. Basic Linear Probe (Default)

The default configuration trains a simple regression head while freezing the entire model:

```bash
python neural_introspection.py --model gpt2-small --head-type regression
```

This is equivalent to a linear probe setup and answers: "Can the model's existing representations predict specific neuron activations?"

#### 2. Testing Self-Modification Hypothesis

To test whether the model learns to regularize its neurons to make them more predictable:

```bash
python neural_introspection.py --model gpt2-small --unfreeze all --track-activations
```

This unfreezes the entire model and tracks how activation distributions change during training.

#### 3. Testing Causal Influence

To understand how layers after the target neuron influence prediction:

```bash
python neural_introspection.py --model gpt2-small --unfreeze after_target --track-gradients
```

This unfreezes only layers after the target neuron and tracks gradient flow.

#### 4. Finding Interesting Neurons

To scan for neurons with diverse activation patterns:

```bash
python neural_introspection.py --model gpt2-small
```

The system automatically selects the most "interesting" neuron (high variance and range).

#### 5. Analyzing a Specific Neuron

To analyze a specific neuron you've identified:

```bash
python neural_introspection.py --model gpt2-small --layer 6 --neuron 500
```

#### 6. Comprehensive Analysis

For maximum insight, enable all monitoring features:

```bash
python neural_introspection.py --model gpt2-small --unfreeze after_target --track-gradients --track-activations --num-samples 500
```

#### 7. Using Token Prediction

To train the model to output digit tokens representing activation strength:

```bash
python neural_introspection.py --model gpt2-small --head-type token --unfreeze all
```

This approach requires model unfreezing and trains the model to use its standard token prediction path.

#### 8. Optimizing for Performance

For best prediction performance:

```bash
python neural_introspection.py --model gpt2-small --num-samples 1000 --batch-size 32 --epochs 50 --learning-rate 5e-5
```

## Developer Information

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- HuggingFace Transformers
- TransformerLens
- Matplotlib, NumPy, Pandas
- tqdm
- scikit-learn

### Architecture

The project is organized into four main modules:

1. **NeuronScanner** (`neuron_selection/scanner.py`)
   - Identifies neurons with diverse activation patterns
   - Visualization of activation distributions
   - Scoring mechanism for neuron interestingness

2. **ActivationDatasetGenerator** (`dataset/generator.py`)
   - Creates datasets from model activations
   - Supports both classification and regression approaches
   - CSV-based persistence to avoid PyTorch unpickling issues

3. **ActivationPredictor** (`architecture/architecture.py`)
   - Architecture for predicting activations with three head types:
     - Regression: Continuous value prediction
     - Classification: Binned activation prediction
     - Token: Uses model's token prediction path for digit output (0-9)
   - Uses the model's own representations to predict specific neuron values
   - Implements both training and inference workflows

4. **PredictorTrainer** (`training/trainer.py`)
   - Training module with validation and visualization
   - Implements gradient and activation monitoring
   - Controls model unfreezing strategies
   - Keeps prediction heads frozen when unfreezing parts of the model

5. **Neural Introspection Pipeline** (`neural_introspection.py`)
   - End-to-end pipeline integrating all components
   - Command-line interface with extensive options
   - Visualization and result analysis

### Key Design Patterns

1. **Module Interdependencies**

```
neural_introspection.py
  ↓
  ├── NeuronScanner
  ↓
  ├── ActivationDatasetGenerator
  ↓
  ├── ActivationPredictor
  ↓
  └── PredictorTrainer
```

2. **Data Flow**

```
Text Inputs → NeuronScanner → Interesting Neuron
           ↓
Text Inputs → ActivationDatasetGenerator → Dataset
                                         ↓
Dataset → ActivationPredictor → PredictorTrainer → Trained Model
                                                 ↓
                                       Evaluation & Visualization
```

3. **Monitoring Components**

```
PredictorTrainer
  ├── GradientTracker (Hooks into model parameters)
  └── ActivationMonitor (Records activation distributions)
```

### Development Recommendations

1. **Testing New Features**

When adding new features, focus on integration with the primary pipeline:
- Add a command-line argument to `neural_introspection.py`
- Implement the core functionality in the appropriate module
- Update the trainer to use the new feature
- Add visualization or reporting capabilities

2. **Extending Model Support**

To support additional model architectures:
- Ensure compatibility with TransformerLens
- Verify that activation access patterns work correctly
- Test neuron scanning with the new architecture
- May require adjustments to layer/neuron indexing

3. **Adding Monitoring Capabilities**

When adding new monitoring tools:
- Implement as a self-contained class in `trainer.py`
- Use hooks rather than modifying existing components
- Include visualization generation
- Export raw data in JSON format for external analysis

4. **Common Pitfalls**

- **Tensor/Device Management**: Always use `.detach().cpu().numpy()` pattern
- **Gradient Tracking**: Handle requires_grad=False carefully
- **Memory Management**: Avoid storing large tensors; save statistics instead
- **Backward Compatibility**: Maintain optional flags for new features

5. **Code Style**

- Type annotations for all functions
- Comprehensive docstrings (Google style)
- Log important events and metrics
- Use tqdm for progress tracking
- Descriptive variable names

## Future Directions

Potential areas for enhancement:

1. **Multi-neuron training**: Extending to network-wide introspection
2. **Enhanced evaluation framework**: Standardized benchmarks
3. **YAML-based configuration**: Experimental configuration
4. **Ensembling approaches**: Multi-head predictions
5. **Meta-learning approaches**: Quick adaptation to new neurons
6. **Activation distillation**: Knowledge distillation for activations
7. **Advanced token prediction**: Extending token prediction beyond digit tokens
8. **Comparative analysis**: Systematically comparing head types across different neurons

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:

```
@software{neural_introspection,
  author = {Author},
  title = {Neural Network Introspection System},
  year = {2025},
  url = {https://github.com/username/neural-introspection},
}
```

## Acknowledgments

This project builds on TransformerLens and was inspired by work in mechanistic interpretability.