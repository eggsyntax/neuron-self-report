# Neuron Self-Report

This project implements a system for training transformer models to "self-report" their internal activation values. The goal is to explore a novel approach to interpretability by leveraging a model's own representations to predict its internal states.

## Project Overview

The system enables a transformer model to predict specific MLP neuron activations from its own internal representations. This creates a form of introspective capability, allowing us to investigate what the model has learned about its own workings.

The core workflow involves:
1.  **Neuron Selection**: Identifying MLP neurons with interesting activation patterns (e.g., high variance, selective firing).
2.  **Dataset Generation**: Creating datasets that map text inputs to the activation values of selected neurons.
3.  **Model Training**: Fine-tuning the base model (or a prediction head attached to it) to predict these neuron activations.
4.  **Evaluation & Analysis**: Assessing the prediction performance and analyzing what the model learns during this process.

## System Components
-   `scanner.py`: Contains `NeuronScanner` for identifying interesting neurons.
-   `dataset.py`: Contains `ActivationDatasetGenerator` for creating activation datasets.
-   `architecture.py`: Contains `ActivationPredictor` for defining models that predict activations, including various head types (regression, classification, token-based).
-   `trainer.py`: Contains `PredictorTrainer` for handling the training process, including different unfreezing strategies and monitoring.
-   `neuron_self_report.py`: The main script for orchestrating the entire pipeline, loading configurations, and running experiments.
-   `mech_interp-notebooks/`: (To be added later) Jupyter notebooks for more in-depth mechanistic interpretability analyses.

## Setup and Installation

1.  Clone the repository.
2.  Install required dependencies:
    ```bash
    pip install torch transformer-lens transformers numpy matplotlib pandas tqdm scikit-learn wandb
    ```
3.  Ensure you have environment variables set up for Weights & Biases if you intend to use it for logging (`WANDB_API_KEY`).

## Usage

The main pipeline is run via `neuron_self_report.py`. It loads experimental settings from a JSON configuration file.

Example configuration parameters (see `plan.md` for a more detailed list):
-   Model name (e.g., `gpt2-small`)
-   Target neuron layer and index
-   Dataset source (synthetic or provided)
-   Prediction head type (regression, classification, token)
-   Training hyperparameters (learning rate, epochs, batch size)
-   Unfreezing strategy

Refer to `plan.md` for a detailed project plan and architecture.
