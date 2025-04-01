# trainer.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
import json
import time
from tqdm.auto import tqdm
import logging
from sklearn.model_selection import train_test_split
from datetime import datetime

# Get the project root directory (CODE1)
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from dataset.generator import ActivationDataset, ActivationDatasetGenerator
from architecture.architecture import ActivationPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trainer")

class PredictorTrainer:
    """Trainer for ActivationPredictor models"""

    def __init__(
        self,
        predictor: ActivationPredictor,
        dataset: ActivationDataset,
        output_dir: str = "training_output",
        device: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize trainer for activation prediction model.

        Args:
            predictor: Model to train
            dataset: Dataset for training
            output_dir: Directory for saving outputs
            device: Device to train on (if None, detect automatically)
            seed: Random seed for reproducibility
        """
        self.predictor = predictor
        self.dataset = dataset
        self.output_dir = output_dir
        self.seed = seed

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # Move predictor to device
        self.predictor.to(device)
        logger.info(f"Trainer initialized on device: {device}")

        # Track training metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "best_val_loss": float("inf"),
            "best_epoch": -1,
            "train_time": 0,
        }

        # No splits or data loaders yet
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def prepare_data(
        self,
        val_split: float = 0.1,
        test_split: float = 0.1,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        """
        Prepare dataset splits and create DataLoaders.

        Args:
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of workers for data loading
        """
        logger.info("Preparing data splits and loaders")

        # Split dataset
        dataset_size = len(self.dataset)
        test_size = int(dataset_size * test_split)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size - test_size

        # Ensure we have at least one sample in each split
        if train_size <= 0 or val_size <= 0 or test_size <= 0:
            raise ValueError(
                f"Dataset too small ({dataset_size}) for the requested splits. "
                f"Got train={train_size}, val={val_size}, test={test_size}"
            )

        # Create splits
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        logger.info(f"Dataset split: train={train_size}, val={val_size}, test={test_size}")

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        logger.info(f"Created DataLoaders with batch_size={batch_size}")

        # Log data insights
        self._log_data_insights()

    def _log_data_insights(self):
        """Log insights about the dataset"""
        if self.dataset.output_tokens:
            # For classification, look at class distribution
            labels = [self.dataset[i]["labels"].item() for i in range(len(self.dataset))]
            unique_labels, counts = np.unique(labels, return_counts=True)

            logger.info(f"Classification dataset with {len(unique_labels)} classes")
            for label, count in zip(unique_labels, counts):
                logger.info(f"  Class {label}: {count} samples ({count/len(labels)*100:.1f}%)")
        else:
            # For regression, look at value distribution
            values = [self.dataset[i]["labels"].item() for i in range(len(self.dataset))]

            logger.info(f"Regression dataset with range [{min(values):.4f}, {max(values):.4f}]")
            logger.info(f"  Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")

    def train(
        self,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        lr_scheduler: str = "linear",
        lr_scheduler_kwargs: Optional[Dict] = None,
        early_stopping: bool = True,
        patience: int = 3,
        grad_clip: Optional[float] = 1.0,
        log_interval: int = 10,
        save_best: bool = True,
        eval_during_training: bool = True,
        freeze_base_model: bool = True,
    ) -> Dict:
        """
        Train the activation predictor model.

        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            lr_scheduler: Learning rate scheduler ('linear', 'cosine', 'exponential', 'none')
            lr_scheduler_kwargs: Extra parameters for the scheduler
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            grad_clip: Max gradient norm (None for no clipping)
            log_interval: Logging interval in steps
            save_best: Whether to save the best model
            eval_during_training: Whether to evaluate on validation set during training
            freeze_base_model: Whether to freeze the base transformer model

        Returns:
            Dictionary of training metrics
        """
        if self.train_loader is None:
            raise ValueError("Data loaders not initialized. Call prepare_data() first.")

        # Freeze base model if requested
        if freeze_base_model:
            logger.info("Freezing base transformer model")
            for param in self.predictor.base_model.parameters():
                param.requires_grad = False

        # Only optimize parameters that require gradients
        trainable_params = [p for p in self.predictor.parameters() if p.requires_grad]
        logger.info(f"Training {sum(p.numel() for p in trainable_params)} parameters")

        # Create optimizer
        optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Create learning rate scheduler
        scheduler = self._create_lr_scheduler(
            optimizer,
            scheduler_type=lr_scheduler,
            epochs=epochs,
            kwargs=lr_scheduler_kwargs or {},
        )

        # Create loss function
        loss_fn = self._get_loss_function()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        start_time = time.time()

        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            # Training phase
            self.predictor.train()
            train_loss = 0.0
            train_metrics = {}

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                optimizer.zero_grad()

                # For classification
                if self.predictor.head_type == "classification":
                    outputs = self.predictor(input_ids, attention_mask)
                    loss = loss_fn(outputs, labels)
                # For regression
                else:
                    outputs = self.predictor(input_ids, attention_mask)
                    loss = loss_fn(outputs, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

                # Optimizer step
                optimizer.step()

                # Track loss
                train_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})

                # Logging
                if step % log_interval == 0:
                    logger.debug(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

            # End of epoch
            train_loss /= len(self.train_loader)
            self.metrics["train_loss"].append(train_loss)

            # Compute training metrics
            train_metrics = self._compute_metrics_on_split("train")
            self.metrics["train_metrics"].append(train_metrics)

            # Validation phase
            if eval_during_training:
                val_loss, val_metrics = self._evaluate()
                self.metrics["val_loss"].append(val_loss)
                self.metrics["val_metrics"].append(val_metrics)

                logger.info(
                    f"Epoch {epoch+1}: "
                    f"Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, "
                    f"{self._format_metrics(train_metrics, 'Train')}, "
                    f"{self._format_metrics(val_metrics, 'Val')}"
                )

                # Check for best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.metrics["best_val_loss"] = val_loss
                    self.metrics["best_epoch"] = epoch + 1
                    patience_counter = 0

                    # Save best model
                    if save_best:
                        self._save_model("best_model")
                        logger.info(f"Saved best model at epoch {epoch+1} with val_loss={val_loss:.4f}")
                else:
                    patience_counter += 1

                # Early stopping
                if early_stopping and patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                logger.info(
                    f"Epoch {epoch+1}: "
                    f"Train Loss={train_loss:.4f}, "
                    f"{self._format_metrics(train_metrics, 'Train')}"
                )

            # Learning rate scheduler step
            if scheduler is not None:
                if lr_scheduler == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

        # End of training
        end_time = time.time()
        training_time = end_time - start_time
        self.metrics["train_time"] = training_time
        logger.info(f"Training completed in {training_time:.1f} seconds")

        # Save final model
        self._save_model("final_model")

        # Save training metrics
        self._save_metrics()

        # Generate visualizations
        self._generate_visualizations()

        # Evaluate on test set
        test_loss, test_metrics = self._evaluate_on_test()
        logger.info(f"Test Loss: {test_loss:.4f}, {self._format_metrics(test_metrics, 'Test')}")

        return self.metrics

    def _create_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        epochs: int,
        kwargs: Dict,
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        if scheduler_type.lower() == "none":
            return None

        num_training_steps = len(self.train_loader) * epochs

        if scheduler_type.lower() == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=kwargs.get("end_factor", 0.1),
                total_iters=num_training_steps,
            )
        elif scheduler_type.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps,
                eta_min=kwargs.get("eta_min", 0),
            )
        elif scheduler_type.lower() == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get("gamma", 0.9),
            )
        elif scheduler_type.lower() == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 2),
                min_lr=kwargs.get("min_lr", 1e-6),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def _get_loss_function(self) -> Callable:
        """Get appropriate loss function based on head type"""
        if self.predictor.head_type == "classification":
            # For classification, use CrossEntropyLoss
            if hasattr(self.predictor.head, "class_weights") and self.predictor.head.class_weights is not None:
                return nn.CrossEntropyLoss(weight=self.predictor.head.class_weights)
            else:
                return nn.CrossEntropyLoss()
        else:
            # For regression, use MSELoss or other appropriate loss
            return nn.MSELoss()

    def _evaluate(self) -> Tuple[float, Dict]:
        """Evaluate model on validation set"""
        return self._evaluate_on_split(self.val_loader, "val")

    def _evaluate_on_test(self) -> Tuple[float, Dict]:
        """Evaluate model on test set"""
        return self._evaluate_on_split(self.test_loader, "test")

    def _evaluate_on_split(
        self,
        data_loader: DataLoader,
        split_name: str
    ) -> Tuple[float, Dict]:
        """
        Evaluate model on given data loader.

        Args:
            data_loader: DataLoader for evaluation
            split_name: Name of the data split (for logging)

        Returns:
            Tuple of (average loss, metrics dict)
        """
        self.predictor.eval()
        total_loss = 0
        all_outputs = []
        all_labels = []

        loss_fn = self._get_loss_function()

        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.predictor(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)

                # Track loss and outputs
                total_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        # Calculate average loss
        avg_loss = total_loss / len(data_loader)

        # Concatenate all outputs and labels
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute metrics
        metrics = self._compute_metrics(all_outputs, all_labels)

        return avg_loss, metrics

    def _compute_metrics_on_split(self, split_name: str) -> Dict:
        """Compute metrics on a specific data split"""
        if split_name == "train":
            return self._evaluate_on_split(self.train_loader, "train")[1]
        elif split_name == "val":
            return self._evaluate_on_split(self.val_loader, "val")[1]
        elif split_name == "test":
            return self._evaluate_on_split(self.test_loader, "test")[1]
        else:
            raise ValueError(f"Unknown split name: {split_name}")

    def _compute_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict:
        """
        Compute evaluation metrics based on head type.

        Args:
            outputs: Model predictions
            labels: Ground truth labels

        Returns:
            Dictionary of metrics
        """
        # Convert to numpy for metric computation
        outputs_np = outputs.detach().numpy()
        labels_np = labels.detach().numpy()

        # Use the metrics implementation from ActivationPredictor
        return self.predictor.compute_metrics(outputs_np, labels_np)

    def _format_metrics(self, metrics: Dict, prefix: str = "") -> str:
        """Format metrics for logging"""
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted.append(f"{prefix}_{key}={value:.4f}")
            else:
                formatted.append(f"{prefix}_{key}={value}")
        return ", ".join(formatted)

    def _save_model(self, name: str):
        """Save model to disk"""
        save_path = os.path.join(self.output_dir, name)
        self.predictor.save(save_path)

    def _save_metrics(self):
        """Save training metrics to disk"""
        metrics_path = os.path.join(self.output_dir, "training_metrics.json")

        # Convert numpy values to Python types for JSON serialization
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list):
                serializable_metrics[key] = [
                    {k: float(v) if isinstance(v, (np.number, np.bool_)) else v
                     for k, v in item.items()}
                    if isinstance(item, dict) else
                    float(item) if isinstance(item, (np.number, np.bool_)) else item
                    for item in value
                ]
            elif isinstance(value, (np.number, np.bool_)):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value

        with open(metrics_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)

    def _generate_visualizations(self):
        """Generate training visualizations"""
        # Create a timestamp for the visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Loss curve
        plt.figure(figsize=(10, 6))
        epochs = list(range(1, len(self.metrics["train_loss"]) + 1))
        plt.plot(epochs, self.metrics["train_loss"], "b-", label="Training Loss")

        if "val_loss" in self.metrics and len(self.metrics["val_loss"]) > 0:
            plt.plot(epochs, self.metrics["val_loss"], "r-", label="Validation Loss")

        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, f"loss_curve_{timestamp}.png"))

        # Additional metric curves
        if self.predictor.head_type == "classification":
            if len(self.metrics["train_metrics"]) > 0 and "accuracy" in self.metrics["train_metrics"][0]:
                plt.figure(figsize=(10, 6))
                train_acc = [m["accuracy"] for m in self.metrics["train_metrics"]]
                plt.plot(epochs, train_acc, "b-", label="Training Accuracy")

                if len(self.metrics["val_metrics"]) > 0:
                    val_acc = [m["accuracy"] for m in self.metrics["val_metrics"]]
                    plt.plot(epochs, val_acc, "r-", label="Validation Accuracy")

                plt.title("Training and Validation Accuracy")
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(self.output_dir, f"accuracy_curve_{timestamp}.png"))
        else:  # Regression
            if len(self.metrics["train_metrics"]) > 0 and "mse" in self.metrics["train_metrics"][0]:
                plt.figure(figsize=(10, 6))
                train_mse = [m["mse"] for m in self.metrics["train_metrics"]]
                plt.plot(epochs, train_mse, "b-", label="Training MSE")

                if len(self.metrics["val_metrics"]) > 0:
                    val_mse = [m["mse"] for m in self.metrics["val_metrics"]]
                    plt.plot(epochs, val_mse, "r-", label="Validation MSE")

                plt.title("Training and Validation MSE")
                plt.xlabel("Epochs")
                plt.ylabel("MSE")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(self.output_dir, f"mse_curve_{timestamp}.png"))

        # Save learning curves data for external visualization
        curves_data = {
            "epochs": epochs,
            "train_loss": self.metrics["train_loss"],
            "val_loss": self.metrics.get("val_loss", []),
            "train_metrics": self.metrics["train_metrics"],
            "val_metrics": self.metrics.get("val_metrics", []),
        }

        # Convert numpy values to Python types for JSON serialization
        serializable_curves = {}
        for key, value in curves_data.items():
            if isinstance(value, np.ndarray):
                serializable_curves[key] = value.tolist()
            elif isinstance(value, list):
                if all(isinstance(item, np.ndarray) for item in value):
                    serializable_curves[key] = [v.tolist() for v in value]
                elif all(isinstance(item, dict) for item in value):
                    serializable_curves[key] = [
                        {k: float(v) if isinstance(v, (np.number, np.bool_)) else v
                         for k, v in item.items()}
                        for item in value
                    ]
                else:
                    serializable_curves[key] = [
                        float(v) if isinstance(v, (np.number, np.bool_)) else v
                        for v in value
                    ]
            else:
                serializable_curves[key] = value

        with open(os.path.join(self.output_dir, f"learning_curves_{timestamp}.json"), "w") as f:
            json.dump(serializable_curves, f, indent=2)

    def evaluate_and_visualize_predictions(
        self,
        split: str = "test",
        num_samples: int = 50,
        output_file: Optional[str] = None,
    ):
        """
        Evaluate model and visualize predictions vs. actual activations.
        
        Args:
            split: Data split to visualize ('train', 'val', or 'test')
            num_samples: Number of samples to visualize
            output_file: Output file name (if None, auto-generate)
        """
        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        elif split == "test":
            dataset = self.test_dataset
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if dataset is None:
            raise ValueError(f"Dataset split '{split}' not initialized")
        
        # Limit the number of samples
        num_samples = min(num_samples, len(dataset))
        
        # Get sample indices
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        # Collect texts, predictions, and actual activations
        texts = []
        predictions = []
        actuals = []
        
        for idx in indices:
            # Get sample
            sample = dataset[idx]
            text_ids = sample["input_ids"]
            
            # Get original text
            if hasattr(dataset, "dataset"):  # If using subset
                original_idx = dataset.indices[idx]
                text = dataset.dataset.inputs[original_idx]
            else:
                text = self.dataset.inputs[idx]
            
            texts.append(text)
            
            # Convert to tensor and add batch dimension
            input_ids = text_ids.unsqueeze(0).to(self.device)
            mask = torch.ones_like(input_ids).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                if hasattr(self.predictor, "target_layer") and hasattr(self.predictor, "target_neuron"):
                    outputs, activations = self.predictor(
                        input_ids, mask, return_activations=True
                    )
                    
                    if self.predictor.head_type == "classification":
                        # Get predicted class
                        if self.predictor.bin_edges is not None:
                            # Convert logits to class probabilities
                            probs = torch.softmax(outputs, dim=1)
                            
                            # Get the number of classes from the probability tensor
                            num_classes = probs.shape[1]
                            
                            # Calculate bin centers
                            bin_edges = self.predictor.bin_edges
                            
                            # Handle mismatch between model output classes and bin edges
                            if num_classes > len(bin_edges) - 1:
                                logger.warning(
                                    f"Model produces {num_classes} classes but we only have {len(bin_edges)-1} bin centers. "
                                    f"Extending bin centers with extrapolation."
                                )
                                
                                # Get existing bin centers
                                known_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                                
                                # Estimate bin width from known centers
                                if len(known_centers) > 1:
                                    bin_width = known_centers[1] - known_centers[0]
                                else:
                                    # If only one center, use edge difference
                                    bin_width = bin_edges[1] - bin_edges[0]
                                
                                # Extrapolate additional centers
                                additional_centers = np.array([
                                    known_centers[-1] + (i+1)*bin_width 
                                    for i in range(num_classes - len(known_centers))
                                ])
                                
                                # Combine known and extrapolated centers
                                bin_centers = np.concatenate([known_centers, additional_centers])
                            else:
                                # Standard case: calculate centers from edges
                                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                                
                                # Handle case where model outputs fewer classes than bin centers
                                bin_centers = bin_centers[:num_classes]
                            
                            # Reshape for broadcasting and calculate weighted average
                            pred = (probs.cpu().numpy() * bin_centers.reshape(1, -1)).sum(axis=1)[0]
                        else:
                            pred = torch.argmax(outputs, dim=1).item()
                    else:
                        pred = outputs.item()
                    
                    predictions.append(pred)
                    actuals.append(activations.item())
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Plot predictions vs. actuals
        plt.scatter(actuals, predictions, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(min(actuals), min(predictions))
        max_val = max(max(actuals), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        
        # Calculate correlation and error
        correlation = np.corrcoef(actuals, predictions)[0, 1]
        mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
        
        # Add metrics as text
        plt.title(f"Predictions vs. Actual Activations ({split} split)")
        plt.xlabel("Actual Activations")
        plt.ylabel("Predicted Activations")
        plt.text(
            0.05, 0.95,
            f"Correlation: {correlation:.4f}\nMSE: {mse:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.1),
        )
        
        # Generate output file name if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"predictions_vs_actuals_{split}_{timestamp}.png")
        else:
            output_file = os.path.join(self.output_dir, output_file)
        
        # Save figure
        plt.grid(True, alpha=0.3)
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Saved predictions visualization to {output_file}")
        logger.info(f"Correlation: {correlation:.4f}, MSE: {mse:.4f}")
        
        # Save individual samples to CSV for detailed analysis
        samples_file = os.path.join(self.output_dir, f"prediction_samples_{split}_{timestamp}.csv")
        with open(samples_file, "w") as f:
            f.write("text,actual,predicted,error\n")
            for text, actual, pred in zip(texts, actuals, predictions):
                # Clean text for CSV
                clean_text = text.replace('"', '""')
                error = pred - actual
                f.write(f'"{clean_text}",{actual:.6f},{pred:.6f},{error:.6f}\n')
        
        logger.info(f"Saved individual predictions to {samples_file}")
        
        return {
            "correlation": correlation,
            "mse": mse,
            "predictions": predictions,
            "actuals": actuals,
            "texts": texts,
            "visualization_file": output_file,
            "samples_file": samples_file,
        }

# Function to load dataset from CSV (to work with generator.py output)
def load_activation_dataset_from_files(
    csv_file: str,
    metadata_file: str,
    tokenizer: Any,
    model_max_length: int = 512,
) -> ActivationDataset:
    """
    Load activation dataset from CSV and metadata files.

    Args:
        csv_file: Path to CSV file with text and activations
        metadata_file: Path to metadata file
        tokenizer: Tokenizer for the model
        model_max_length: Maximum sequence length

    Returns:
        ActivationDataset
    """
    import pandas as pd

    # Load data
    df = pd.read_csv(csv_file)

    # Load metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Extract dataset parameters
    dataset_info = metadata.get("dataset_info", {})
    max_length = dataset_info.get("max_length", model_max_length)
    output_tokens = dataset_info.get("output_tokens", False)
    num_bins = dataset_info.get("num_bins", 10)

    # Create dataset
    inputs = df["text"].tolist()
    activations = df["activation"].tolist()

    dataset = ActivationDataset(
        inputs=inputs,
        activations=activations,
        tokenizer=tokenizer,
        max_length=max_length,
        output_tokens=output_tokens,
        num_bins=num_bins,
    )

    # If using tokens and bin_info is available, restore discretization
    if output_tokens and "bin_info" in dataset_info:
        bin_info = dataset_info["bin_info"]
        dataset.min_val = bin_info.get("min_val", min(activations))
        dataset.max_val = bin_info.get("max_val", max(activations))
        dataset.bin_edges = np.array(bin_info.get("bin_edges", []))

        if "bin" in df.columns:
            dataset.discretized = df["bin"].to_numpy()
        else:
            dataset.discretized = np.digitize(activations, dataset.bin_edges[1:])

        dataset.bin_info = bin_info

    return dataset

# Utility function to train a model from saved files
def train_from_saved_dataset(
    dataset_csv: str,
    dataset_metadata: str,
    predictor: ActivationPredictor,
    output_dir: str = "training_output",
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    device: Optional[str] = None,
    **training_kwargs,
) -> PredictorTrainer:
    """
    Train a model using a saved dataset.

    Args:
        dataset_csv: Path to dataset CSV file
        dataset_metadata: Path to dataset metadata file
        predictor: ActivationPredictor to train
        output_dir: Output directory
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        **training_kwargs: Additional training parameters

    Returns:
        Trained PredictorTrainer
    """
    # Load dataset
    dataset = load_activation_dataset_from_files(
        dataset_csv,
        dataset_metadata,
        predictor.tokenizer,
    )

    # Create trainer
    trainer = PredictorTrainer(
        predictor=predictor,
        dataset=dataset,
        output_dir=output_dir,
        device=device,
    )

    # Prepare data
    trainer.prepare_data(batch_size=batch_size)

    # Train model
    trainer.train(
        epochs=epochs,
        learning_rate=learning_rate,
        **training_kwargs,
    )

    return trainer

def test_trainer():
    """Test the PredictorTrainer functionality"""
    import tempfile
    from transformer_lens import HookedTransformer
    # from generator.generator import ActivationDatasetGenerator
    # from architecture.architecture import ActivationPredictor

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up a small model and dataset
        print("Loading test model...")
        model_name = "gpt2-small"
        model = HookedTransformer.from_pretrained(model_name)
        tokenizer = model.tokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Sample texts
        sample_texts = [
            "The cat sat on the mat.",
            "Machine learning models can be difficult to interpret.",
            "Transformers use attention mechanisms to process sequences.",
            "Neural networks have revolutionized artificial intelligence.",
            "The quick brown fox jumps over the lazy dog.",
            "Scientists study the complex patterns in data.",
            "Language models can generate coherent text passages.",
            "The brain processes information through neural connections.",
            "Deep learning has transformed computer vision tasks.",
            "Researchers work to make AI systems more transparent.",
        ]

        # Create dataset generator
        print("Generating dataset...")
        generator = ActivationDatasetGenerator(model, tokenizer, device=device)

        # Pick a random neuron for testing
        layer = 6
        neuron = 500

        # Generate a classification dataset
        dataset, metadata = generator.generate_dataset(
            texts=sample_texts,
            layer=layer,
            neuron_idx=neuron,
            layer_type="mlp_out",
            token_pos="last",
            output_tokens=True,
            num_bins=3,
        )

        unique_classes = np.unique(dataset.discretized)
        print(f"Dataset contains classes: {unique_classes} (max={unique_classes.max()})")

        # Create predictor
        print("Creating predictor...")
        predictor = ActivationPredictor(
            base_model=model,
            head_type="classification",
            num_classes=len(unique_classes),
            target_layer=layer,
            target_neuron=neuron,
            layer_type="mlp_out",
            token_pos="last",
            head_config={"hidden_dim": 32},
            device=device,
            bin_edges=np.array(dataset.bin_info["bin_edges"]),
        )

        # Create trainer
        print("Creating trainer...")
        trainer = PredictorTrainer(
            predictor=predictor,
            dataset=dataset,
            output_dir=temp_dir,
            device=device,
        )

        # Prepare data
        trainer.prepare_data(batch_size=2, val_split=0.2, test_split=0.2)

        # Train for a few epochs
        print("Training model...")
        metrics = trainer.train(
            epochs=2,
            learning_rate=1e-3,
            weight_decay=0.01,
            lr_scheduler="none",
            early_stopping=False,
            log_interval=1,
        )

        print(f"Final training loss: {metrics['train_loss'][-1]:.4f}")

        # Evaluate and visualize
        print("Evaluating and visualizing...")
        eval_results = trainer.evaluate_and_visualize_predictions(split="test")

        print(f"Test correlation: {eval_results['correlation']:.4f}")
        print(f"Test MSE: {eval_results['mse']:.4f}")

        print("\nTrainer test completed successfully!")

        return trainer, eval_results

if __name__ == "__main__":
    test_trainer()

