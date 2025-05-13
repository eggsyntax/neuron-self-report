# trainer.py - Enhanced with unfreezing, gradient monitoring, and activation tracking

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

# Get the project root directory
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from dataset.generator import ActivationDataset, ActivationDatasetGenerator
from architecture.architecture import ActivationPredictor

# Import unfreezing utilities
try:
    from selective_unfreezing import (
        apply_unfreezing_strategy, 
        get_trainable_parameters_info,
        freeze_entire_model,
        unfreeze_entire_model,
        unfreeze_after_layer
    )
    UNFREEZING_AVAILABLE = True
except ImportError:
    UNFREEZING_AVAILABLE = False
    print("Warning: Selective unfreezing utilities not available. Using basic freezing only.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trainer")

class GradientTracker:
    """Tracks gradient flow through the model during training."""
    
    def __init__(self, model: nn.Module, track_interval: int = 10):
        """
        Initialize gradient tracker.
        
        Args:
            model: Model to track gradients for
            track_interval: How often to track gradients (in steps)
        """
        self.model = model
        self.track_interval = track_interval
        self.tracked_gradients = {}
        self.step_counter = 0
        self.tracked_steps = []
        
        # Register hooks for all parameters
        self.hooks = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(lambda grad, name=name: self._save_grad_hook(name, grad))
                self.hooks.append(hook)
                self.tracked_gradients[name] = []
    
    def _save_grad_hook(self, name: str, grad: torch.Tensor):
        """Hook to save gradient statistics."""
        # Only save on the tracking interval
        if self.step_counter % self.track_interval == 0:
            if grad is not None:
                # Save basic statistics rather than whole tensors to save memory
                grad_stats = {
                    'mean': float(grad.abs().mean().item()),
                    'std': float(grad.std().item()),
                    'max': float(grad.abs().max().item()),
                    'norm': float(grad.norm().item()),
                    'step': self.step_counter,
                }
                self.tracked_gradients[name].append(grad_stats)
    
    def step(self):
        """Record a training step."""
        if self.step_counter % self.track_interval == 0:
            self.tracked_steps.append(self.step_counter)
        self.step_counter += 1
    
    def remove_hooks(self):
        """Remove gradient hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_layer_gradients(self):
        """Aggregate gradients by layer."""
        layer_gradients = {}
        
        for name, grad_stats_list in self.tracked_gradients.items():
            if not grad_stats_list:  # Skip if no gradients recorded
                continue
                
            # Try to extract layer information from parameter name
            layer = None
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part in ['blocks', 'layers', 'layer'] and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer = int(parts[i + 1])
                    break
            
            # Group by component if we couldn't identify the layer
            if layer is None:
                if 'head' in name:
                    component = 'head'
                elif 'embed' in name:
                    component = 'embedding'
                elif 'norm' in name or 'ln' in name:
                    component = 'normalization'
                elif 'attn' in name or 'attention' in name:
                    component = 'attention'
                elif 'mlp' in name or 'ffn' in name:
                    component = 'mlp'
                else:
                    component = 'other'
                
                if component not in layer_gradients:
                    layer_gradients[component] = []
                
                # Append all gradient stats
                for stats in grad_stats_list:
                    layer_gradients[component].append(stats)
            else:
                # For actual layers, create entries if they don't exist
                layer_key = f"layer_{layer}"
                if layer_key not in layer_gradients:
                    layer_gradients[layer_key] = []
                
                # Append all gradient stats
                for stats in grad_stats_list:
                    layer_gradients[layer_key].append(stats)
        
        # Compute average stats per layer across all recorded steps
        summary = {}
        for layer, stats_list in layer_gradients.items():
            if not stats_list:
                continue
                
            # Group by step first
            step_groups = {}
            for stats in stats_list:
                step = stats['step']
                if step not in step_groups:
                    step_groups[step] = []
                step_groups[step].append(stats)
            
            # Average across parameters for each step
            step_avgs = []
            for step, step_stats in step_groups.items():
                avg_stats = {
                    'step': step,
                    'mean': np.mean([s['mean'] for s in step_stats]),
                    'std': np.mean([s['std'] for s in step_stats]),
                    'max': np.mean([s['max'] for s in step_stats]),
                    'norm': np.mean([s['norm'] for s in step_stats]),
                }
                step_avgs.append(avg_stats)
            
            # Sort by step
            step_avgs.sort(key=lambda x: x['step'])
            summary[layer] = step_avgs
        
        return summary
    
    def generate_gradient_plots(self, output_dir: str = '.'):
        """Generate visualizations of gradient flow."""
        layer_gradients = self.get_layer_gradients()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract steps for x-axis
        steps = self.tracked_steps
        
        # Generate plots
        # 1. Mean gradient magnitude by layer
        plt.figure(figsize=(12, 8))
        for layer, stats_list in layer_gradients.items():
            if len(stats_list) != len(steps):
                # Skip if we don't have data for all steps
                continue
                
            means = [stats['mean'] for stats in stats_list]
            plt.plot(steps, means, label=layer)
        
        plt.title('Mean Gradient Magnitude by Layer')
        plt.xlabel('Training Step')
        plt.ylabel('Mean Gradient Magnitude')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_means.png'))
        plt.close()
        
        # 2. Gradient norm by layer
        plt.figure(figsize=(12, 8))
        for layer, stats_list in layer_gradients.items():
            if len(stats_list) != len(steps):
                continue
                
            norms = [stats['norm'] for stats in stats_list]
            plt.plot(steps, norms, label=layer)
        
        plt.title('Gradient Norm by Layer')
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_norms.png'))
        plt.close()
        
        # 3. Gradient max by layer
        plt.figure(figsize=(12, 8))
        for layer, stats_list in layer_gradients.items():
            if len(stats_list) != len(steps):
                continue
                
            maxes = [stats['max'] for stats in stats_list]
            plt.plot(steps, maxes, label=layer)
        
        plt.title('Maximum Gradient Magnitude by Layer')
        plt.xlabel('Training Step')
        plt.ylabel('Max Gradient')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_maxes.png'))
        plt.close()
        
        # Save raw data as JSON for further analysis
        grad_data = {
            'steps': steps,
            'layers': {k: v for k, v in layer_gradients.items()}
        }
        
        with open(os.path.join(output_dir, 'gradient_data.json'), 'w') as f:
            json.dump(grad_data, f, indent=2)
        
        return {
            'plots': [
                os.path.join(output_dir, 'gradient_means.png'),
                os.path.join(output_dir, 'gradient_norms.png'),
                os.path.join(output_dir, 'gradient_maxes.png'),
            ],
            'data': os.path.join(output_dir, 'gradient_data.json')
        }

class ActivationMonitor:
    """Monitors neuron activation distributions during training."""
    
    def __init__(
        self, 
        predictor: ActivationPredictor,
        dataset: Dataset,
        batch_size: int = 32,
        device: str = 'cpu',
        monitor_interval: int = 5,  # Every N epochs
    ):
        """
        Initialize activation monitor.
        
        Args:
            predictor: ActivationPredictor model
            dataset: Dataset to monitor activations on
            batch_size: Batch size for processing
            device: Device to run on
            monitor_interval: How often to monitor activations (in epochs)
        """
        self.predictor = predictor
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.monitor_interval = monitor_interval
        
        # Storage for activation distributions
        self.activations = {}
        self.epoch_activations = {}
        self.initial_activations = None
    
    def collect_activations(self, epoch: int):
        """
        Collect activations for the current epoch.
        
        Args:
            epoch: Current epoch number
        """
        # Only collect on the specified interval
        if epoch % self.monitor_interval != 0 and epoch != 0:
            return
        
        # Create dataloader for efficient processing
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        all_activations = []
        
        # Process dataset
        self.predictor.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get actual neuron activations (not predictions)
                if hasattr(self.predictor, 'target_layer') and hasattr(self.predictor, 'target_neuron'):
                    _, activations = self.predictor(
                        input_ids, 
                        attention_mask, 
                        return_activations=True
                    )
                    
                    # Convert to list for storage
                    all_activations.extend(activations.cpu().numpy().tolist())
        
        # Store initial activations separately
        if epoch == 0:
            self.initial_activations = np.array(all_activations)
        
        # Store all activations
        self.epoch_activations[epoch] = np.array(all_activations)
    
    def generate_activation_plots(self, output_dir: str = '.'):
        """
        Generate visualizations of activation distributions.
        
        Args:
            output_dir: Directory to save plots to
            
        Returns:
            Dictionary of plot paths and statistics
        """
        if not self.epoch_activations:
            logger.warning("No activation data collected. Skipping activation plots.")
            return {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate distribution plots
        plots = []
        
        # 1. Initial distribution
        if self.initial_activations is not None:
            plt.figure(figsize=(10, 6))
            plt.hist(self.initial_activations, bins=30, alpha=0.7)
            plt.title(f"Initial Activation Distribution\nLayer {self.predictor.target_layer}, Neuron {self.predictor.target_neuron}")
            plt.xlabel("Activation Value")
            plt.ylabel("Frequency")
            plt.grid(alpha=0.3)
            
            # Add statistics
            mean = np.mean(self.initial_activations)
            std = np.std(self.initial_activations)
            min_val = np.min(self.initial_activations)
            max_val = np.max(self.initial_activations)
            
            stats_text = (
                f"Mean: {mean:.4f}\n"
                f"Std Dev: {std:.4f}\n"
                f"Range: [{min_val:.4f}, {max_val:.4f}]"
            )
            
            plt.annotate(
                stats_text, 
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", alpha=0.1)
            )
            
            initial_plot = os.path.join(output_dir, 'initial_activation_dist.png')
            plt.tight_layout()
            plt.savefig(initial_plot)
            plt.close()
            plots.append(initial_plot)
        
        # 2. Final distribution
        final_epoch = max(self.epoch_activations.keys())
        final_activations = self.epoch_activations[final_epoch]
        
        plt.figure(figsize=(10, 6))
        plt.hist(final_activations, bins=30, alpha=0.7)
        plt.title(f"Final Activation Distribution (Epoch {final_epoch})\nLayer {self.predictor.target_layer}, Neuron {self.predictor.target_neuron}")
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        
        # Add statistics
        mean = np.mean(final_activations)
        std = np.std(final_activations)
        min_val = np.min(final_activations)
        max_val = np.max(final_activations)
        
        stats_text = (
            f"Mean: {mean:.4f}\n"
            f"Std Dev: {std:.4f}\n"
            f"Range: [{min_val:.4f}, {max_val:.4f}]"
        )
        
        plt.annotate(
            stats_text, 
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", alpha=0.1)
        )
        
        final_plot = os.path.join(output_dir, 'final_activation_dist.png')
        plt.tight_layout()
        plt.savefig(final_plot)
        plt.close()
        plots.append(final_plot)
        
        # 3. Comparison plot (if we have initial activations)
        if self.initial_activations is not None:
            plt.figure(figsize=(12, 6))
            
            # Initial distribution
            plt.hist(self.initial_activations, bins=30, alpha=0.5, label=f"Initial", color='blue')
            
            # Final distribution
            plt.hist(final_activations, bins=30, alpha=0.5, label=f"Final (Epoch {final_epoch})", color='red')
            
            plt.title(f"Activation Distribution Change\nLayer {self.predictor.target_layer}, Neuron {self.predictor.target_neuron}")
            plt.xlabel("Activation Value")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Calculate KL divergence or other distribution metrics
            # Use histograms for approximation
            hist_initial, bin_edges = np.histogram(self.initial_activations, bins=30, density=True)
            hist_final, _ = np.histogram(final_activations, bins=bin_edges, density=True)
            
            # Add small epsilon to avoid division by zero
            hist_initial = hist_initial + 1e-10
            hist_final = hist_final + 1e-10
            
            # Normalize
            hist_initial = hist_initial / np.sum(hist_initial)
            hist_final = hist_final / np.sum(hist_final)
            
            # Calculate KL divergence: KL(P||Q) = sum(P * log(P/Q))
            kl_divergence = np.sum(hist_initial * np.log(hist_initial / hist_final))
            
            # Calculate other statistics
            initial_mean = np.mean(self.initial_activations)
            initial_std = np.std(self.initial_activations)
            final_mean = np.mean(final_activations)
            final_std = np.std(final_activations)
            
            # Mean shift
            mean_shift = final_mean - initial_mean
            # Std deviation change (as percentage)
            std_change_pct = (final_std - initial_std) / initial_std * 100 if initial_std != 0 else float('inf')
            
            comparison_stats = {
                'kl_divergence': float(kl_divergence),
                'initial_mean': float(initial_mean),
                'initial_std': float(initial_std),
                'final_mean': float(final_mean),
                'final_std': float(final_std),
                'mean_shift': float(mean_shift),
                'std_change_pct': float(std_change_pct),
            }
            
            # Add comparison statistics
            stats_text = (
                f"KL Divergence: {kl_divergence:.4f}\n"
                f"Mean Shift: {mean_shift:.4f}\n"
                f"StdDev Change: {std_change_pct:.1f}%\n"
                f"Initial: μ={initial_mean:.4f}, σ={initial_std:.4f}\n"
                f"Final: μ={final_mean:.4f}, σ={final_std:.4f}"
            )
            
            plt.annotate(
                stats_text, 
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", alpha=0.1)
            )
            
            comparison_plot = os.path.join(output_dir, 'activation_distribution_change.png')
            plt.tight_layout()
            plt.savefig(comparison_plot)
            plt.close()
            plots.append(comparison_plot)
            
            # 4. Generate evolution plot
            if len(self.epoch_activations) > 2:
                # Track statistics over epochs
                epochs = sorted(self.epoch_activations.keys())
                means = [np.mean(self.epoch_activations[e]) for e in epochs]
                stds = [np.std(self.epoch_activations[e]) for e in epochs]
                
                # Create a 2-panel plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Plot mean evolution
                ax1.plot(epochs, means, 'o-', color='blue')
                ax1.set_ylabel('Mean Activation')
                ax1.set_title(f'Evolution of Activation Distribution\nLayer {self.predictor.target_layer}, Neuron {self.predictor.target_neuron}')
                ax1.grid(alpha=0.3)
                
                # Plot std evolution
                ax2.plot(epochs, stds, 'o-', color='red')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Activation StdDev')
                ax2.grid(alpha=0.3)
                
                evolution_plot = os.path.join(output_dir, 'activation_evolution.png')
                plt.tight_layout()
                plt.savefig(evolution_plot)
                plt.close()
                plots.append(evolution_plot)
        
        # Save raw data for further analysis
        activation_data = {
            'target_layer': self.predictor.target_layer,
            'target_neuron': self.predictor.target_neuron,
            'initial_activations': self.initial_activations.tolist() if self.initial_activations is not None else None,
            'final_activations': final_activations.tolist(),
            'epoch_statistics': {
                epoch: {
                    'mean': float(np.mean(activations)),
                    'std': float(np.std(activations)),
                    'min': float(np.min(activations)),
                    'max': float(np.max(activations)),
                }
                for epoch, activations in self.epoch_activations.items()
            },
            'comparison_statistics': comparison_stats if self.initial_activations is not None else None,
        }
        
        data_file = os.path.join(output_dir, 'activation_data.json')
        with open(data_file, 'w') as f:
            json.dump(activation_data, f, indent=2)
        
        return {
            'plots': plots,
            'data': data_file,
            'statistics': comparison_stats if self.initial_activations is not None else None,
        }

class PredictorTrainer:
    """Trainer for ActivationPredictor models with enhanced monitoring"""

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
        
        # No monitors yet
        self.gradient_tracker = None
        self.activation_monitor = None

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
        track_gradients: bool = False,
        track_activations: bool = False,
        activation_monitor_interval: int = 5,
        gradient_track_interval: int = 50,
        unfreeze_strategy: str = "none",
        unfreeze_from_layer: Optional[int] = None,
        unfreeze_components: Optional[Union[str, List[str]]] = None,
        freeze_base_model: bool = False,  # For backward compatibility
    ) -> Dict:
        """
        Train the activation predictor model with enhanced monitoring and unfreezing.

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
            track_gradients: Whether to track gradient flow during training
            track_activations: Whether to track neuron activation distributions
            activation_monitor_interval: How often to monitor activations (in epochs)
            gradient_track_interval: How often to track gradients (in steps)
            unfreeze_strategy: Strategy for unfreezing layers ('none', 'all', 'after_target', 'from_layer', 'selective')
            unfreeze_from_layer: Layer to start unfreezing from (for 'from_layer' strategy)
            unfreeze_components: Components to unfreeze (for 'selective' strategy)
            freeze_base_model: Legacy parameter for backward compatibility

        Returns:
            Dictionary of training metrics and monitoring results
        """
        if self.train_loader is None:
            raise ValueError("Data loaders not initialized. Call prepare_data() first.")

        # Initialize monitoring if requested
        if track_gradients:
            self.gradient_tracker = GradientTracker(
                self.predictor, 
                track_interval=gradient_track_interval
            )
            logger.info("Gradient tracking enabled")
        
        if track_activations:
            self.activation_monitor = ActivationMonitor(
                self.predictor,
                self.dataset,
                batch_size=self.train_loader.batch_size,
                device=self.device,
                monitor_interval=activation_monitor_interval
            )
            logger.info("Activation monitoring enabled")
            
            # Collect initial activations
            self.activation_monitor.collect_activations(epoch=0)
            logger.info("Initial activation distribution collected")

        # Apply unfreezing strategy
        if UNFREEZING_AVAILABLE:
            # If old 'freeze_base_model' is True, use 'none' strategy (everything frozen)
            if freeze_base_model and unfreeze_strategy == "none":
                # Keep the old behavior
                freeze_entire_model(self.predictor.base_model)
                logger.info("Using legacy freeze_base_model=True (everything frozen)")
            else:
                # Use the new unfreezing system
                unfreezing_results = apply_unfreezing_strategy(
                    model=self.predictor.base_model,
                    strategy=unfreeze_strategy,
                    target_layer=self.predictor.target_layer,
                    from_layer=unfreeze_from_layer,
                    components=unfreeze_components
                )
                
                # Add unfreezing results to metrics
                self.metrics["unfreezing"] = unfreezing_results
                
                logger.info(f"Applied unfreezing strategy: {unfreeze_strategy}")
                logger.info(f"Unfrozen parameters: {unfreezing_results['unfrozen_params']:,} "
                           f"({unfreezing_results['unfrozen_percentage']:.2f}% of base model)")
        else:
            # Fallback to simple freezing if selective unfreezing is not available
            if freeze_base_model:
                for param in self.predictor.base_model.parameters():
                    param.requires_grad = False
                logger.info("Freezing base model parameters (selective unfreezing not available)")

        # Fixed approach: When unfreezing the model, use a fixed projection head
        # This forces the model itself to adapt its representations 
        if unfreeze_strategy == "none":
            # When not unfreezing the model, train only the head
            for param in self.predictor.head.parameters():
                param.requires_grad = True
            logger.info("Training only the prediction head (model layers frozen)")
        else:
            # When unfreezing model layers, keep the head fixed
            for param in self.predictor.head.parameters():
                param.requires_grad = False
            logger.info(f"Using fixed head with trainable model layers (unfreezing strategy: {unfreeze_strategy})")

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

                # For classification and token prediction
                if self.predictor.head_type in ["classification", "token"]:
                    outputs = self.predictor(input_ids, attention_mask)
                    
                    # Check for dimension mismatch in token prediction case
                    if self.predictor.head_type == "token":
                        # Log dimensions for debugging
                        logger.info(f"Token prediction dimensions - outputs: {outputs.shape}, labels: {labels.shape}")
                        
                        # First check if labels are 1D (no second dimension)
                        if len(labels.shape) == 1:
                            # Labels are already in class indices format (0 to n)
                            logger.info("Labels are already in class indices format")
                            
                            # For token prediction, we assume labels are already in the 0-9 range exactly
                            # Just verify this is the case and warn if not
                            if torch.max(labels) > 9 or torch.min(labels) < 0:
                                # This shouldn't happen with our modifications to the dataset generator
                                logger.warning(f"WARNING: Token prediction labels outside 0-9 range: [{torch.min(labels).item()}, {torch.max(labels).item()}]")
                                # Ensure they're in range by clipping
                                labels = torch.clamp(labels, 0, 9)
                        
                        # Handle 2D labels
                        elif len(labels.shape) > 1:
                            # Check for shape mismatch
                            if outputs.shape[1] != labels.shape[1]:
                                logger.info(f"Dimension mismatch in token prediction - outputs: {outputs.shape}, labels: {labels.shape}")
                                # We have a dimension mismatch. Token prediction outputs 10 classes (for digits 0-9)
                                # but our labels might be in one-hot encoding or have a different number of classes.
                            
                            # For token prediction, if labels are one-hot encoded, convert to class indices
                            if labels.shape[1] > 1:
                                # Convert one-hot to indices
                                logger.info("Converting one-hot labels to class indices for token prediction")
                                labels = torch.argmax(labels, dim=1)
                        
                        # For token prediction, verify labels are in the 0-9 range
                        if torch.max(labels) > 9 or torch.min(labels) < 0:
                            # Just warn since we should have already handled this in the dataset
                            logger.warning(f"WARNING: Token prediction labels outside 0-9 range after conversion: [{torch.min(labels).item()}, {torch.max(labels).item()}]")
                            # Ensure they're in range by clipping
                            labels = torch.clamp(labels, 0, 9)
                    
                    loss = loss_fn(outputs, labels)
                # For regression
                else:
                    outputs = self.predictor(input_ids, attention_mask)
                    loss = loss_fn(outputs, labels)
                    
                    # DEBUG: Log raw outputs, labels and loss during training
                    if step == 0 and epoch < 2:  # Only log for first batch in first 2 epochs
                        logger.info(f"DEBUG - Epoch {epoch+1}, First batch:")
                        logger.info(f"  Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
                        logger.info(f"  Raw outputs: {outputs.detach().cpu().numpy()[:3]}")
                        logger.info(f"  Raw labels: {labels.detach().cpu().numpy()[:3]}")
                        logger.info(f"  Loss: {loss.item()}")
                        
                        # If we have normalization params available on the predictor
                        if hasattr(self.predictor, 'activation_mean') and hasattr(self.predictor, 'activation_std'):
                            logger.info(f"  Normalization params - Mean: {self.predictor.activation_mean}, Std: {self.predictor.activation_std}")
                            
                            # If normalized, show what denormalized values would be
                            denorm_outputs = outputs.detach().cpu().numpy() * self.predictor.activation_std + self.predictor.activation_mean
                            logger.info(f"  Denormalized outputs: {denorm_outputs[:3]}")

                # Backward pass
                loss.backward()

                # Gradient clipping
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

                # Track gradients if enabled
                if track_gradients and self.gradient_tracker is not None:
                    self.gradient_tracker.step()
                
                # DEBUG: Check gradient flow in first epoch, first step
                if epoch == 0 and step == 0:
                    # Check if gradients are flowing properly
                    has_grad = 0
                    no_grad = 0
                    for name, param in self.predictor.named_parameters():
                        if param.requires_grad:
                            if param.grad is not None and param.grad.abs().sum().item() > 0:
                                has_grad += 1
                            else:
                                no_grad += 1
                                if no_grad < 5:  # Only log a few to avoid spamming
                                    logger.info(f"DEBUG - No gradient for param: {name}, shape: {param.shape}")
                    
                    logger.info(f"DEBUG - Parameters with gradients: {has_grad}, without gradients: {no_grad}")
                    
                    # Specifically check the target neuron's parameters if available
                    if hasattr(self.predictor, 'target_layer') and hasattr(self.predictor, 'target_neuron'):
                        # Try to access the target neuron's parameters, structure varies by model
                        target_layer = self.predictor.target_layer
                        target_neuron = self.predictor.target_neuron
                        
                        # For GPT-2 style models, try to find MLP weights for target layer
                        try:
                            # Paths might be different depending on model architecture
                            possible_paths = [
                                f"base_model.blocks.{target_layer}.mlp.W_out.weight[{target_neuron}]",
                                f"base_model.transformer.h.{target_layer}.mlp.c_proj.weight[{target_neuron}]"
                            ]
                            
                            for path in possible_paths:
                                parts = path.split('.')
                                current = self.predictor
                                for part in parts[:-1]:
                                    if '[' in part:  # Handle indexing
                                        part_name, idx = part.split('[')
                                        idx = int(idx.replace(']', ''))
                                        current = getattr(current, part_name)[idx]
                                    else:
                                        current = getattr(current, part) if hasattr(current, part) else None
                                        if current is None:
                                            break
                                
                                if current is not None:
                                    final_part = parts[-1]
                                    if '[' in final_part:
                                        part_name, idx = final_part.split('[')
                                        idx = int(idx.replace(']', ''))
                                        param = getattr(current, part_name)[idx]
                                    else:
                                        param = getattr(current, final_part)
                                    
                                    if hasattr(param, 'grad') and param.grad is not None:
                                        logger.info(f"DEBUG - Target neuron param '{path}' grad: {param.grad.abs().sum().item()}")
                                    else:
                                        logger.info(f"DEBUG - Target neuron param '{path}' has no gradient")
                        except Exception as e:
                            logger.info(f"DEBUG - Error accessing target neuron parameters: {e}")

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
            
            # Track activations if enabled
            if track_activations and self.activation_monitor is not None:
                self.activation_monitor.collect_activations(epoch=epoch+1)

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
                    scheduler.step(val_loss if eval_during_training else train_loss)
                else:
                    scheduler.step()

        # End of training
        end_time = time.time()
        training_time = end_time - start_time
        self.metrics["train_time"] = training_time
        logger.info(f"Training completed in {training_time:.1f} seconds")

        # Cleanup gradient tracking if used
        if track_gradients and self.gradient_tracker is not None:
            self.gradient_tracker.remove_hooks()
            
            # Generate gradient visualizations
            gradient_viz_dir = os.path.join(self.output_dir, "gradient_analysis")
            os.makedirs(gradient_viz_dir, exist_ok=True)
            
            gradient_results = self.gradient_tracker.generate_gradient_plots(gradient_viz_dir)
            self.metrics["gradient_analysis"] = gradient_results
            logger.info(f"Gradient analysis saved to {gradient_viz_dir}")
        
        # Generate activation distribution visualizations if used
        if track_activations and self.activation_monitor is not None:
            activation_viz_dir = os.path.join(self.output_dir, "activation_analysis")
            os.makedirs(activation_viz_dir, exist_ok=True)
            
            activation_results = self.activation_monitor.generate_activation_plots(activation_viz_dir)
            self.metrics["activation_analysis"] = activation_results
            logger.info(f"Activation analysis saved to {activation_viz_dir}")

        # Save final model
        self._save_model("final_model")

        # Save training metrics
        self._save_metrics()

        # Generate visualizations
        self._generate_visualizations()

        # Evaluate on test set
        test_loss, test_metrics = self._evaluate_on_test()
        logger.info(f"Test Loss: {test_loss:.4f}, {self._format_metrics(test_metrics, 'Test')}")
        
        # DEBUG: Log final test metrics with explanation
        logger.info(f"DEBUG - Final test metrics details:")
        for key, value in test_metrics.items():
            logger.info(f"  - {key}: {value}")
            
        # DEBUG: If regression, explain potential normalization issues
        if hasattr(self.predictor, 'head_type') and self.predictor.head_type == 'regression':
            if hasattr(self.predictor, 'activation_mean') and hasattr(self.predictor, 'activation_std'):
                logger.info(f"DEBUG - Normalization params used in prediction:")
                logger.info(f"  - activation_mean: {self.predictor.activation_mean}")
                logger.info(f"  - activation_std: {self.predictor.activation_std}")
                
                # Explain how normalization affects MSE calculation
                logger.info(f"NOTE: The MSE value in test metrics is raw (unnormalized). To compare with final report MSE:")
                logger.info(f"  - Raw MSE: {test_metrics.get('mse', 'N/A')}")
                logger.info(f"  - MSE after normalization: {test_metrics.get('mse', 0)/(self.predictor.activation_std**2) if self.predictor.activation_std else 'N/A'}")

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
        if self.predictor.head_type == "classification" or self.predictor.head_type == "token":
            # For classification and token prediction, use CrossEntropyLoss
            if self.predictor.head_type == "classification" and hasattr(self.predictor.head, "class_weights") and self.predictor.head.class_weights is not None:
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

        # DEBUG: Log beginning of evaluation
        # logger.info(f"DEBUG - Starting evaluation on {split_name} set with {len(data_loader)} batches")

        # first_batch_logged = False
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.predictor(input_ids, attention_mask)
                
                # Handle potential dimension mismatch for token prediction evaluation
                if self.predictor.head_type == "token":
                    # First check if labels are 1D (class indices)
                    if len(labels.shape) == 1:
                        # Labels are already in class indices format
                        # Map to 0-9 range if needed
                        # Just ensure labels are in the 0-9 range (should already be handled in dataset)
                        if torch.max(labels) > 9 or torch.min(labels) < 0:
                            logger.warning(f"WARNING: Evaluation - token prediction labels outside 0-9 range: [{torch.min(labels).item()}, {torch.max(labels).item()}]")
                            labels = torch.clamp(labels, 0, 9)
                    # For 2D labels (one-hot or multi-class)
                    elif len(labels.shape) > 1:
                        # Check if dimensions don't match
                        if outputs.shape[1] != labels.shape[1]:
                            # Convert one-hot to indices if needed
                            if labels.shape[1] > 1:
                                labels = torch.argmax(labels, dim=1)
                            
                            # Map labels to 0-9 range if needed
                            if torch.max(labels) >= 10:
                                max_label = torch.max(labels).item()
                                labels = torch.floor((labels.float() / max_label) * 9).long()
                
                loss = loss_fn(outputs, labels)
                
                # DEBUG: Log first batch outputs
                # if not first_batch_logged:
                #     logger.info(f"DEBUG - {split_name} evaluation, first batch:")
                #     logger.info(f"  Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
                #     logger.info(f"  First 3 outputs: {outputs.cpu().numpy()[:3]}")
                #     logger.info(f"  First 3 labels: {labels.cpu().numpy()[:3]}")
                #     logger.info(f"  Batch loss: {loss.item()}")
                #     first_batch_logged = True

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
                if hasattr(self.predictor, 'target_layer') and hasattr(self.predictor, 'target_neuron'):
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
                    elif self.predictor.head_type == "token":
                        # For token prediction, convert the multi-class logits to a continuous value
                        # First convert to probabilities
                        probs = torch.softmax(outputs, dim=1)
                        
                        # Then use weighted average with digit values (0-9)
                        digit_values = np.arange(10)
                        pred = (probs.cpu().numpy() * digit_values.reshape(1, -1)).sum(axis=1)[0]
                        
                        logger.info(f"Token prediction value: {pred:.4f}")
                        
                        # If we have activation statistics, we can map to the original activation range
                        if hasattr(self.predictor, 'activation_mean') and hasattr(self.predictor, 'activation_std'):
                            # Assuming the token prediction maps from [0-9] to normalized activation space
                            min_act = self.predictor.activation_mean - 2 * self.predictor.activation_std
                            max_act = self.predictor.activation_mean + 2 * self.predictor.activation_std
                            act_range = max_act - min_act
                            
                            # Map from [0-9] to activation range
                            pred = min_act + (pred / 9.0) * act_range
                            logger.info(f"Mapped to activation range: {pred:.4f}")
                    else:
                        # For regression predictions, outputs should be a scalar
                        if isinstance(outputs, torch.Tensor) and outputs.numel() == 1:
                            pred = outputs.item()
                        else:
                            # Handle unexpected format gracefully
                            logger.warning(f"Unexpected output format: {outputs.shape if isinstance(outputs, torch.Tensor) else type(outputs)}")
                            # Use the first element if tensor or convert to float if needed
                            pred = outputs[0] if isinstance(outputs, (list, np.ndarray)) or (isinstance(outputs, torch.Tensor) and outputs.numel() > 1) else float(outputs)
                    
                    # Get the raw activation value
                    raw_activation = activations.item()
                    
                    # For proper comparison with regression predictions, we need to normalize the activations
                    # using the same parameters that were used during training
                    if self.predictor.head_type == "regression" and self.predictor.activation_mean is not None and self.predictor.activation_std is not None:
                        # Either normalize the actual or denormalize the prediction
                        # Normalizing the actual is more consistent with model's internal logic
                        normalized_activation = (raw_activation - self.predictor.activation_mean) / self.predictor.activation_std
                        
                        # Now we have two options:
                        # 1. Use normalized activation and raw prediction (before denormalization)
                        # 2. Use raw activation and denormalized prediction
                        
                        # Option 1: For better understanding of model's internal working
                        # predictions.append(outputs.item())  # Raw model output
                        # actuals.append(normalized_activation)  # Normalized ground truth
                        
                        # Option 2: For interpretability in original activation space
                        predictions.append(pred)  # Denormalized prediction
                        actuals.append(raw_activation)  # Raw ground truth
                        
                        # Log normalization adjustment for transparency
                        logger.debug(f"Activation: raw={raw_activation:.4f}, normalized={normalized_activation:.4f}")
                        logger.debug(f"Prediction: raw={outputs.item():.4f}, denormalized={pred:.4f}")
                    else:
                        # For classification head, use values as they are
                        predictions.append(pred)
                        actuals.append(raw_activation)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Convert to numpy arrays for analysis
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        
        # Calculate error statistics for informational purposes only
        mean_error = np.mean(predictions_array - actuals_array)
        error_std = np.std(predictions_array - actuals_array)
        logger.info(f"Prediction error statistics - Mean: {mean_error:.4f}, StdDev: {error_std:.4f}")
        
        # Simple scatter plot without any bias correction
        plt.scatter(actuals_array, predictions_array, alpha=0.6, label="Predictions")
        
        # Add perfect prediction line
        min_val = min(min(actuals_array), min(predictions_array))
        max_val = max(max(actuals_array), max(predictions_array))
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
        plt.legend()
        
        # Calculate correlation and error metrics
        correlation = np.corrcoef(actuals_array, predictions_array)[0, 1]
        mse = np.mean((predictions_array - actuals_array) ** 2)
        mae = np.mean(np.abs(predictions_array - actuals_array))
        
        # No corrected predictions anymore - we're using a single consistent approach
        
        # Add metrics as text
        plt.title(f"Predictions vs. Actual Activations ({split} split)")
        plt.xlabel("Actual Activations")
        plt.ylabel("Predicted Activations")
        
        # Create metrics text
        metrics_text = f"Metrics:\n  Correlation: {correlation:.4f}\n  MSE: {mse:.4f}\n  MAE: {mae:.4f}\n  Mean Error: {mean_error:.4f}"
            
        # Log metrics calculation details
        logger.info(f"Visualization metrics calculation:")
        logger.info(f"  - Correlation: {correlation:.6f}")
        logger.info(f"  - MSE: {mse:.6f}")
        logger.info(f"  - MAE: {mae:.6f}")
        logger.info(f"  - Mean error: {mean_error:.6f}")
            
        # Show examples of actual vs predicted values
        logger.info(f"Sample actual vs predicted values:")
        for i in range(min(5, len(actuals_array))):
            logger.info(f"  #{i}: Actual={actuals_array[i]:.4f}, Predicted={predictions_array[i]:.4f}, Error={predictions_array[i]-actuals_array[i]:.4f}")
        
        plt.text(
            0.05, 0.95,
            metrics_text,
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
        
        # Log metrics
        logger.info(f"Saved predictions visualization to {output_file}")
        logger.info(f"Metrics - Correlation: {correlation:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        
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
        
        # Build the result dictionary
        result = {
            "correlation": correlation,
            "mse": mse,
            "mae": mae,
            "mean_error": mean_error,
            "predictions": predictions,
            "actuals": actuals,
            "texts": texts,
            "visualization_file": output_file,
            "samples_file": samples_file,
        }
            
        return result

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
            # Test new monitoring capabilities
            track_gradients=True,
            track_activations=True,
            # Test unfreezing
            unfreeze_strategy="after_target",
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