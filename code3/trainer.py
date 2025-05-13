# trainer.py
# PredictorTrainer: For training with monitoring capabilities

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Union, List, Callable

# Try to import wandb for logging, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Assuming ActivationPredictor is in architecture.py
from architecture import ActivationPredictor 

class PredictorTrainer:
    """
    Handles the training process for the ActivationPredictor model.
    """
    def __init__(self,
                 model: ActivationPredictor,
                 config: Dict[str, Any], # Training configuration
                 train_dataset: TensorDataset,
                 val_dataset: Optional[TensorDataset] = None,
                 device: Optional[str] = None):
        """
        Initializes the PredictorTrainer.

        Args:
            model: The ActivationPredictor model instance.
            config: A dictionary containing training parameters like lr, batch_size, epochs, etc.
                    Also includes model-specific params like 'output_type', 'unfreeze_strategy'.
            train_dataset: PyTorch TensorDataset for training.
            val_dataset: Optional PyTorch TensorDataset for validation.
            device: The device to run training on.
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        if device:
            self.device = device
        elif hasattr(model, 'device'):
            self.device = model.device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('learning_rate', 1e-3))
        self.loss_fn = self._get_loss_function()

        self.epochs = self.config.get('epochs', 10)
        self.batch_size = self.config.get('batch_size', 32)

        # Setup Weights & Biases if available and configured
        if WANDB_AVAILABLE and self.config.get("use_wandb", False):
            try:
                wandb.init(project=self.config.get("wandb_project", "neuron-self-report"), 
                           config=self.config,
                           name=self.config.get("wandb_run_name", None))
                self.use_wandb = True
                print("Weights & Biases initialized.")
            except Exception as e:
                print(f"Failed to initialize Weights & Biases: {e}. Proceeding without W&B.")
                self.use_wandb = False
        else:
            self.use_wandb = False
            if self.config.get("use_wandb", False) and not WANDB_AVAILABLE:
                print("Warning: W&B logging requested but wandb library not found. Install with 'pip install wandb'.")

        self._apply_unfreeze_strategy()

    def _get_loss_function(self) -> Callable:
        """Determines the loss function based on the prediction head type."""
        output_type = self.config.get('output_type', 'regression')
        if output_type == 'regression':
            return nn.MSELoss()
        elif output_type == 'classification':
            return nn.CrossEntropyLoss()
        elif output_type in ['token_binary', 'token_digit']:
            # For token prediction, we predict logits and use CrossEntropyLoss.
            # The target will be the ID of the 'on'/'off' token or '0'-'9' token.
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported output_type for loss function: {output_type}")

    def _apply_unfreeze_strategy(self):
        """Applies the model unfreezing strategy based on config."""
        strategy = self.config.get('unfreeze_strategy', 'head_only')
        target_layer = self.config.get('neuron_layer', None) # Needed for 'layers_after_target'

        if strategy == 'head_only':
            self.model.freeze_base_model(True)
            # Ensure head parameters are trainable (should be by default if newly created)
            if self.model.head is not None:
                for param in self.model.head.parameters():
                    param.requires_grad = True
        elif strategy == 'all_layers':
            self.model.freeze_base_model(False) # Unfreeze all base model params
        elif strategy == 'layers_after_target':
            if target_layer is None:
                raise ValueError("neuron_layer must be specified in config for 'layers_after_target' unfreeze strategy.")
            self.model.unfreeze_layers_after_target(target_layer)
        elif strategy == 'selective_components':
            components = self.config.get('components_to_unfreeze', [])
            if not components:
                print("Warning: 'selective_components' strategy chosen but no components specified. Defaulting to head_only.")
                self.model.freeze_base_model(True) # Fallback to head_only
            else:
                self.model.unfreeze_selective_components(components)
        else:
            raise ValueError(f"Unsupported unfreeze_strategy: {strategy}")
        
        # Re-initialize optimizer with only trainable parameters
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                    lr=self.config.get('learning_rate', 1e-3))
        print(f"Applied unfreeze strategy: {strategy}. Optimizer updated with trainable parameters.")


    def train_epoch(self, train_loader: DataLoader) -> float:
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training Epoch")):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass arguments depend on model's forward signature
            # Assuming ActivationPredictor's forward takes input_ids, feature_extraction_hook_point, target_token_position
            # These need to be part of the config or handled appropriately.
            # For now, assume inputs are directly usable as input_ids.
            # The hook point and token position for feature extraction are model architecture choices,
            # not typically part of the batch data.
            
            predictions = self.model(
                input_ids=inputs, # Assuming inputs from dataloader are input_ids
                feature_extraction_hook_point=self.config.get('feature_extraction_hook_point', None),
                target_token_position=self.config.get('target_token_position_for_features', 'last') 
            )
            
            # Adjust targets for token-based prediction if necessary
            # E.g., if targets are strings 'on'/'off', map them to token IDs.
            # This mapping should be pre-calculated and stored, or targets in dataset should be IDs.
            # For simplicity, assume targets are already in the correct format for the loss function.
            # For CrossEntropyLoss with token logits, targets should be class indices (token IDs).
            # For MSELoss, targets should be continuous values.
            
            # If output_type is classification or token-based, targets might need to be LongTensor
            if self.config.get('output_type') in ['classification', 'token_binary', 'token_digit']:
                targets = targets.long() # Ensure targets are LongTensor for CrossEntropyLoss
            else: # Regression
                targets = targets.float().unsqueeze(-1) # Ensure targets are Float and match pred shape [batch, 1]


            loss = self.loss_fn(predictions, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if self.use_wandb and batch_idx % self.config.get("wandb_log_freq_batch", 50) == 0:
                wandb.log({"batch_train_loss": loss.item()})
                
        return total_loss / len(train_loader)

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluates the model on the validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Evaluating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                predictions = self.model(
                    input_ids=inputs,
                    feature_extraction_hook_point=self.config.get('feature_extraction_hook_point', None),
                    target_token_position=self.config.get('target_token_position_for_features', 'last')
                )

                if self.config.get('output_type') in ['classification', 'token_binary', 'token_digit']:
                    targets = targets.long()
                else: # Regression
                    targets = targets.float().unsqueeze(-1)

                loss = self.loss_fn(predictions, targets)
                total_loss += loss.item()
                
                all_preds.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / len(val_loader)
        metrics = {"val_loss": avg_loss}
        
        # TODO: Add more metrics based on output_type (e.g., accuracy for classification, MSE/corr for regression)
        # all_preds_cat = torch.cat(all_preds)
        # all_targets_cat = torch.cat(all_targets)
        # if self.config.get('output_type') == 'regression':
        #    mse = ((all_preds_cat - all_targets_cat)**2).mean().item()
        #    metrics['val_mse'] = mse
        # elif self.config.get('output_type') == 'classification':
        #    accuracy = (all_preds_cat.argmax(dim=1) == all_targets_cat).float().mean().item()
        #    metrics['val_accuracy'] = accuracy
            
        return metrics

    def train(self):
        """Main training loop."""
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False) if self.val_dataset else None
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stopping_patience = self.config.get('early_stopping_patience', 3)

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            avg_train_loss = self.train_epoch(train_loader)
            print(f"Average Training Loss: {avg_train_loss:.4f}")
            
            log_dict = {"epoch": epoch + 1, "avg_train_loss": avg_train_loss}
            
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                print(f"Validation Metrics: {val_metrics}")
                log_dict.update(val_metrics)
                
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    epochs_no_improve = 0
                    # TODO: Save best model checkpoint
                    # torch.save(self.model.state_dict(), "best_model.pt")
                    # print("Saved best model checkpoint.")
                else:
                    epochs_no_improve += 1
            
            if self.use_wandb:
                wandb.log(log_dict)

            # TODO: Implement activation distribution monitoring and visualization generation
            # This could be done at the end of each epoch or less frequently.
            # Example: self.monitor_activation_distributions()
            # Example: self.generate_training_visualizations()

            if val_loader and epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {early_stopping_patience} epochs with no improvement.")
                break
        
        if self.use_wandb:
            wandb.finish()
        
        print("Training finished.")

    # Placeholder for monitoring and visualization methods
    def monitor_activation_distributions(self):
        # This would involve running some sample data through the model (or parts of it)
        # and logging statistics about internal activations, especially if parts of base model are unfrozen.
        print("Placeholder: Monitoring activation distributions...")
        pass

    def generate_training_visualizations(self, history: Dict[str, List[float]]):
        # Generate plots for training/validation loss, other metrics over epochs.
        # history might be a dict like {'train_loss': [...], 'val_loss': [...]}
        print("Placeholder: Generating training visualizations...")
        # Example:
        # plt.figure(figsize=(10,5))
        # plt.plot(history['train_loss'], label='Train Loss')
        # if 'val_loss' in history:
        #     plt.plot(history['val_loss'], label='Validation Loss')
        # plt.title('Training History')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # if self.use_wandb:
        #     wandb.log({"training_history_plot": wandb.Image(plt)})
        # plt.show() # or savefig
        pass

# Example Usage (conceptual, real usage from main pipeline)
if __name__ == '__main__':
    from transformer_lens import HookedTransformer # Import for example usage
    print("PredictorTrainer example usage (conceptual setup):")
    # This requires a fully set up ActivationPredictor model and datasets.
    # For a runnable example, we'd need:
    # 1. A HookedTransformer instance
    # 2. An ActivationPredictor instance (e.g., for regression)
    # 3. Dummy TensorDatasets for train and val
    # 4. A configuration dictionary

    # --- Dummy Model Setup ---
    if torch.backends.mps.is_available():
        dummy_device = "mps"
    elif torch.cuda.is_available():
        dummy_device = "cuda"
    else:
        dummy_device = "cpu"
    
    try:
        dummy_base_model = HookedTransformer.from_pretrained("gpt2-small", device=dummy_device) # Small model
        dummy_predictor_model = ActivationPredictor(
            base_model=dummy_base_model,
            prediction_head_type="regression",
            base_model_output_dim=dummy_base_model.cfg.d_model,
            device=dummy_device
        )

        # --- Dummy Data Setup ---
        # (batch_size, seq_len) for inputs, (batch_size) for targets (regression)
        # For real use, input_ids come from tokenized text, targets from ActivationDatasetGenerator
        dummy_train_inputs = torch.randint(0, dummy_base_model.cfg.d_vocab, (100, 10)) 
        dummy_train_targets = torch.randn(100) 
        dummy_train_dataset = TensorDataset(dummy_train_inputs, dummy_train_targets)

        dummy_val_inputs = torch.randint(0, dummy_base_model.cfg.d_vocab, (50, 10))
        dummy_val_targets = torch.randn(50)
        dummy_val_dataset = TensorDataset(dummy_val_inputs, dummy_val_targets)

        # --- Dummy Config ---
        dummy_config = {
            "learning_rate": 1e-4,
            "epochs": 3, # Short for example
            "batch_size": 16,
            "output_type": "regression", # Matches dummy_predictor_model
            "unfreeze_strategy": "head_only", # Train only the new head
            "feature_extraction_hook_point": f"blocks.{dummy_base_model.cfg.n_layers - 1}.hook_resid_post", # Example
            "target_token_position_for_features": "last",
            "use_wandb": False, # Set to True and configure if you want to test W&B
            # "wandb_project": "my-test-project", 
            # "wandb_run_name": "trainer_example_run"
        }

        print(f"Setting up trainer with device: {dummy_device}")
        trainer = PredictorTrainer(model=dummy_predictor_model,
                                   config=dummy_config,
                                   train_dataset=dummy_train_dataset,
                                   val_dataset=dummy_val_dataset,
                                   device=dummy_device)
        
        print("Starting dummy training loop...")
        trainer.train()
        print("Dummy training example finished.")

    except Exception as e:
        print(f"Error in PredictorTrainer example: {e}")
        import traceback
        traceback.print_exc()
        print("This example requires TransformerLens and a model like gpt2-small to be downloadable.")
