# tests/test_trainer.py
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from unittest.mock import patch, MagicMock, PropertyMock

from architecture import ActivationPredictor
from trainer import PredictorTrainer
from .conftest import small_test_model, device # Import fixtures

# Minimal config for trainer tests
@pytest.fixture
def dummy_config(tmp_path): # tmp_path is a pytest fixture for temporary directory
    return {
        "learning_rate": 1e-3,
        "epochs": 3,
        "batch_size": 2,
        "output_type": "regression",
        "unfreeze_strategy": "head_only",
        "feature_extraction_hook_point": "blocks.1.hook_resid_post", # For small_test_model (2 layers)
        "target_token_position_for_features": "last",
        "use_wandb": False,
        "early_stopping_patience": 2,
        "output_dir": str(tmp_path / "trainer_output") # Use temp dir for test outputs
    }

@pytest.fixture
def dummy_predictor_model(small_test_model, device):
    # Regression model for simplicity in most tests
    return ActivationPredictor(
        base_model=small_test_model,
        prediction_head_type="regression",
        base_model_output_dim=small_test_model.cfg.d_model,
        device=device
    )

@pytest.fixture
def dummy_datasets(device):
    # input_ids (batch, seq_len), targets (batch) for regression
    train_inputs = torch.randint(0, 50, (10, 5), device=device) # 10 samples, seq_len 5
    train_targets = torch.randn(10, device=device)
    train_ds = TensorDataset(train_inputs, train_targets)

    val_inputs = torch.randint(0, 50, (6, 5), device=device)
    val_targets = torch.randn(6, device=device)
    val_ds = TensorDataset(val_inputs, val_targets)
    return train_ds, val_ds

class TestPredictorTrainer:

    def test_trainer_initialization(self, dummy_predictor_model, dummy_config, dummy_datasets, device):
        train_ds, val_ds = dummy_datasets
        trainer = PredictorTrainer(
            model=dummy_predictor_model,
            config=dummy_config,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=device
        )
        assert trainer.model == dummy_predictor_model
        assert trainer.device == device
        assert isinstance(trainer.loss_fn, nn.MSELoss) # Based on dummy_config
        assert trainer.epochs == dummy_config["epochs"]

    def test_get_loss_function(self, dummy_predictor_model, dummy_config, dummy_datasets, device):
        train_ds, _ = dummy_datasets
        
        # Regression
        cfg_reg = dummy_config.copy()
        cfg_reg["output_type"] = "regression"
        trainer_reg = PredictorTrainer(dummy_predictor_model, cfg_reg, train_ds, device=device)
        assert isinstance(trainer_reg.loss_fn, nn.MSELoss)

        # Classification
        cfg_class = dummy_config.copy()
        cfg_class["output_type"] = "classification"
        # Need a model with classification head for this to be fully valid, but testing loss fn selection
        trainer_class = PredictorTrainer(dummy_predictor_model, cfg_class, train_ds, device=device)
        assert isinstance(trainer_class.loss_fn, nn.CrossEntropyLoss)
        
        # Token (binary/digit)
        cfg_token = dummy_config.copy()
        cfg_token["output_type"] = "token_binary"
        trainer_token = PredictorTrainer(dummy_predictor_model, cfg_token, train_ds, device=device)
        assert isinstance(trainer_token.loss_fn, nn.CrossEntropyLoss)

    @patch('trainer.optim.Adam')
    def test_apply_unfreeze_strategy(self, mock_adam, dummy_predictor_model, dummy_config, dummy_datasets, device):
        train_ds, _ = dummy_datasets
        
        # Head only
        cfg = dummy_config.copy()
        cfg["unfreeze_strategy"] = "head_only"
        with patch.object(dummy_predictor_model, 'freeze_base_model') as mock_freeze:
            trainer = PredictorTrainer(dummy_predictor_model, cfg, train_ds, device=device)
            mock_freeze.assert_called_once_with(True)
            # Check optimizer was called with only head params (approximate check)
            # This is hard to check precisely without inspecting params passed to Adam
            mock_adam.assert_called() 

        # All layers
        cfg["unfreeze_strategy"] = "all_layers"
        with patch.object(dummy_predictor_model, 'freeze_base_model') as mock_freeze:
            trainer = PredictorTrainer(dummy_predictor_model, cfg, train_ds, device=device)
            mock_freeze.assert_called_once_with(False)
            mock_adam.assert_called()

    @patch.object(ActivationPredictor, 'forward')
    def test_train_epoch(self, mock_model_forward, dummy_predictor_model, dummy_config, dummy_datasets, device):
        train_ds, _ = dummy_datasets
        trainer = PredictorTrainer(dummy_predictor_model, dummy_config, train_ds, device=device)
        train_loader = DataLoader(train_ds, batch_size=trainer.batch_size)

        # Mock model output
        # For regression, output shape [batch_size, 1]
        # Ensure the mock output requires grad, as if it came from a trainable layer
        mock_output = torch.randn(trainer.batch_size, 1, device=device, requires_grad=True)
        mock_model_forward.return_value = mock_output
        
        avg_loss = trainer.train_epoch(train_loader)
        assert isinstance(avg_loss, float)
        assert mock_model_forward.call_count == len(train_loader)

    @patch.object(ActivationPredictor, 'forward')
    def test_evaluate(self, mock_model_forward, dummy_predictor_model, dummy_config, dummy_datasets, device):
        train_ds, val_ds = dummy_datasets
        trainer = PredictorTrainer(dummy_predictor_model, dummy_config, train_ds, val_ds, device=device)
        val_loader = DataLoader(val_ds, batch_size=trainer.batch_size)

        mock_model_forward.return_value = torch.randn(trainer.batch_size, 1, device=device)
        
        metrics = trainer.evaluate(val_loader)
        assert "val_loss" in metrics
        assert isinstance(metrics["val_loss"], float)
        assert mock_model_forward.call_count == len(val_loader)

    @patch('trainer.PredictorTrainer.train_epoch')
    @patch('trainer.PredictorTrainer.evaluate')
    @patch('trainer.PredictorTrainer.generate_training_visualizations')
    @patch('trainer.wandb', MagicMock()) # Mock wandb module itself if WANDB_AVAILABLE is True
    def test_train_loop_logic(self, mock_viz, mock_eval, mock_train_epoch, 
                              dummy_predictor_model, dummy_config, dummy_datasets, device):
        train_ds, val_ds = dummy_datasets
        
        cfg = dummy_config.copy()
        cfg["epochs"] = 5
        cfg["early_stopping_patience"] = 2 # Stop after 2 epochs of no improvement
        cfg["use_wandb"] = False # Test without W&B for simplicity here

        trainer = PredictorTrainer(dummy_predictor_model, cfg, train_ds, val_ds, device=device)

        # Simulate train_epoch and evaluate behavior
        mock_train_epoch.return_value = 0.5 # Constant train loss
        
        # Simulate validation loss improving then worsening for early stopping
        mock_eval.side_effect = [
            {"val_loss": 0.4}, # Epoch 1 (improve)
            {"val_loss": 0.3}, # Epoch 2 (improve)
            {"val_loss": 0.35},# Epoch 3 (no improve)
            {"val_loss": 0.36},# Epoch 4 (no improve - stop)
            {"val_loss": 0.2} # Should not be reached
        ]
        
        history = trainer.train()
        
        assert mock_train_epoch.call_count == 4 # Stops after 4th epoch
        assert mock_eval.call_count == 4
        assert len(history["train_loss"]) == 4
        assert len(history["val_loss"]) == 4
        mock_viz.assert_called_once_with(history)

    @patch('matplotlib.pyplot.savefig')
    def test_generate_training_visualizations(self, mock_savefig, dummy_predictor_model, dummy_config, dummy_datasets, device):
        train_ds, val_ds = dummy_datasets
        trainer = PredictorTrainer(dummy_predictor_model, dummy_config, train_ds, val_ds, device=device)
        
        history = {"train_loss": [0.5, 0.4, 0.3], "val_loss": [0.6, 0.5, 0.45]}
        trainer.generate_training_visualizations(history)
        mock_savefig.assert_called_once()
        # Check if the path contains the output_dir from config
        args, _ = mock_savefig.call_args
        save_path = args[0]
        assert dummy_config["output_dir"] in save_path
        assert "training_loss_history.png" in save_path

    # TODO: Test W&B logging if use_wandb is True (mock wandb.log, wandb.init, wandb.Image)
    # TODO: Test different unfreeze strategies more thoroughly by checking param.requires_grad
    # TODO: Test target tensor transformations in train_epoch for different output_types
