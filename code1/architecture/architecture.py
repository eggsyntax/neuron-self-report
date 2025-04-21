# architecture/architecture.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Tuple, Union, Optional, Any
from transformers import PreTrainedTokenizerBase
from transformer_lens import HookedTransformer
import numpy as np
import os
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ActivationHead(nn.Module):
    """Base class for activation prediction heads"""
    
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.1,
    ):
        """
        Base activation prediction head.
        
        Args:
            input_dim: Dimension of input features
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_config(self) -> Dict:
        """Return configuration for serialization"""
        return {
            "class_name": self.__class__.__name__,
            "input_dim": self.input_dim,
            "dropout": self.dropout.p,
        }

class TokenPredictionHead(ActivationHead):
    """Head that uses the model's token logits for activation prediction"""
    
    def __init__(
        self,
        input_dim: int,
        digit_tokens: List[int],
        dropout: float = 0.1,
    ):
        """
        Initialize token prediction head.
        
        Args:
            input_dim: Dimension of input features (not directly used for token prediction)
            digit_tokens: List of token IDs for the digits 0-9 to use for prediction
            dropout: Dropout probability (not directly used for token prediction)
        """
        super().__init__(input_dim, dropout)
        
        self.digit_tokens = digit_tokens
        self.has_hidden = False  # No hidden layer needed since we use model's logits
        
    def forward(self, x):
        """
        Forward pass is a no-op for token prediction head.
        The actual prediction happens in ActivationPredictor.forward() for this head type.
        """
        # Token prediction head doesn't transform the features
        # It's handled specially in the ActivationPredictor's forward method
        return x
    
    def get_config(self) -> Dict:
        """Return configuration for serialization"""
        config = super().get_config()
        config.update({
            "digit_tokens": self.digit_tokens,
        })
        return config


class RegressionHead(ActivationHead):
    """Head for predicting continuous activation values"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        output_scale: Optional[float] = None,
    ):
        """
        Initialize regression head for continuous activation prediction.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer (if None, no hidden layer)
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', or 'silu')
            output_scale: Optional scaling factor for output normalization
        """
        super().__init__(input_dim, dropout)
        
        # Set hidden dimension
        self.has_hidden = hidden_dim is not None
        if hidden_dim is None:
            hidden_dim = input_dim
            
        # Set activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        # Output scaling (for normalization)
        self.output_scale = output_scale
            
        # Create layers
        if self.has_hidden:
            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, 1)
        else:
            self.output = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._init_weights()
            
    def _init_weights(self):
        """Initialize weights using standard approach"""
        if self.has_hidden:
            nn.init.normal_(self.hidden.weight, std=0.02)
            nn.init.zeros_(self.hidden.bias)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)
        
    def forward(self, x):
        """Forward pass for regression"""
        # Apply dropout
        x = self.dropout(x)
        
        # Apply hidden layer if present
        if self.has_hidden:
            x = self.hidden(x)
            x = self.activation(x)
            x = self.dropout(x)
            
        # Output layer
        x = self.output(x)
        
        # Apply output scaling if specified
        if self.output_scale is not None:
            x = x * self.output_scale
            
        return x.squeeze(-1)  # Remove last dimension
        
        # NOTE: We're not doing any normalization here. The model should be
        # trained to output values in the right scale. Any normalization
        # should be done consistently at the dataset level only.
    
    def get_config(self) -> Dict:
        """Return configuration for serialization"""
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden.out_features if self.has_hidden else None,
            "activation": self.activation.__name__,
            "output_scale": self.output_scale,
        })
        return config

class ClassificationHead(ActivationHead):
    """Head for predicting discretized activation values"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize classification head for discretized activation prediction.
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
            hidden_dim: Dimension of hidden layer (if None, no hidden layer)
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', or 'silu')
            class_weights: Optional weights for imbalanced classes
        """
        super().__init__(input_dim, dropout)
        
        # Set parameters
        self.num_classes = num_classes
        self.has_hidden = hidden_dim is not None
        if hidden_dim is None:
            hidden_dim = input_dim
            
        # Set activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        # Store class weights
        self.register_buffer("class_weights", class_weights)
            
        # Create layers
        if self.has_hidden:
            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, num_classes)
        else:
            self.output = nn.Linear(input_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
            
    def _init_weights(self):
        """Initialize weights using standard approach"""
        if self.has_hidden:
            nn.init.normal_(self.hidden.weight, std=0.02)
            nn.init.zeros_(self.hidden.bias)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)
        
    def forward(self, x):
        """Forward pass for classification"""
        # Apply dropout
        x = self.dropout(x)
        
        # Apply hidden layer if present
        if self.has_hidden:
            x = self.hidden(x)
            x = self.activation(x)
            x = self.dropout(x)
            
        # Output layer
        logits = self.output(x)
        return logits
    
    def get_config(self) -> Dict:
        """Return configuration for serialization"""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden.out_features if self.has_hidden else None,
            "activation": self.activation.__name__,
            "class_weights": self.class_weights.cpu().numpy().tolist() if self.class_weights is not None else None,
        })
        return config

class ActivationPredictor(nn.Module):
    """Model for predicting internal transformer activation values
    
    This model uses a transformer's own representations to predict specific
    internal activation values, enabling a form of introspection/self-reporting.
    During training, it predicts values; once trained, it effectively "reports" 
    its own internal states with quantifiable confidence.
    """
    
    def __init__(
        self,
        base_model: HookedTransformer,
        head_type: str = "classification",
        num_classes: Optional[int] = None,
        target_layer: Optional[int] = None,
        target_neuron: Optional[int] = None,
        layer_type: str = "mlp_out",
        token_pos: Union[int, str] = "last",
        feature_layer: int = -1,
        head_config: Optional[Dict] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        device: str = "mps",
        bin_edges: Optional[np.ndarray] = None,
        activation_mean: Optional[float] = None,
        activation_std: Optional[float] = None,
    ):
        """
        Initialize activation prediction/reporting model.
        
        Args:
            base_model: TransformerLens model to use as base
            head_type: Type of prediction head ('classification' or 'regression')
            num_classes: Number of classes for classification head
            target_layer: Layer to predict activations for
            target_neuron: Neuron index to predict activations for
            layer_type: Type of layer for activation access
            token_pos: Token position to predict ('last' or specific index)
            feature_layer: Layer to extract features from (-1 for last layer)
            head_config: Configuration for prediction head
            tokenizer: Tokenizer for the model (if None, use base_model.tokenizer)
            device: Device to run on
            bin_edges: For classification, the bin edges for converting from 
                       continuous predictions to class probabilities
            activation_mean: For regression, mean activation for normalization
            activation_std: For regression, std deviation for normalization
        """
        super().__init__()
        
        # Store parameters
        self.base_model = base_model
        self.target_layer = target_layer
        self.target_neuron = target_neuron
        self.layer_type = layer_type
        self.token_pos = token_pos
        self.head_type = head_type
        self.feature_layer = feature_layer
        self.device = device
        self.bin_edges = bin_edges
        self.activation_mean = activation_mean
        self.activation_std = activation_std
        
        # Get tokenizer from base model if not provided
        self.tokenizer = tokenizer if tokenizer is not None else base_model.tokenizer
        
        # Move base model to device and set to eval mode
        self.base_model.to(device)
        self.base_model.eval()
        
        # Enable caching for activation access
        self.base_model.use_cache_hook = True
        
        # Get model dimensions
        self.d_model = base_model.cfg.d_model
        
        # Create head based on type
        head_config = head_config or {}
        # Create a copy to avoid modifying the original
        head_config_copy = head_config.copy()
        
        # Only remove input_dim to avoid collision
        if 'input_dim' in head_config_copy:
            del head_config_copy['input_dim']
            
        if head_type == "classification":
            # Handle num_classes parameter
            if num_classes is None and 'num_classes' in head_config_copy:
                num_classes = head_config_copy.pop('num_classes')  # Extract and remove
                
            # For testing only - default to 5 if not specified
            if num_classes is None:
                num_classes = 10
                print(f"WARNING: Using default num_classes={num_classes} for classification head")
                
            self.head = ClassificationHead(
                input_dim=self.d_model,
                num_classes=num_classes,
                **head_config_copy
            )
        elif head_type == "regression":
            self.head = RegressionHead(
                input_dim=self.d_model,
                **head_config_copy
            )
        elif head_type == "token":
            # For token-based prediction, we need to map tokens to digit values
            
            # Get token IDs for digits 0-9
            digit_tokens = []
            for i in range(10):
                digit_token = self.tokenizer.encode(str(i))[-1]  # Get the token ID for each digit
                digit_tokens.append(digit_token)
                
            logger.info(f"Token prediction head will use digit tokens: {digit_tokens}")
            
            self.head = TokenPredictionHead(
                input_dim=self.d_model,
                digit_tokens=digit_tokens,
                **head_config_copy
            )
        else:
            raise ValueError(f"Unsupported head type: {head_type}")
        
        # Move head to device
        self.head.to(device)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None,
        return_activations: bool = False,
        return_uncertainties: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_activations: Whether to return actual activations too
            return_uncertainties: Whether to return uncertainty estimates
            
        Returns:
            Predictions (and optionally actual activations)
        """
        batch_size = input_ids.shape[0]
        
        # Determine token positions
        if self.token_pos == "last":
            if attention_mask is not None:
                # Get last non-padding token for each item in batch
                seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
                token_positions = seq_lengths
            else:
                # If no mask, just use last token
                token_positions = torch.tensor([input_ids.shape[1] - 1] * batch_size, 
                                              device=input_ids.device)
        else:
            # Use specified position
            token_positions = torch.tensor([min(self.token_pos, input_ids.shape[1] - 1)] * batch_size, 
                                          device=input_ids.device)
        
        # Run model with caching for activation extraction
        with torch.set_grad_enabled(self.training):
            # Special handling for token prediction head
            if self.head_type == "token":
                # For token prediction, get model's raw logits
                outputs = self.base_model(
                    input_ids,
                    attention_mask=attention_mask,
                    return_type="logits"  # Get raw logits
                )
                
                # Extract logits at desired token position
                logits_at_pos = []
                for i in range(batch_size):
                    pos = token_positions[i].item()
                    # Get logits at specific position
                    pos_logits = outputs[i, pos]
                    logits_at_pos.append(pos_logits)
                
                # Stack logits for batch processing
                logits_at_pos = torch.stack(logits_at_pos, dim=0)
                
                # Extract only logits for the digits tokens (0-9)
                if hasattr(self.head, "digit_tokens"):
                    digit_tokens = self.head.digit_tokens
                    # Get just the logits for digit tokens
                    digit_logits = torch.stack([logits_at_pos[:, token_idx] for token_idx in digit_tokens], dim=1)
                    predictions = digit_logits
                else:
                    # Fallback if we don't have digit tokens
                    predictions = logits_at_pos
                
                # Run the rest of the activation extraction code for other needed variables
                _, cache = self.base_model.run_with_cache(
                    input_ids,
                    attention_mask=attention_mask,
                    return_type=None,
                )
            else:
                # Standard path for classification and regression
                _, cache = self.base_model.run_with_cache(
                    input_ids,
                    attention_mask=attention_mask,
                    return_type=None,
                )
                
                # If in training mode, we need to preserve the computational graph to allow
                # gradient flow back to model parameters. Using the cache breaks this connection.
                if self.training:
                    # Get embeddings and run through model maintaining the computation graph
                    with torch.set_grad_enabled(True):
                        # Get embeddings (input to first layer)
                        embeddings = self.base_model.embed(input_ids)
                        
                        # Run through each layer up to our target feature layer
                        layer_inputs = embeddings
                        
                        # Layer by layer execution to maintain computation graph
                        for i in range(self.base_model.cfg.n_layers):
                            # Run through transformer block
                            block = self.base_model.blocks[i]
                            layer_outputs = block(layer_inputs)
                            layer_inputs = layer_outputs
                            
                            # At our feature layer, extract the features we need
                            if i == self.feature_layer or (self.feature_layer == -1 and i == self.base_model.cfg.n_layers - 1):
                                # Extract just the token positions we need
                                features = []
                                for j, pos in enumerate(token_positions):
                                    feature = layer_outputs[j, pos]
                                    features.append(feature)
                                
                                # Stack for batch processing
                                features = torch.stack(features, dim=0)
                                break
                else:
                    # For inference, we can use the cache approach which is faster
                    features = []
                    for i in range(batch_size):
                        pos = token_positions[i].item()
                        # Use the residual stream at the specified feature layer for prediction
                        feature = cache["resid_post", self.feature_layer][i, pos]
                        features.append(feature)
                    
                    # Stack features for batch processing
                    features = torch.stack(features, dim=0)
                
                # Run through prediction head
                predictions = self.head(features)
            
            # Extract actual activations if requested
            if (return_activations or return_uncertainties) and self.target_layer is not None and self.target_neuron is not None:
                act_layer = self.target_layer
                act_neuron = self.target_neuron
                
                activations = []
                for i in range(batch_size):
                    pos = token_positions[i].item()
                    activation = cache[self.layer_type, act_layer][i, pos, act_neuron]
                    activations.append(activation)
                    
                activations = torch.tensor(activations, device=self.device)
                
                # Calculate uncertainties if requested
                if return_uncertainties:
                    if self.head_type == "classification" or self.head_type == "token":
                        # For classification and token prediction, get softmax probabilities
                        probs = F.softmax(predictions, dim=1)
                        max_probs, _ = torch.max(probs, dim=1)
                        # Higher max probability means lower uncertainty
                        uncertainties = 1.0 - max_probs
                    else:
                        # For regression, use a simple heuristic
                        # This could be enhanced with more sophisticated uncertainty estimation
                        # Simple distance between actual and predicted
                        uncertainties = torch.abs(predictions - activations)
                    
                    return predictions, activations, uncertainties
                
                return predictions, activations
        
        return predictions
    
    def predict(
        self,
        texts: List[str],
        batch_size: int = 8,
        return_activations: bool = False,
        return_uncertainties: bool = False,
        return_raw: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate predictions for a list of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            return_activations: Whether to return actual activations
            return_uncertainties: Whether to return uncertainty estimates
            return_raw: Whether to return raw predictions (without normalization)
            
        Returns:
            Numpy array of predictions (and optionally activations/uncertainties)
        """
        all_predictions = []
        all_activations = [] if return_activations else None
        all_uncertainties = [] if return_uncertainties else None
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            
            # Generate predictions
            with torch.no_grad():
                if return_uncertainties:
                    preds, acts, uncs = self.forward(
                        inputs.input_ids,
                        inputs.attention_mask,
                        return_activations=True,
                        return_uncertainties=True,
                    )
                    all_predictions.append(preds.detach().cpu().numpy())
                    if return_activations:
                        all_activations.append(acts.detach().cpu().numpy())
                    all_uncertainties.append(uncs.detach().cpu().numpy())
                elif return_activations:
                    preds, acts = self.forward(
                        inputs.input_ids,
                        inputs.attention_mask,
                        return_activations=True,
                    )
                    all_predictions.append(preds.detach().cpu().numpy())
                    all_activations.append(acts.detach().cpu().numpy())
                else:
                    preds = self.forward(inputs.input_ids, inputs.attention_mask)
                    all_predictions.append(preds.detach().cpu().numpy())
        
        # Combine batches
        predictions = np.concatenate(all_predictions, axis=0)
        
        # Process predictions based on head type
        if not return_raw:
            if self.head_type == "token":
                # For token prediction, convert logits to probabilities
                tensor_preds = torch.tensor(predictions)
                probs = F.softmax(tensor_preds, dim=1).numpy()
                
                # Map digit tokens (0-9) directly to their values
                digit_values = np.arange(10)  # Values 0-9
                continuous_preds = np.sum(probs * digit_values.reshape(1, -1), axis=1)
                
                # If we're using tokens to predict activations, we need a mapping approach
                # This is a simple linear mapping from [0-9] range to the activation range
                # NOTE: In general, the token approach is less precise than regression
                if self.activation_mean is not None and self.activation_std is not None:
                    # Map from [0-9] to activation range using the provided statistics
                    min_act = self.activation_mean - 2 * self.activation_std  # approximate min
                    max_act = self.activation_mean + 2 * self.activation_std  # approximate max
                    act_range = max_act - min_act
                    
                    # Linear mapping: [0-9] -> [min_act, max_act]
                    continuous_preds = min_act + (continuous_preds / 9.0) * act_range
                    logger.info(f"Mapping token predictions from [0-9] to activation range [{min_act:.2f}, {max_act:.2f}]")
                
                predictions = continuous_preds
                
            elif self.head_type == "classification" and self.bin_edges is not None:
                # For classification, convert logits to class probabilities using PyTorch
                tensor_preds = torch.tensor(predictions)
                probs = F.softmax(tensor_preds, dim=1).numpy()
                
                # Convert class probabilities to continuous values using bin centers
                bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
                
                # Handle mismatch between number of classes and bin centers
                num_classes = probs.shape[1]
                num_centers = len(bin_centers)
                
                if num_classes == num_centers:
                    # Ideal case: classes match bin centers exactly
                    continuous_preds = np.sum(probs * bin_centers.reshape(1, -1), axis=1)
                elif num_classes < num_centers:
                    # If we have more bin centers than classes, use only the first num_classes centers
                    bin_centers_subset = bin_centers[:num_classes]
                    continuous_preds = np.sum(probs * bin_centers_subset.reshape(1, -1), axis=1)
                    logger.warning(f"More bin centers ({num_centers}) than classes ({num_classes}). Using first {num_classes} centers.")
                else:
                    # If we have more classes than bin centers, extrapolate centers with consistent spacing
                    if num_centers >= 2:
                        bin_spacing = (bin_centers[-1] - bin_centers[0]) / (num_centers - 1)
                        extended_centers = np.linspace(
                            bin_centers[0],
                            bin_centers[0] + (num_classes - 1) * bin_spacing,
                            num_classes
                        )
                        continuous_preds = np.sum(probs * extended_centers.reshape(1, -1), axis=1)
                        logger.warning(f"More classes ({num_classes}) than bin centers ({num_centers}). Extrapolating additional centers.")
                    else:
                        # Fallback if we can't estimate spacing
                        continuous_preds = np.argmax(probs, axis=1).astype(float)
                        logger.warning(f"Cannot properly map classes to continuous values. Using class indices as approximation.")
                
                predictions = continuous_preds
                
            # For regression, we're now making the head produce outputs in the target range directly
            # No denormalization needed - the raw outputs from the model should be in the target range
            
            # This is the old approach with normalization we're removing:
            # elif self.head_type == "regression" and self.activation_mean is not None and self.activation_std is not None:
            #     predictions = predictions * self.activation_std + self.activation_mean
        
        # Prepare return values
        if return_activations and all_activations:
            activations = np.concatenate(all_activations, axis=0)
            if return_uncertainties and all_uncertainties:
                uncertainties = np.concatenate(all_uncertainties, axis=0)
                return predictions, activations, uncertainties
            return predictions, activations
        elif return_uncertainties and all_uncertainties:
            uncertainties = np.concatenate(all_uncertainties, axis=0)
            return predictions, uncertainties
        return predictions
    
    def report(
        self,
        texts: List[str],
        confidence_threshold: float = 0.8,
        batch_size: int = 8,
    ) -> Dict:
        """
        Report activations with confidence measures.
        
        This method extends predict() to provide a more interpretable report
        about the model's internal activations, including confidence measures.
        
        Args:
            texts: List of input texts
            confidence_threshold: Threshold for high-confidence reports
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with predictions, confidences, and indicators
        """
        if self.head_type == "classification" or self.head_type == "token":
            preds, activations, uncertainties = self.predict(
                texts, 
                batch_size=batch_size,
                return_activations=True,
                return_uncertainties=True,
                return_raw=True,  # Get raw outputs for proper confidence calculation
            )
            
            # Calculate confidence scores (1 - uncertainty)
            confidence_scores = 1.0 - uncertainties
            
            # Get continuous predictions for easier interpretation
            if self.head_type == "token":
                # For token prediction, map directly to digit values
                tensor_preds = torch.tensor(preds)
                probs = F.softmax(tensor_preds, dim=1).numpy()
                digit_values = np.arange(10)  # Values 0-9
                continuous_preds = np.sum(probs * digit_values.reshape(1, -1), axis=1)
                
                # Scale to match activation distribution if needed
                if self.activation_mean is not None and self.activation_std is not None:
                    normalized_preds = continuous_preds / 9.0
                    z_min, z_max = -2, 2
                    scaled_preds = normalized_preds * (z_max - z_min) + z_min
                    continuous_preds = scaled_preds * self.activation_std + self.activation_mean
            elif self.bin_edges is not None:
                # Convert to PyTorch tensor for softmax operation 
                tensor_preds = torch.tensor(preds)
                probs = F.softmax(tensor_preds, dim=1).numpy()
                bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
                continuous_preds = np.sum(probs * bin_centers.reshape(1, -1), axis=1)
            else:
                # If bin edges aren't available, just use class indices as approximation
                continuous_preds = np.argmax(preds, axis=1).astype(float)
            
            report_data = {
                "raw_predictions": preds,
                "predicted_activations": continuous_preds,
                "actual_activations": activations,
                "confidence_scores": confidence_scores,
                "is_high_confidence": confidence_scores >= confidence_threshold,
                "error": np.abs(continuous_preds - activations),
                "classified_bin": np.argmax(preds, axis=1),
            }
        else:
            # For regression head
            preds, activations = self.predict(
                texts, 
                batch_size=batch_size,
                return_activations=True,
            )
            
            # Calculate simple confidence scores based on prediction error
            # Note: More sophisticated uncertainty estimation could be implemented here
            prediction_errors = np.abs(preds - activations)
            max_error = np.max(prediction_errors) + 1e-8  # Avoid division by zero
            confidence_scores = 1.0 - (prediction_errors / max_error)
            
            report_data = {
                "predicted_activations": preds,
                "actual_activations": activations,
                "confidence_scores": confidence_scores,
                "is_high_confidence": confidence_scores >= confidence_threshold,
                "error": prediction_errors,
            }
        
        # Add summary statistics
        report_data.update({
            "mean_error": np.mean(report_data["error"]),
            "mean_confidence": np.mean(confidence_scores),
            "high_confidence_percentage": np.mean(report_data["is_high_confidence"]) * 100,
        })
            
        return report_data
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if self.head_type == "classification":
            # For classification, compute accuracy and F1 score
            pred_classes = np.argmax(predictions, axis=1)
            if len(targets.shape) > 1 and targets.shape[1] > 1:
                # If targets are one-hot, convert to class indices
                target_classes = np.argmax(targets, axis=1)
            else:
                target_classes = targets.astype(int)
                
            # Accuracy
            accuracy = np.mean(pred_classes == target_classes)
            metrics["accuracy"] = accuracy
            
            # If binary classification, compute F1 score
            if predictions.shape[1] == 2:
                true_positives = np.sum((pred_classes == 1) & (target_classes == 1))
                false_positives = np.sum((pred_classes == 1) & (target_classes == 0))
                false_negatives = np.sum((pred_classes == 0) & (target_classes == 1))
                
                precision = true_positives / (true_positives + false_positives + 1e-8)
                recall = true_positives / (true_positives + false_negatives + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                metrics["precision"] = precision
                metrics["recall"] = recall
                metrics["f1_score"] = f1
        else:
            # For regression, compute MSE, MAE, and R^2
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            
            # R^2 score
            ss_total = np.sum((targets - np.mean(targets)) ** 2)
            ss_residual = np.sum((targets - predictions) ** 2)
            r2 = 1 - (ss_residual / (ss_total + 1e-8))
            
            metrics["mse"] = mse
            metrics["mae"] = mae
            metrics["r2_score"] = r2
        
        return metrics
    
    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Directory to save to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save head
        head_path = os.path.join(path, "head.pt")
        torch.save(self.head.state_dict(), head_path)
        
        # Save configuration
        config = {
            "head_type": self.head_type,
            "target_layer": self.target_layer,
            "target_neuron": self.target_neuron,
            "layer_type": self.layer_type,
            "token_pos": self.token_pos,
            "feature_layer": self.feature_layer,
            "head_config": self.head.get_config(),
            "base_model_name": self.base_model.cfg.model_name,
            "d_model": self.d_model,
            "bin_edges": self.bin_edges.tolist() if isinstance(self.bin_edges, np.ndarray) else self.bin_edges,
            "activation_mean": self.activation_mean,
            "activation_std": self.activation_std,
        }
        
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(
        cls,
        path: str,
        base_model: Optional[HookedTransformer] = None,
        base_model_name: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        device: str = "mps",
    ) -> "ActivationPredictor":
        """
        Load model from disk.
        
        Args:
            path: Directory to load from
            base_model: TransformerLens model (if None, load based on config)
            base_model_name: Name of model to load (if base_model is None)
            tokenizer: Tokenizer for the model (if None, use base_model.tokenizer)
            device: Device to load model to
            
        Returns:
            Loaded ActivationPredictor
        """
        # Load configuration
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Load base model if not provided
        if base_model is None:
            if base_model_name is None:
                base_model_name = config["base_model_name"]
            
            from transformer_lens import HookedTransformer
            base_model = HookedTransformer.from_pretrained(base_model_name)
        
        # Create head configuration
        head_config = config.get("head_config", {})
        head_type = config["head_type"]
        
        # Get num_classes from the appropriate place
        num_classes = head_config.get("num_classes")
        
        # Only remove class_name and input_dim to avoid parameter collisions
        # (but keep num_classes in the head_config if needed)
        if "class_name" in head_config:
            del head_config["class_name"]
        if "input_dim" in head_config:
            del head_config["input_dim"]
            
        # Only remove num_classes from head_config if we extracted it above
        if "num_classes" in head_config and num_classes is not None:
            del head_config["num_classes"]
        
        # Convert bin_edges back to numpy array if present
        bin_edges = config.get("bin_edges")
        if bin_edges is not None:
            bin_edges = np.array(bin_edges)
        
        # Extract type-compatible values from the config
        target_layer = config.get("target_layer")  # Might be None
        target_neuron = config.get("target_neuron")  # Might be None
        
        # Create model
        model = cls(
            base_model=base_model,
            head_type=head_type,
            num_classes=head_config.get("num_classes"),
            target_layer=target_layer,
            target_neuron=target_neuron,
            layer_type=config["layer_type"],
            token_pos=config["token_pos"],
            feature_layer=config.get("feature_layer", -1),
            head_config=head_config,
            tokenizer=tokenizer,
            device=device,
            bin_edges=bin_edges,
            activation_mean=config.get("activation_mean"),
            activation_std=config.get("activation_std"),
        )
        
        # Load head weights
        head_path = os.path.join(path, "head.pt")
        model.head.load_state_dict(torch.load(head_path, map_location=device))
        
        return model

def test_activation_predictor():
    """Test the ActivationPredictor functionality."""
    import os
    import random
    from transformer_lens import HookedTransformer
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Use a small test model
    print("Loading model...")
    model_name = "gpt2-small"  # Small model for quick testing
    model = HookedTransformer.from_pretrained(model_name)
    
    # Initialize on appropriate device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Sample texts
    sample_texts = [
        "The cat sat on the mat.",
        "Machine learning models can be difficult to interpret.",
        "Transformers use attention mechanisms to process sequences.",
        "Neural networks have revolutionized artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    # Test classification predictor
    print("\nTesting classification predictor...")
    # Create bin edges for classification
    bin_edges = np.linspace(-10, 10, 6)  # 5 bins from -10 to 10
    
    class_predictor = ActivationPredictor(
        base_model=model,
        head_type="classification",
        num_classes=5,  # Match number of bins
        target_layer=6,
        target_neuron=500,
        layer_type="mlp_out",
        token_pos="last",
        feature_layer=-1,
        head_config={"hidden_dim": 32},
        device=device,
        bin_edges=bin_edges,
    )
    
    # Test forward pass
    print("Testing forward pass...")
    inputs = class_predictor.tokenizer(
        sample_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    
    with torch.no_grad():
        outputs, activations = class_predictor.forward(
            inputs.input_ids,
            inputs.attention_mask,
            return_activations=True,
        )
    
    print(f"Outputs shape: {outputs.shape}")
    print(f"Activations shape: {activations.shape}")
    
    # Test prediction method
    print("\nTesting batch prediction...")
    predictions, real_activations = class_predictor.predict(
        sample_texts,
        batch_size=2,
        return_activations=True,
    )
    
    print(f"Predictions shape: {predictions.shape if isinstance(predictions, np.ndarray) else 'scalar'}")
    print(f"Sample predictions: {predictions[:2]}")
    print(f"Real activations: {real_activations[:2]}")
    
    # Test report method
    print("\nTesting reporting functionality...")
    report = class_predictor.report(sample_texts, confidence_threshold=0.6)
    
    print(f"Mean error: {report['mean_error']:.4f}")
    print(f"Mean confidence: {report['mean_confidence']:.4f}")
    print(f"High confidence percentage: {report['high_confidence_percentage']:.2f}%")
    
    # Test save and load
    print("\nTesting save and load...")
    save_dir = "test_predictor_output"
    os.makedirs(save_dir, exist_ok=True)
    
    class_predictor.save(save_dir)
    print(f"Model saved to {save_dir}")
    
    loaded_predictor = ActivationPredictor.load(
        path=save_dir,
        base_model=model,
        device=device,
    )
    
    with torch.no_grad():
        loaded_outputs = loaded_predictor.forward(inputs.input_ids, inputs.attention_mask)
    
    print(f"Loaded outputs shape: {loaded_outputs.shape}")
    print(f"Outputs match: {torch.allclose(outputs, loaded_outputs)}")
    
    # Test regression predictor
    print("\nTesting regression predictor...")
    reg_predictor = ActivationPredictor(
        base_model=model,
        head_type="regression",
        target_layer=6,
        target_neuron=500,
        layer_type="mlp_out",
        token_pos="last",
        feature_layer=-1,
        head_config={"hidden_dim": 32},
        device=device,
        activation_mean=0.0,  # Example normalization parameters
        activation_std=1.0,
    )
    
    with torch.no_grad():
        reg_outputs = reg_predictor.forward(inputs.input_ids, inputs.attention_mask)
    
    print(f"Regression outputs shape: {reg_outputs.shape}")
    print(f"Sample regression outputs: {reg_outputs[:2].detach().cpu().numpy()}")
    
    # Test regression report
    print("\nTesting regression reporting...")
    reg_report = reg_predictor.report(sample_texts[:3])
    
    print(f"Regression mean error: {reg_report['mean_error']:.4f}")
    print(f"Regression mean confidence: {reg_report['mean_confidence']:.4f}")
    
    # Test metrics computation
    print("\nTesting metrics computation...")
    # Classification metrics
    dummy_preds = np.random.rand(5, 5)  # 5 samples, 5 classes
    dummy_targets = np.random.randint(0, 5, size=5)  # Random class indices
    
    metrics = class_predictor.compute_metrics(dummy_preds, dummy_targets)
    print("Classification metrics:", metrics)
    
    # Regression metrics
    dummy_reg_preds = np.random.randn(5)
    dummy_reg_targets = np.random.randn(5)
    
    reg_metrics = reg_predictor.compute_metrics(dummy_reg_preds, dummy_reg_targets)
    print("Regression metrics:", reg_metrics)
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    # Run test when the file is executed directly
    test_activation_predictor()