# architecture.py
# ActivationPredictor: For predicting activations using different head types

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from typing import Optional, Dict, Any, Union, List # Ensure List is definitely here

class ActivationPredictor(nn.Module):
    """
    A module that takes a base transformer model and adds a prediction head 
    to predict neuron activations.
    """
    def __init__(self, 
                 base_model: HookedTransformer,
                 prediction_head_type: str, # "regression", "classification", "token_binary", "token_digit"
                 base_model_output_dim: Optional[int] = None, # d_model or d_residual, needed for custom heads
                 num_classes: Optional[int] = None, # For classification head
                 device: Optional[str] = None):
        """
        Initializes the ActivationPredictor.

        Args:
            base_model: The pre-trained HookedTransformer model.
            prediction_head_type: Type of prediction head to use.
            base_model_output_dim: The dimension of the features from the base model that will be fed into the head.
                                   Typically model.cfg.d_model or model.cfg.d_residual.
                                   Required for 'regression' and 'classification' heads.
            num_classes: Number of classes for the classification head. Required if head_type is 'classification'.
            device: Device to run the model on.
        """
        super().__init__()
        self.base_model = base_model
        self.prediction_head_type = prediction_head_type
        
        # Ensure self.device is always a string
        if device:
            self.device: str = device
        elif hasattr(base_model, 'cfg') and hasattr(base_model.cfg, 'device') and base_model.cfg.device is not None:
            self.device: str = base_model.cfg.device
        else:
            self.device: str = 'cpu'
        
        self.base_model.to(self.device)
        # It's generally assumed the base_model's parameters might be frozen/unfrozen by the Trainer.
        # By default, we don't alter their requires_grad status here.

        if prediction_head_type in ["regression", "classification"]:
            if base_model_output_dim is None:
                raise ValueError("base_model_output_dim must be provided for regression/classification heads.")
            self.base_model_output_dim = base_model_output_dim
        
        if prediction_head_type == "regression":
            # Predicts a single continuous value
            self.head = nn.Linear(self.base_model_output_dim, 1)
        elif prediction_head_type == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification head.")
            self.num_classes = num_classes
            self.head = nn.Linear(self.base_model_output_dim, self.num_classes)
        elif prediction_head_type in ["token_binary", "token_digit"]:
            # For token prediction pathways, the "head" is the base model's existing unembedding layer.
            # No separate head module is strictly needed here, but we might define a conceptual one
            # or handle it directly in the forward pass.
            # The base_model.unembed is W_U, shape [d_model, d_vocab]
            # The base_model.ln_final is LayerNorm
            # Output logits = base_model.ln_final(features) @ base_model.unembed.W_U + base_model.b_U (if bias exists)
            self.head = None # Using the model's own unembedding
            print(f"Token prediction head type '{prediction_head_type}' will use base model's unembedding.")
        else:
            raise ValueError(f"Unsupported prediction_head_type: {prediction_head_type}")

        if self.head is not None:
            self.head.to(self.device)

        print(f"ActivationPredictor initialized with head type: {prediction_head_type}, device: {self.device}")

    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, # For HuggingFace compatibility, less used in TransformerLens direct
                feature_extraction_hook_point: Optional[str] = None, # e.g. 'blocks.11.hook_resid_post'
                target_token_position: Optional[Union[str, int]] = None # 'last' or specific index for feature extraction
               ) -> torch.Tensor:
        """
        Forward pass of the ActivationPredictor.

        Args:
            input_ids: Tensor of input token IDs. Shape: [batch_size, seq_len].
            attention_mask: Optional attention mask.
            feature_extraction_hook_point: The hook point from which to extract features for the prediction head.
                                           Required for 'regression' and 'classification' heads.
                                           If None and head type is token-based, uses final layer output.
            target_token_position: The token position from which to take features. 'last' or an integer index.
                                   Required for 'regression' and 'classification' heads.

        Returns:
            Prediction tensor. Shape depends on the head type:
            - Regression: [batch_size, 1]
            - Classification: [batch_size, num_classes]
            - Token-based: [batch_size, d_vocab] (logits for all tokens at the target position)
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if self.prediction_head_type in ["regression", "classification"]:
            if feature_extraction_hook_point is None:
                raise ValueError("feature_extraction_hook_point is required for regression/classification heads.")
            if target_token_position is None:
                raise ValueError("target_token_position is required for regression/classification heads.")

            # Cache to store the features
            feature_cache = {}
            def hook_fn(activation_tensor, hook):
                feature_cache['features'] = activation_tensor.detach().clone() # Detach if base_model is frozen

            # Run model with the hook to get features
            with torch.set_grad_enabled(self.base_model.training): # Respect base model's training mode for hooks
                 # If base_model params are frozen, no_grad() context might be used by trainer
                _ = self.base_model.run_with_hooks(
                    input_ids, 
                    fwd_hooks=[(feature_extraction_hook_point, hook_fn)],
                    # attention_mask=attention_mask # TransformerLens run_with_hooks doesn't typically use HF attention_mask directly
                )
            
            if 'features' not in feature_cache:
                raise RuntimeError(f"Failed to capture features from hook_point: {feature_extraction_hook_point}")
            
            features = feature_cache['features'] # Shape: [batch_size, seq_len, d_model_or_resid]

            # Select features from the target token position
            if target_token_position == "last":
                # Assuming input_ids are not padded or padding is handled by model.
                # For TransformerLens, typically the full sequence length is used.
                # If input_ids has padding, a more robust way to get last token index is needed.
                token_features = features[:, -1, :] # Features of the last token
            elif isinstance(target_token_position, int):
                if not (0 <= target_token_position < features.shape[1]):
                     raise ValueError(f"target_token_position index {target_token_position} out of bounds for seq_len {features.shape[1]}")
                token_features = features[:, target_token_position, :]
            else:
                raise ValueError(f"Invalid target_token_position: {target_token_position}")
            
            # Pass features through the prediction head
            if self.head is not None:
                output = self.head(token_features) # Shape: [batch_size, 1] or [batch_size, num_classes]
            else:
                # This case should ideally not be reached if head is None and type is reg/class
                raise RuntimeError("Regression/Classification head is None, cannot process.")


        elif self.prediction_head_type in ["token_binary", "token_digit"]:
            # Use the base model's output (logits) directly.
            # The `feature_extraction_hook_point` would typically be the final layer's output before unembedding.
            # Or, we can just run the model and get its standard output logits.
            
            # TransformerLens models return logits directly
            # The `target_token_position` will be used to select which token's logits we care about.
            model_output_logits = self.base_model(input_ids, return_type="logits") # Shape: [batch_size, seq_len, d_vocab]
            
            if target_token_position == "last":
                output = model_output_logits[:, -1, :] # Logits for the last token position
            elif isinstance(target_token_position, int):
                if not (0 <= target_token_position < model_output_logits.shape[1]):
                     raise ValueError(f"target_token_position index {target_token_position} out of bounds for seq_len {model_output_logits.shape[1]}")
                output = model_output_logits[:, target_token_position, :] # Logits for a specific token position
            else:
                # If no specific token position, perhaps return all? Or raise error.
                # For predicting a single activation, we need a specific token's output.
                raise ValueError(f"target_token_position is required for token-based prediction and must be 'last' or an int.")
        else:
            raise ValueError(f"Unsupported prediction_head_type: {self.prediction_head_type}")
            
        return output

    def freeze_base_model(self, freeze: bool = True):
        """Helper to freeze/unfreeze base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = not freeze
        print(f"Base model parameters frozen: {freeze}")

    def unfreeze_layers_after_target(self, target_neuron_layer: int):
        """Unfreezes layers after (and including) the target_neuron_layer. Head is always trainable."""
        self.freeze_base_model(True) # Freeze all first
        for i in range(target_neuron_layer, self.base_model.cfg.n_layers):
            for param in self.base_model.blocks[i].parameters():
                param.requires_grad = True
        # Also unfreeze final LayerNorm and unembedding if they exist and are separate
        if hasattr(self.base_model, 'ln_final'):
            for param in self.base_model.ln_final.parameters():
                param.requires_grad = True
        if hasattr(self.base_model, 'unembed'): # W_U
            for param in self.base_model.unembed.parameters():
                param.requires_grad = True
        print(f"Unfroze layers from {target_neuron_layer} onwards. Head is trainable.")

    def unfreeze_selective_components(self, components_to_unfreeze: List[str]):
        """Unfreezes only specified components (e.g., ['mlp', 'attn_out']). Head is always trainable."""
        self.freeze_base_model(True) # Freeze all first
        for name, module in self.base_model.named_modules():
            for comp_name_part in components_to_unfreeze:
                if comp_name_part in name: # Simple substring match
                    for param in module.parameters():
                        param.requires_grad = True
                    print(f"Unfroze component containing: {name} (matched by {comp_name_part})")
        print(f"Selectively unfroze components. Head is trainable.")


# Example Usage (for testing purposes)
if __name__ == '__main__':
    print("Setting up a dummy model for ActivationPredictor example...")
    try:
        if torch.backends.mps.is_available():
            default_device = "mps"
        elif torch.cuda.is_available():
            default_device = "cuda"
        else:
            default_device = "cpu"

        model_name = "gpt2-small" # or "EleutherAI/pythia-14m"
        base_model_instance = HookedTransformer.from_pretrained(model_name, device=default_device)
        base_model_instance.eval() # Important if not fine-tuning the base model itself

        # Dummy input
        dummy_input_ids = base_model_instance.to_tokens("This is a test sentence.").to(default_device) # Shape [1, seq_len]
        
        # --- Regression Head Example ---
        print("\n--- Regression Head Example ---")
        # Assume we want to predict an activation based on features from the last layer's residual stream
        # For gpt2-small, d_model = 768. n_layers = 12.
        # Last layer residual stream hook point:
        feature_hook_point_reg = f"blocks.{base_model_instance.cfg.n_layers - 1}.hook_resid_post"
        
        reg_predictor = ActivationPredictor(base_model=base_model_instance,
                                            prediction_head_type="regression",
                                            base_model_output_dim=base_model_instance.cfg.d_model,
                                            device=default_device)
        reg_predictor.eval() # Set predictor to eval mode
        
        with torch.no_grad():
            reg_output = reg_predictor(dummy_input_ids, 
                                       feature_extraction_hook_point=feature_hook_point_reg,
                                       target_token_position="last")
        print(f"Regression output shape: {reg_output.shape}") # Expected: [1, 1]
        print(f"Regression output value: {reg_output.item()}")

        # --- Classification Head Example ---
        print("\n--- Classification Head Example ---")
        num_act_bins = 5 # Example: predict which of 5 bins an activation falls into
        class_predictor = ActivationPredictor(base_model=base_model_instance,
                                              prediction_head_type="classification",
                                              base_model_output_dim=base_model_instance.cfg.d_model,
                                              num_classes=num_act_bins,
                                              device=default_device)
        class_predictor.eval()
        with torch.no_grad():
            class_output = class_predictor(dummy_input_ids,
                                           feature_extraction_hook_point=feature_hook_point_reg, # Can use same features
                                           target_token_position="last")
        print(f"Classification output shape: {class_output.shape}") # Expected: [1, num_act_bins]
        print(f"Classification output logits: {class_output}")


        # --- Token Prediction Head Example (e.g., for 'on'/'off' or '0'-'9' tokens) ---
        print("\n--- Token Prediction (Binary/Digit) Example ---")
        # This uses the model's own unembedding.
        # The 'head' is conceptual; the ActivationPredictor class just routes to model's logits.
        token_predictor = ActivationPredictor(base_model=base_model_instance,
                                              prediction_head_type="token_binary", # or "token_digit"
                                              device=default_device)
        token_predictor.eval()
        with torch.no_grad():
            # For token prediction, we usually care about the logits for the *next* token prediction.
            # So, if we feed "This is a test", we might look at logits from the "test" position.
            token_output_logits = token_predictor(dummy_input_ids,
                                                  target_token_position="last") # Logits at the final position
        
        print(f"Token predictor output logits shape: {token_output_logits.shape}") # Expected: [1, d_vocab]
        print(f"Top 5 predicted token IDs at last position: {torch.topk(token_output_logits[0], 5).indices.tolist()}")
        
        # Example of freezing/unfreezing (conceptual, actual use in Trainer)
        print("\n--- Freezing/Unfreezing Examples ---")
        reg_predictor.freeze_base_model(True)
        # reg_predictor.freeze_base_model(False) # Unfreeze all
        # reg_predictor.unfreeze_layers_after_target(target_neuron_layer=base_model_instance.cfg.n_layers // 2)
        # reg_predictor.unfreeze_selective_components(['mlp.c_fc', 'attn.W_O']) # Example component names

    except Exception as e:
        print(f"An error occurred during the ActivationPredictor example usage: {e}")
        import traceback
        traceback.print_exc()
        print("Ensure required libraries are installed and model can be loaded.")
