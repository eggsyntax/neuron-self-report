# selective_unfreezing.py
"""
Module for selectively unfreezing parts of transformer models.

This module provides functions to control which parts of a transformer model
are frozen or unfrozen during fine-tuning, allowing for targeted adaptation
while preventing information leakage.
"""

import logging
from typing import List, Optional, Union, Dict, Set
import torch
from transformer_lens import HookedTransformer

logger = logging.getLogger("unfreezing")

def freeze_entire_model(model: torch.nn.Module) -> int:
    """
    Freeze all parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of parameters frozen
    """
    frozen_count = 0
    for param in model.parameters():
        param.requires_grad = False
        frozen_count += param.numel()
    
    return frozen_count

def unfreeze_entire_model(model: torch.nn.Module) -> int:
    """
    Unfreeze all parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of parameters unfrozen
    """
    unfrozen_count = 0
    for param in model.parameters():
        param.requires_grad = True
        unfrozen_count += param.numel()
    
    return unfrozen_count

def get_layer_from_name(name: str) -> Optional[int]:
    """
    Extract layer number from parameter name.
    
    Args:
        name: Parameter name
        
    Returns:
        Layer number or None if not found
    """
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part in ["blocks", "layers", "layer"]:
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1])
    
    return None

def unfreeze_after_layer(model: torch.nn.Module, layer_index: int) -> int:
    """
    Unfreeze all parameters in layers after the specified layer.
    
    Args:
        model: PyTorch model
        layer_index: Layer index (unfreeze everything after this layer)
        
    Returns:
        Number of parameters unfrozen
    """
    unfrozen_count = 0
    
    for name, param in model.named_parameters():
        current_layer = get_layer_from_name(name)
        
        # If we can't determine the layer, leave it frozen
        if current_layer is None:
            continue
        
        # Unfreeze if the layer is after our target layer
        if current_layer > layer_index:
            param.requires_grad = True
            unfrozen_count += param.numel()
    
    return unfrozen_count

def unfreeze_components(
    model: torch.nn.Module, 
    components: List[str],
    after_layer: Optional[int] = None
) -> int:
    """
    Selectively unfreeze specific components of the model.
    
    Args:
        model: PyTorch model
        components: List of component types to unfreeze ('attention', 'mlp', etc.)
        after_layer: Only unfreeze components after this layer (if provided)
        
    Returns:
        Number of parameters unfrozen
    """
    unfrozen_count = 0
    
    # Lowercase and create a set for faster lookups
    component_set = {c.lower() for c in components}
    
    for name, param in model.named_parameters():
        name_lower = name.lower()
        
        # Check if this parameter belongs to any of the specified components
        should_unfreeze = False
        for component in component_set:
            if component in name_lower:
                should_unfreeze = True
                break
        
        # If after_layer is specified, only unfreeze components in later layers
        if should_unfreeze and after_layer is not None:
            current_layer = get_layer_from_name(name)
            if current_layer is None or current_layer <= after_layer:
                should_unfreeze = False
        
        if should_unfreeze:
            param.requires_grad = True
            unfrozen_count += param.numel()
    
    return unfrozen_count

def apply_unfreezing_strategy(
    model: torch.nn.Module,
    strategy: str,
    target_layer: Optional[int] = None,
    from_layer: Optional[int] = None,
    components: Optional[Union[str, List[str]]] = None
) -> Dict:
    """
    Apply a specific unfreezing strategy to the model.
    
    Args:
        model: PyTorch model (typically a transformer)
        strategy: Unfreezing strategy ('none', 'all', 'after_target', 'from_layer', 'selective')
        target_layer: Target layer for 'after_target' strategy
        from_layer: Starting layer for 'from_layer' strategy
        components: Component types to unfreeze for 'selective' strategy
            
    Returns:
        Dictionary with statistics about unfreezing
    """
    # Initialize results
    results = {
        "strategy": strategy,
        "total_params": sum(p.numel() for p in model.parameters()),
        "unfrozen_params": 0,
        "unfrozen_percentage": 0.0,
    }
    
    # First freeze everything
    freeze_entire_model(model)
    
    # Apply the requested strategy
    if strategy == "none":
        logger.info("Strategy: Freezing entire model")
        # Everything already frozen, nothing to do
        
    elif strategy == "all":
        logger.info("Strategy: Unfreezing entire model")
        results["unfrozen_params"] = unfreeze_entire_model(model)
        
    elif strategy == "after_target":
        if target_layer is None:
            logger.warning("Target layer not specified for 'after_target' strategy. Defaulting to freezing all.")
        else:
            logger.info(f"Strategy: Unfreezing all parameters after layer {target_layer}")
            results["unfrozen_params"] = unfreeze_after_layer(model, target_layer)
            
    elif strategy == "from_layer":
        if from_layer is None:
            logger.warning("Starting layer not specified for 'from_layer' strategy. Defaulting to freezing all.")
        else:
            logger.info(f"Strategy: Unfreezing all parameters from layer {from_layer} onwards")
            results["unfrozen_params"] = unfreeze_after_layer(model, from_layer - 1)
            
    elif strategy == "selective":
        if not components:
            logger.warning("No components specified for 'selective' strategy. Defaulting to freezing all.")
        else:
            # Parse components string if needed
            if isinstance(components, str):
                component_list = [c.strip() for c in components.split(",") if c.strip()]
            else:
                component_list = components
                
            after_layer = None
            if target_layer is not None:
                logger.info(f"Strategy: Selectively unfreezing components {component_list} after layer {target_layer}")
                after_layer = target_layer
            else:
                logger.info(f"Strategy: Selectively unfreezing components {component_list} in all layers")
                
            results["unfrozen_params"] = unfreeze_components(model, component_list, after_layer)
    
    # Calculate percentage unfrozen
    if results["total_params"] > 0:
        results["unfrozen_percentage"] = (results["unfrozen_params"] / results["total_params"]) * 100
    
    logger.info(f"Unfrozen parameters: {results['unfrozen_params']:,} "
                f"({results['unfrozen_percentage']:.2f}% of total)")
    
    # print("=== Parameters and their trainable status after unfreezing ===")
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")

    return results

def get_trainable_parameters_info(model: torch.nn.Module) -> Dict:
    """
    Get detailed information about trainable vs frozen parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with trainable parameter statistics
    """
    # Count parameters by component type
    component_counts = {}
    layer_counts = {}
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        # Count by component
        component = "other"
        if "embed" in name.lower():
            component = "embedding"
        elif "norm" in name.lower() or "ln" in name.lower():
            component = "normalization"
        elif "attn" in name.lower() or "attention" in name.lower():
            component = "attention"
        elif "mlp" in name.lower() or "ffn" in name.lower():
            component = "mlp"
        
        if component not in component_counts:
            component_counts[component] = {"total": 0, "trainable": 0}
        
        component_counts[component]["total"] += param_count
        
        # Count by layer
        layer = get_layer_from_name(name)
        if layer is not None:
            if layer not in layer_counts:
                layer_counts[layer] = {"total": 0, "trainable": 0}
            
            layer_counts[layer]["total"] += param_count
        
        # Count trainable parameters
        if param.requires_grad:
            trainable_params += param_count
            
            if component in component_counts:
                component_counts[component]["trainable"] += param_count
            
            if layer is not None and layer in layer_counts:
                layer_counts[layer]["trainable"] += param_count
    
    # Calculate percentages
    for component in component_counts:
        total = component_counts[component]["total"]
        trainable = component_counts[component]["trainable"]
        component_counts[component]["percentage"] = (trainable / total * 100) if total > 0 else 0
    
    for layer in layer_counts:
        total = layer_counts[layer]["total"]
        trainable = layer_counts[layer]["trainable"]
        layer_counts[layer]["percentage"] = (trainable / total * 100) if total > 0 else 0
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": (trainable_params / total_params * 100) if total_params > 0 else 0,
        "by_component": component_counts,
        "by_layer": layer_counts
    }