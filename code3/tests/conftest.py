# tests/conftest.py
import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

@pytest.fixture(scope="session")
def small_model_config() -> HookedTransformerConfig:
    """
    Provides a config for a very small HookedTransformer model for testing.
    """
    return HookedTransformerConfig(
        n_layers=2,
        d_model=32,
        d_head=16,
        n_heads=2,
        d_mlp=64,
        d_vocab=100, # Small vocab for speed
        n_ctx=32,    # Small context window
        act_fn="relu",
        normalization_type="LN", # LayerNorm
        tokenizer_name="gpt2", # Use a real tokenizer name
        use_attn_result=True,
        use_split_qkv_input=True,
        seed=42
    )

@pytest.fixture(scope="session")
def small_test_model(small_model_config: HookedTransformerConfig) -> HookedTransformer:
    """
    Provides a small, predictable HookedTransformer instance for testing.
    Initialized on CPU for consistency in tests unless device is critical.
    """
    model = HookedTransformer(small_model_config)
    model.eval() # Set to eval mode by default for tests
    return model

@pytest.fixture(scope="session")
def device() -> str:
    """
    Determines the device to use for tests that might need specific hardware.
    Prefers MPS > CUDA > CPU.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
