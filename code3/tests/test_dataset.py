# tests/test_dataset.py
# pylint: disable=import-error
import pytest # type: ignore
import torch
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from transformer_lens import HookedTransformer
# Assuming conftest.py provides small_test_model and device fixtures
# from ..dataset import ActivationDatasetGenerator # Relative import if tests is a package
# For now, let's assume direct import path works or adjust PYTHONPATH if needed
from dataset import ActivationDatasetGenerator
from .conftest import small_test_model, device # Import fixtures

class TestActivationDatasetGenerator:

    @pytest.fixture
    def setup_generator(self, small_test_model: HookedTransformer, device: str):
        """Fixture to set up ActivationDatasetGenerator with a small model."""
        # Example hook point and neuron for testing
        # For a 2-layer model, layer 0 or 1. d_mlp = 64.
        hook_point = f"blocks.0.mlp.hook_post" 
        neuron_index = 10 
        
        # Ensure model is on the correct device for the test
        small_test_model.to(device)
        
        generator = ActivationDatasetGenerator(
            model=small_test_model,
            hook_point=hook_point,
            neuron_layer=0,
            neuron_index=neuron_index,
            device=device
        )
        return generator, small_test_model # Return model for tokenizer access

    def test_generator_initialization(self, setup_generator):
        generator, model = setup_generator
        assert generator.model == model
        assert generator.hook_point == "blocks.0.mlp.hook_post"
        assert generator.neuron_index == 10
        assert generator.device == model.cfg.device

    def test_get_activations_mocked(self, setup_generator, device: str):
        generator, model = setup_generator
        texts = ["hello world", "test sentence"]
        
        # Mock the model's run_with_hooks to control its output
        # The hook_fn inside _get_activations will capture this
        # The shape of mock_activation should be [batch_size, seq_len, activation_dim]
        # For our small_test_model, d_mlp = 64. Let seq_len be 5 for "hello world" (incl BOS)
        # and 4 for "test sentence" (incl BOS)
        
        # Mock activations for "hello world" (BOS, h, e, l, l, o,  , w, o, r, l, d) -> let's say 12 tokens
        # Let's assume model.to_tokens("hello world", prepend_bos=True) gives seq_len = S1
        # And model.to_tokens("test sentence", prepend_bos=True) gives seq_len = S2
        # For simplicity, let's assume fixed seq_len for mock, e.g., 5
        mock_seq_len = 5
        mock_activation_dim = model.cfg.d_mlp # 64 for small_test_model
        
        # This will be returned by the mocked hook when run_with_hooks is called
        # The hook inside _get_activations will then process this.
        # The hook function itself is complex to mock directly, so we mock what it *sees*.
        
        # We need to mock `model.run_with_hooks`
        # The hook function in _get_activations appends to activation_cache['activation']
        # So, the mock needs to simulate the hook_fn being called and populating this cache.

        # Simpler: mock the entire _get_activations method for some tests,
        # or mock the run_with_hooks to simulate the hook behavior.

        # Let's mock run_with_hooks to call our fake hook_fn
        # which populates the cache in the way the real one would.
        
        # Mock return values for two calls
        # Activation for "hello world" at last token, neuron 10
        # Activation for "test sentence" at last token, neuron 10
        
        # This is tricky because the hook function is defined inside _get_activations.
        # A more direct way is to patch `model.run_with_hooks` and have it
        # directly manipulate the `activation_cache` that `_get_activations` uses.
        # However, that cache is local to `_get_activations`.

        # Alternative: Patch the hook_fn itself if it were a method of the class, but it's not.
        
        # Simplest for now: mock the entire model.run_with_hooks call
        # to simulate the effect of the hook.
        
        # Let's assume "hello world" tokenized is 3 tokens + BOS = 4
        # "test sentence" tokenized is 3 tokens + BOS = 4
        # (This depends on the dummy tokenizer of small_test_model, which is None,
        # so to_tokens will use gpt2 tokenizer by default if not careful)
        
        # To make it robust, let's use the actual tokenizer from a real small model for this test part
        # or ensure our small_test_model has a minimal tokenizer.
        # For now, let's assume a fixed tokenization length for the mock.
        
        # Mocking strategy:
        # The hook_fn inside _get_activations does: activation_cache['activation'] = activation_tensor.detach().clone()
        # We need run_with_hooks to effectively do this.
        
        # Let's make a fake activation tensor that would be passed to the hook
        # For "hello world", assume 4 tokens. Last token index = 3.
        # For "test sentence", assume 4 tokens. Last token index = 3.
        
        # Mock values for the specific neuron 10
        expected_act_val1 = 0.5
        expected_act_val2 = -0.3
        
        # This is the tensor that the hook would see for the whole layer
        mock_layer_activation1 = torch.randn(1, 4, mock_activation_dim, device=device) # batch, seq, d_mlp
        mock_layer_activation1[0, 3, generator.neuron_index] = expected_act_val1 # Set value for target neuron at last token
        
        mock_layer_activation2 = torch.randn(1, 4, mock_activation_dim, device=device)
        mock_layer_activation2[0, 3, generator.neuron_index] = expected_act_val2

        # We need to make run_with_hooks call the hook_fn with these tensors.
        # The hook_fn is passed as an argument to run_with_hooks.
        # This is getting complicated to mock precisely.

        # Let's test a higher level: generate_dataset_from_texts and mock _get_activations
        with patch.object(ActivationDatasetGenerator, '_get_activations') as mock_get_acts:
            # _get_activations returns: Tuple[List[torch.Tensor], List[List[str]]]
            # List of tensors, where each tensor is the activation of the target neuron (scalar)
            mock_get_acts.return_value = (
                [torch.tensor(expected_act_val1), torch.tensor(expected_act_val2)],
                [["<|BOS|>", "hello", "world", "!"], ["<|BOS|>", "test", "sent", "."]] # Dummy tokens
            )
            
            df = generator.generate_dataset_from_texts(texts, token_position="last")
            
            mock_get_acts.assert_called_once_with(texts, "last")
            assert len(df) == 2
            assert "activation_value" in df.columns
            assert df.iloc[0]["activation_value"] == pytest.approx(expected_act_val1)
            assert df.iloc[1]["activation_value"] == pytest.approx(expected_act_val2)
            assert df.iloc[0]["text"] == texts[0]

    def test_generate_synthetic_dataset_mocked(self, setup_generator):
        generator, _ = setup_generator
        num_samples = 5

        # Mock load_dataset from HuggingFace
        mock_hf_data = [
            {"text": "Sample text one for scanner."},
            {"text": "Another sample text here."},
            {"text": "This is the third sample."},
            {"text": "Yet another piece of text."},
            {"text": "Final sample text for this test."},
            {"text": "This one should be ignored due to num_samples limit."}
        ]
        
        # Mock the iterable dataset
        mock_dataset_iterable = MagicMock()
        mock_dataset_iterable.__iter__.return_value = iter(mock_hf_data)

        with patch('dataset.load_dataset', return_value=mock_dataset_iterable) as mock_load_ds, \
             patch.object(ActivationDatasetGenerator, 'generate_dataset_from_texts') as mock_gen_from_texts:
            
            # Make generate_dataset_from_texts return a dummy DataFrame
            dummy_df = pd.DataFrame({
                "text": [d["text"] for d in mock_hf_data[:num_samples]],
                "activation_value": np.random.rand(num_samples).tolist()
            })
            mock_gen_from_texts.return_value = dummy_df
            
            df_synthetic = generator.generate_synthetic_dataset(
                num_samples=num_samples,
                source_dataset_name="fake_dataset",
                source_dataset_config="fake_config",
                text_field="text",
                min_text_length=5 # All mock texts satisfy this
            )
            
            mock_load_ds.assert_called_once_with("fake_dataset", "fake_config", split='train', streaming=True)
            
            # Check that generate_dataset_from_texts was called with the first num_samples texts
            texts_passed_to_gen = mock_gen_from_texts.call_args[0][0]
            assert len(texts_passed_to_gen) == num_samples
            assert texts_passed_to_gen[0] == "Sample text one for scanner."
            
            assert df_synthetic.equals(dummy_df)

    def test_balance_dataset_simple(self, setup_generator):
        generator, _ = setup_generator
        # Case 1: Test where balancing should occur (nunique > num_bins)
        data_imbalanced = {
            "text": [f"text_{i}" for i in range(25)], # 5+15+5 = 25 samples
            # Introduce more unique values to ensure nunique > num_bins
            "activation_value": [0.1, 0.11, 0.12, 0.13, 0.14] * 1 + \
                                [0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54] * 1 + \
                                [0.8, 0.81, 0.82, 0.83, 0.84] * 1
        }
        df_imbalanced = pd.DataFrame(data_imbalanced)
        # Here, nunique will be 5+15+5 = 25. Let num_bins = 3. 25 > 3, so balancing should occur.
        
        balanced_df = generator.balance_dataset(df_imbalanced, column_to_balance="activation_value", num_bins=3)
        
        # After pd.cut with 3 bins, we find the smallest bin count.
        # Example: if bins are [0.1, 0.3, 0.6, 0.85], counts might be [5, 15, 5]. Smallest is 5.
        # So, expected length is 3 bins * 5 samples/bin = 15.
        # This depends on how pd.cut forms the bins with the new data.
        # Let's check the actual bin counts from the code's logic.
        # For this data, with 3 bins, it's likely to be 5 samples from each of the original groups if bins align.
        # If pd.cut creates bins like [0.1, ~0.35], [~0.35, ~0.65], [~0.65, 0.85]
        # Counts would be 5, 15, 5. Smallest is 5. Expected length 3*5 = 15.
        assert len(balanced_df) == 15, "Balancing did not result in the expected number of samples."
        
        # Check if counts per bin are roughly equal (or exactly equal to min_bin_size)
        # This requires re-binning the balanced_df to check.
        # For simplicity, we trust the groupby().apply(sample) logic for now.
        # A more rigorous test would check the distribution.
        
        # Test with already "balanced" or few unique values
        data_few_unique = {
             "text": [f"text_{i}" for i in range(6)],
             "activation_value": [0.1, 0.1, 0.5, 0.5, 0.9, 0.9]
        }
        df_few = pd.DataFrame(data_few_unique)
        # Expect it to return the original df due to nunique <= num_bins
        balanced_df_few = generator.balance_dataset(df_few, num_bins=3)
        assert len(balanced_df_few) == len(df_few)

        # Test with multi-dimensional (object type) activations - should return original
        data_obj = {
            "text": ["text1"], "activation_value": [[0.1, 0.2]]
        }
        df_obj = pd.DataFrame(data_obj)
        balanced_df_obj = generator.balance_dataset(df_obj)
        assert df_obj.equals(balanced_df_obj)

    # TODO: Add tests for _get_activations with a real (small) model if feasible,
    #       or more intricate mocking of run_with_hooks.
    # TODO: Test edge cases for token_position in _get_activations.
    # TODO: Test metadata saving/loading (if implemented more robustly).
