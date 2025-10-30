# tests/test_scanner.py
# pylint: disable=import-error
import pytest # type: ignore
import torch
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from scanner import NeuronScanner
from .conftest import small_test_model, device # Import fixtures

class TestNeuronScanner:

    @pytest.fixture
    def setup_scanner(self, small_test_model: HookedTransformer, device: str):
        """Fixture to set up NeuronScanner with a small model."""
        small_test_model.to(device)
        scanner = NeuronScanner(model=small_test_model, device=device)
        # Configure a dummy output for tests that might save visualizations
        scanner.configure_output({"output_dir": "tests/output/scanner_tests"})
        return scanner, small_test_model

    def test_scanner_initialization(self, setup_scanner):
        scanner, model = setup_scanner
        assert scanner.model == model
        assert scanner.device == model.cfg.device

    def test_create_hook_fn(self, setup_scanner):
        """Test the _create_hook_fn internal logic."""
        scanner, model = setup_scanner
        neuron_activations_map = {}
        layer_idx = 0
        target_token_idx = 2 # Example target token

        # Create a dummy activation tensor that the hook would receive
        # batch_size=1, seq_len=5, d_mlp=model.cfg.d_mlp
        dummy_activation_tensor = torch.randn(1, 5, model.cfg.d_mlp).to(scanner.device)
        # Set a specific, predictable value for a neuron we'll check
        expected_value = 1.2345
        target_neuron_for_test = 5
        dummy_activation_tensor[0, target_token_idx, target_neuron_for_test] = expected_value
        
        hook_fn = scanner._create_hook_fn(neuron_activations_map, layer_idx, target_token_idx)
        
        # Simulate the hook being called
        # The 'hook' argument to hook_fn is not used in its current implementation, so pass None or MagicMock
        hook_fn(dummy_activation_tensor, hook=None) 
        
        assert (layer_idx, target_neuron_for_test) in neuron_activations_map
        assert len(neuron_activations_map[(layer_idx, target_neuron_for_test)]) == 1
        assert neuron_activations_map[(layer_idx, target_neuron_for_test)][0] == pytest.approx(expected_value)
        # Check another neuron to ensure it was also captured
        assert (layer_idx, 0) in neuron_activations_map 
        assert neuron_activations_map[(layer_idx,0)][0] == pytest.approx(dummy_activation_tensor[0, target_token_idx, 0].item())


    @patch.object(NeuronScanner, '_create_hook_fn')
    def test_get_all_mlp_neuron_activations_for_texts_mocked_hook_creation(self, mock_create_hook, setup_scanner):
        """Test _get_all_mlp_neuron_activations_for_texts, mocking the hook creation part."""
        scanner, model = setup_scanner
        texts = ["text one", "another text"]
        
        # Mock the hook function that _create_hook_fn returns
        mock_hook_instance = MagicMock()
        mock_create_hook.return_value = mock_hook_instance

        # Mock run_with_hooks to verify it's called correctly
        with patch.object(model, 'run_with_hooks') as mock_run_with_hooks:
            scanner._get_all_mlp_neuron_activations_for_texts(texts, layers_to_scan=[0,1]) # Scan 2 layers

            # Expected calls to _create_hook_fn: len(texts) * num_layers_to_scan
            # No, _create_hook_fn is called once per layer inside the loop over texts.
            # The hook_fn it returns is then used for that layer for that text.
            # So, for each text, we create hooks for each layer.
            # The actual hook_fn (mock_hook_instance) will be called by run_with_hooks.
            
            # Check _create_hook_fn calls
            # For each text, it iterates layers_to_scan.
            # So, num_texts * num_layers_to_scan calls to _create_hook_fn
            # This is incorrect. _create_hook_fn is called inside the text loop, for each layer.
            # The list of hooks is [(hook_point, created_hook_fn), ...]
            # So, for each text, run_with_hooks is called once with a list of hooks.
            # Each element in that list of hooks is created by _create_hook_fn.
            
            # Expected calls to _create_hook_fn:
            # For "text one": layer 0, layer 1
            # For "another text": layer 0, layer 1
            # This seems wrong. The hook list is created *inside* the text loop.
            
            # Let's re-evaluate:
            # Outer loop: for text in texts:
            #   Inner loop: for layer_idx in layers_to_scan:
            #     hooks.append((..., scanner._create_hook_fn(...)))
            #   model.run_with_hooks(..., fwd_hooks=hooks)
            
            # So, _create_hook_fn is called num_layers_to_scan times *per text*.
            # Total calls to _create_hook_fn = len(texts) * len(layers_to_scan)
            assert mock_create_hook.call_count == len(texts) * 2 
            
            # Check run_with_hooks calls
            assert mock_run_with_hooks.call_count == len(texts)
            # For each call to run_with_hooks, fwd_hooks should have len(layers_to_scan) tuples
            for actual_call in mock_run_with_hooks.call_args_list:
                _, kwargs = actual_call
                assert 'fwd_hooks' in kwargs
                assert len(kwargs['fwd_hooks']) == 2 # layers_to_scan = [0,1]
                for hook_tuple in kwargs['fwd_hooks']:
                    assert hook_tuple[1] == mock_hook_instance # The created hook is the mocked one


    def test_calculate_neuron_statistics(self, setup_scanner):
        scanner, _ = setup_scanner
        neuron_activations_map = {
            (0, 0): [0.0, 0.0, 0.0, 1.0, 2.0, 3.0], # 50% zeros, var > 0
            (0, 1): [1.0, 1.0, 1.0, 1.0],          # 0% zeros, var = 0
            (1, 0): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0], # 90% zeros
            (1, 1): [] # Empty activations
        }
        stats_df = scanner.calculate_neuron_statistics(neuron_activations_map)
        
        assert len(stats_df) == 3 # Neuron (1,1) should be skipped
        
        neuron00_stats = stats_df[(stats_df['layer'] == 0) & (stats_df['neuron_index'] == 0)].iloc[0]
        assert neuron00_stats['percent_zeros'] == pytest.approx(50.0)
        assert neuron00_stats['percent_non_zeros'] == pytest.approx(50.0)
        assert neuron00_stats['variance'] == pytest.approx(np.var([0,0,0,1,2,3]))
        assert neuron00_stats['activation_range'] == pytest.approx(3.0)

        neuron01_stats = stats_df[(stats_df['layer'] == 0) & (stats_df['neuron_index'] == 1)].iloc[0]
        assert neuron01_stats['percent_zeros'] == pytest.approx(0.0)
        assert neuron01_stats['variance'] == pytest.approx(0.0)

        neuron10_stats = stats_df[(stats_df['layer'] == 1) & (stats_df['neuron_index'] == 0)].iloc[0]
        assert neuron10_stats['percent_zeros'] == pytest.approx(90.0)


    def test_score_and_select_neurons(self, setup_scanner):
        scanner, _ = setup_scanner
        # Create a dummy stats_df
        data = {
            'layer': [0, 0, 1, 1, 1],
            'neuron_index': [0, 1, 0, 1, 2],
            'variance': [1.0, 0.5, 2.0, 0.1, 1.5],
            'activation_range': [2.0, 1.0, 3.0, 0.5, 2.5],
            'percent_zeros': [30, 10, 50, 35, 40], # Neuron (0,1) fails this
            'percent_non_zeros': [70, 90, 50, 65, 60]
        }
        stats_df = pd.DataFrame(data)
        
        top_neurons = scanner.score_and_select_neurons(stats_df, top_n=2)
        
        assert len(top_neurons) == 2
        # Candidate neurons are (0,0), (1,0), (1,1), (1,2)
        # (0,1) is filtered out.
        # We need to check scores.
        # For (0,0): var=1, range=2. Assume norm_var, norm_range are calculated.
        # For (1,0): var=2, range=3. Highest var and range. Should be top.
        # For (1,2): var=1.5, range=2.5.
        # For (1,1): var=0.1, range=0.5. Lowest.
        
        # Check if (1,0) is the top neuron (it has highest variance and range among candidates)
        assert top_neurons.iloc[0]['layer'] == 1
        assert top_neurons.iloc[0]['neuron_index'] == 0
        
        # Test with no candidates
        data_no_candidates = data.copy()
        data_no_candidates['percent_zeros'] = [10,10,10,10,10]
        stats_df_no_cand = pd.DataFrame(data_no_candidates)
        top_no_cand = scanner.score_and_select_neurons(stats_df_no_cand)
        assert top_no_cand.empty

    # More comprehensive scan test might involve mocking _get_all_mlp_neuron_activations_for_texts
    # or running on the small_test_model if it's fast enough.
    @patch.object(NeuronScanner, '_get_all_mlp_neuron_activations_for_texts')
    def test_scan_main_method(self, mock_get_activations, setup_scanner):
        scanner, _ = setup_scanner
        texts = ["test1", "test2"]
        
        # Mock the output of _get_all_mlp_neuron_activations_for_texts
        mock_activations_map = {
            (0,0): [0.0, 1.0, 0.0, 2.0], # 50% zeros
            (0,1): [1.0, 1.0, 1.0, 1.0]  # 0% zeros
        }
        mock_get_activations.return_value = mock_activations_map
        
        top_neurons = scanner.scan(texts, top_n_to_display=1)
        
        mock_get_activations.assert_called_once()
        assert len(top_neurons) == 1
        assert top_neurons.iloc[0]['layer'] == 0
        assert top_neurons.iloc[0]['neuron_index'] == 0 # Only (0,0) meets sparsity criteria

    # TODO: Add tests for visualize_activation_distribution (e.g., check if plt.savefig is called)
    # TODO: Add Hypothesis tests for calculate_neuron_statistics and score_and_select_neurons
    #       with generated dataframes.
