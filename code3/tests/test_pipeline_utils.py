# tests/test_pipeline_utils.py
import pytest
import json
import os
import shutil
from unittest.mock import patch, mock_open

# Functions to test are in neuron_self_report.py
# Need to be careful about how they are imported if neuron_self_report.py also has a main execution block.
# For now, assume direct import is possible or PYTHONPATH is set.
# If neuron_self_report.py is run as a script, its functions might not be easily importable.
# A common practice is to put utility functions in a separate utils.py.
# Given the current structure, we'll try to import directly.
import neuron_self_report as pipeline_module 

class TestPipelineUtils:

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Creates a temporary config file for testing load_config."""
        config_dir = tmp_path / "config_tests"
        config_dir.mkdir()
        config_file = config_dir / "test_config.json"
        
        test_data = {
            "model_name": "test_model_override",
            "learning_rate": 0.01,
            "custom_param": "custom_value"
        }
        with open(config_file, 'w') as f:
            json.dump(test_data, f)
        return str(config_file), test_data

    def test_load_config_file_found(self, temp_config_file):
        config_path, expected_user_data = temp_config_file
        
        # Get a copy of default config to check against
        default_config_copy = pipeline_module.DEFAULT_CONFIG.copy()
        
        loaded_config = pipeline_module.load_config(config_path)
        
        # Check that defaults are present
        for key, value in default_config_copy.items():
            if key not in expected_user_data: # Default should be used if not overridden
                assert loaded_config.get(key) == value
        
        # Check that user overrides are applied
        assert loaded_config["model_name"] == expected_user_data["model_name"]
        assert loaded_config["learning_rate"] == expected_user_data["learning_rate"]
        assert loaded_config["custom_param"] == expected_user_data["custom_param"]
        # Check a default value that wasn't overridden
        assert loaded_config["epochs"] == pipeline_module.DEFAULT_CONFIG["epochs"]


    def test_load_config_file_not_found(self, tmp_path):
        non_existent_path = str(tmp_path / "non_existent_config.json")
        loaded_config = pipeline_module.load_config(non_existent_path)
        # Should return a copy of DEFAULT_CONFIG
        assert loaded_config == pipeline_module.DEFAULT_CONFIG

    def test_load_config_invalid_json(self, tmp_path):
        invalid_json_file = tmp_path / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write("{'model_name': 'bad_json',") # Invalid JSON (single quotes, trailing comma)
        
        loaded_config = pipeline_module.load_config(str(invalid_json_file))
        # Should return a copy of DEFAULT_CONFIG
        assert loaded_config == pipeline_module.DEFAULT_CONFIG

    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_determine_device(self, mock_cuda_available, mock_mps_available):
        # Test MPS preference
        mock_mps_available.return_value = True
        mock_cuda_available.return_value = True # Even if CUDA is available, MPS should be chosen if available
        assert pipeline_module.determine_device(None) == "mps"

        # Test CUDA preference if MPS not available
        mock_mps_available.return_value = False
        mock_cuda_available.return_value = True
        assert pipeline_module.determine_device(None) == "cuda"

        # Test CPU fallback
        mock_mps_available.return_value = False
        mock_cuda_available.return_value = False
        assert pipeline_module.determine_device(None) == "cpu"

        # Test user override
        mock_mps_available.return_value = True 
        mock_cuda_available.return_value = True # Both MPS and CUDA are "available" via mocks
        assert pipeline_module.determine_device("cpu") == "cpu" # User wants CPU
        
        # User wants CUDA, and mock says it's available
        mock_mps_available.return_value = False # MPS not available for this sub-case
        mock_cuda_available.return_value = True 
        assert pipeline_module.determine_device("cuda") == "cuda" 
        
        # Test user override when requested device is not available
        mock_mps_available.return_value = True # MPS is available
        mock_cuda_available.return_value = False # CUDA is NOT available
        assert pipeline_module.determine_device("cuda") == "cpu" # Fallback, CUDA not available

        mock_mps_available.return_value = False # MPS is NOT available
        mock_cuda_available.return_value = True # CUDA is available
        assert pipeline_module.determine_device("mps") == "cpu" # Fallback, MPS not available


    def test_setup_output_directory(self, tmp_path):
        test_output_dir_name = "test_run_output"
        config = {
            "output_dir": str(tmp_path / test_output_dir_name),
            # Add other minimal necessary config items if setup_output_directory uses them
            "model_name": "test_model", "neuron_layer":0, "neuron_index":0 
        }

        # First run: creates the directory
        output_dir_path = pipeline_module.setup_output_directory(config)
        assert os.path.exists(output_dir_path)
        assert os.path.exists(os.path.join(output_dir_path, "run_config.json"))

        # Second run: archives the first run and creates a new one
        # Modify config slightly for a "new" run, though setup_output_directory doesn't use these for naming archive
        config["learning_rate"] = 0.002 
        new_output_dir_path = pipeline_module.setup_output_directory(config)
        assert os.path.exists(new_output_dir_path)
        assert os.path.exists(os.path.join(new_output_dir_path, "run_config.json"))

        # Check if archive directory exists and contains the first run
        archive_parent = tmp_path / "previous-outputs"
        assert os.path.exists(archive_parent)
        archived_runs = os.listdir(archive_parent)
        assert len(archived_runs) == 1
        assert test_output_dir_name in archived_runs[0] # Check if original name is part of archive
        assert os.path.exists(os.path.join(archive_parent, archived_runs[0], "run_config.json"))


    # TODO: Tests for target processing logic (may require refactoring from main_pipeline)
    # e.g., test_process_regression_targets, test_process_classification_targets, etc.
    # These would take activation_values (np.array) and config, return targets_tensor.
