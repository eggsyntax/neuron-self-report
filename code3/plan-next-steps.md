Okay, here's a list of the 10 most important next tasks to build upon the current foundation:

1.  **Implement Robust Token ID Mapping for Token-Based Targets:**
    *   **Task:** In `neuron_self_report.py` (target processing section) and potentially `ActivationDatasetGenerator`, ensure that the mapping of activation values to "on"/"off" or "0"-"9" token IDs is robust. This includes handling cases where these specific string tokens (e.g., " on", " 0") might not exist in a model's vocabulary or might tokenize into multiple sub-tokens.
    *   **Importance:** Critical for the token-based prediction experiments to function correctly and reliably across different models.
    *   **Action:** Add logic to check if the target strings tokenize to single IDs. If not, either raise an informative error or allow configuration of alternative token strings/IDs. Update `DEFAULT_CONFIG` for `token_on_id`, `token_off_id`, `token_digit_ids` to be populated dynamically based on the loaded model's tokenizer if not user-specified.

2.  **Detailed Evaluation Metrics in `PredictorTrainer`:**
    *   **Task:** Implement more comprehensive evaluation metrics in `PredictorTrainer.evaluate()`.
        *   For "regression": Mean Squared Error (MSE), Pearson Correlation.
        *   For "classification": Accuracy, Precision, Recall, F1-score (per class and macro/micro averaged).
        *   For "token_binary" / "token_digit": Accuracy (based on predicting the correct target token ID).
    *   **Importance:** Essential for quantitatively assessing model performance for different prediction tasks.
    *   **Action:** Modify `evaluate()` to calculate and return these metrics. Update `history` tracking and `generate_training_visualizations` to include plots for these new metrics.

3.  **Save and Load Best Model Checkpoints:**
    *   **Task:** Implement functionality in `PredictorTrainer.train()` to save the state dictionary of the `ActivationPredictor` model when validation loss improves (or based on another key metric). Add a mechanism (perhaps in `neuron_self_report.py` or as a utility) to load these checkpoints for later analysis or inference.
    *   **Importance:** Allows for retaining the best performing models without retraining and facilitates further analysis.
    *   **Action:** Use `torch.save()` in the trainer and add a corresponding loading function/script.

4.  **Implement Activation Distribution Monitoring in `PredictorTrainer`:**
    *   **Task:** Fulfill the TODO for `monitor_activation_distributions` in `PredictorTrainer`. If parts of the base model are unfrozen, track how the distribution (mean, variance, sparsity) of the *target neuron's actual activations* changes over epochs on a fixed validation set.
    *   **Importance:** Key for the "Self-Modification Testing" experiment to see if the model learns to regularize the target neuron.
    *   **Action:** Add a method that runs a validation set through the base model (with hooks for the target neuron) at intervals and logs/visualizes these statistics.

5.  **Refine `NeuronScanner` Visualization and Selection:**
    *   **Task:**
        *   Ensure `NeuronScanner.visualize_activation_distribution` can be easily called from the main pipeline for the *selected* neuron after a scan. This might involve `scanner.scan()` returning the `neuron_activations_map` or the specific activations for top neurons.
        *   Implement the TODO for interactive neuron selection in `neuron_self_report.py` if `scanner_auto_select_top_neuron` is `False`.
    *   **Importance:** Improves usability and insight during the neuron selection phase.
    *   **Action:** Adjust `NeuronScanner.scan()` return values or add helper methods. Add `input()` prompt logic in `main_pipeline`.

6.  **Expand Unit Test Coverage (especially with Hypothesis):**
    *   **Task:**
        *   Add `hypothesis` tests for functions like `calculate_neuron_statistics`, `score_and_select_neurons` in `test_scanner.py`.
        *   Test edge cases for target processing in `test_pipeline_utils.py` (once refactored or by testing `main_pipeline`'s data prep part).
        *   Test different unfreeze strategies in `test_trainer.py` by checking `requires_grad` status of parameters more thoroughly.
        *   Test the actual forward pass logic of `ActivationPredictor` for regression/classification heads more directly by better mocking the hook mechanism or refactoring feature extraction.
    *   **Importance:** Increases confidence in the robustness and correctness of the code.
    *   **Action:** Write new test functions using `hypothesis` strategies and add more detailed assertion checks.

7.  **Refactor Target Processing Logic:**
    *   **Task:** Move the target processing logic (for classification, token_binary, token_digit) from `main_pipeline` in `neuron_self_report.py` into `ActivationDatasetGenerator` or a dedicated utility module.
    *   **Importance:** Improves code organization, makes `main_pipeline` cleaner, and makes target processing more easily testable.
    *   **Action:** Create new methods in `ActivationDatasetGenerator` or a `utils.py` and update `main_pipeline` to call them. Write unit tests for these new methods.

8.  **Begin Mechanistic Interpretability Scripts (Notebooks):**
    *   **Task:** Start creating the first Jupyter notebook, e.g., for "Linear Weights Analysis" as outlined in `plan.md`. This would involve loading a trained `ActivationPredictor` (especially a linear probe) and comparing its weights to something meaningful from the base model.
    *   **Importance:** Starts work on the core analysis goals of the project.
    *   **Action:** Create `mech_interp-notebooks/linear_weights_analysis.ipynb` and implement initial loading and comparison logic.

9.  **More Sophisticated Dataset Balancing:**
    *   **Task:** Review and potentially enhance the `balance_dataset` method in `ActivationDatasetGenerator`. The current method is simple. Consider strategies that might be more effective for highly skewed activation data or for ensuring representation of rare activation events if needed.
    *   **Importance:** High-quality datasets are crucial for training effective probes.
    *   **Action:** Research and implement more advanced balancing techniques if deemed necessary after initial experiments.

10. **End-to-End Integration Tests/Example Runs:**
    *   **Task:** Create a few example `config.json` files that test different full pipeline configurations (e.g., regression head, token_binary head with scanner enabled, classification head with provided dataset). Run these and ensure outputs (CSVs, plots, model files if saving is implemented) are generated correctly.
    *   **Importance:** Verifies that all components work together as expected.
    *   **Action:** Create diverse config files and manually run/document the pipeline for these cases. This can also form the basis for automated integration tests later.
