# Next Session Plan for Neural Network Introspection System

This document provides instructions and context for a fresh instance of Cline to continue development on the Neural Network Introspection System.

## I. Project Status Summary

*   **Overall Goal:** Train Language Models (LLMs) to self-report or predict their internal neuron activation values.
*   **Current State (as of end of previous session):**
    *   Core Python modules (`dataset.py`, `scanner.py`, `architecture.py`, `trainer.py`, `neuron_self_report.py`) have been scaffolded with initial functional implementations.
    *   Configuration is centralized in `config.json`. `neuron_self_report.py` loads this file directly, and no longer contains an in-script `DEFAULT_CONFIG`.
    *   A unit test suite has been established in the `tests/` directory using `pytest`. As of the last check, all 29 tests were passing.
    *   Basic training loss visualization (loss vs. epoch plot saved to output directory) is implemented in `PredictorTrainer`.
    *   Target processing logic for all specified output types (`regression`, `classification`, `token_binary`, `token_digit`) has been implemented in `neuron_self_report.py`.
    *   `NeuronScanner` functionality is integrated into the main pipeline, controllable via `config.json`.
    *   Regression target values are normalized (zero mean, unit variance) in `neuron_self_report.py` to aid training stability.
    *   The default learning rate in `config.json` is `1e-4`.
*   **Key Documents for Reference:**
    *   `plan.md`: The original detailed project plan, updated with clarifications from early discussions.
    *   `plan-next-steps.md`: A list of 10 prioritized next tasks compiled towards the end of the previous session.
    *   `README.md`: Contains a basic project overview and setup instructions.
    *   `config.json`: The central configuration file. All default values should reside here.

## II. Last Known Issue / Point of Interruption

*   **Issue:** The user reported an `Abort trap: 6` with an MPSNDArray error: `failed assertion '[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: total bytes of NDArray > 2**32'` during a run.
    *   This occurred when testing with neuron L8N481, 500 samples, 10 epochs, on an MPS (Apple Silicon GPU) device.
    *   The error message points to an attempt to allocate an MPSNDArray exceeding 4GB.
*   **Error Log Snippet:**
    ```
    Epoch 1/10
    Training Epoch:   0%|                                                              | 0/12 [00:00<?, ?it/s]
    /AppleInternal/Library/BuildRoots/d187755d-b9a3-11ef-83e5-aabfac210453/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShaders/MPSCore/Types/MPSNDArray.mm:850: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: total bytes of NDArray > 2**32'
    Abort trap: 6
    ```
*   **Hypothesis:** This is likely an MPS-specific memory allocation issue, possibly related to tensor creation, sizing, or device transfer during data loading/processing, model forward pass, or gradient computation, especially given the batch size and model dimensions. The error occurs very early in the first training epoch.

## III. Instructions for Next Session (for a fresh instance of Cline)

1.  **Familiarize with Existing Plans & Code:**
    *   Thoroughly review `plan.md` for the overall project architecture, component responsibilities, and long-term goals.
    *   Review `plan-next-steps.md` for the previously identified list of upcoming development tasks.
    *   Review this document (`next-session-plan.md`) for immediate context and priorities.
    *   Browse the existing codebase (`dataset.py`, `scanner.py`, `architecture.py`, `trainer.py`, `neuron_self_report.py`) to understand current implementations.

2.  **CRITICAL: Investigate and Resolve the MPSNDArray Error:**
    *   **Priority:** This is the absolute first priority. The pipeline is not reliably usable on MPS devices until this is fixed.
    *   **Reproduction:**
        *   Modify `config.json` to match the user's failing setup:
            *   `"model_name": "gpt2-small"` (or as specified by user for L8N481 context)
            *   `"neuron_layer": 8`
            *   `"neuron_index": 481`
            *   `"num_samples": 500`
            *   `"epochs": 10`
            *   Ensure `"device": null` or `"device": "mps"` to target the MPS device.
        *   Run `python neuron_self_report.py --config config.json`.
    *   **Debugging Steps:**
        *   If reproducible, the error occurs at the very start of the first training epoch. This points to an issue in the data loading/batching part of `PredictorTrainer.train_epoch`, or the first forward pass of the model with the first batch.
        *   **Reduce Batch Size:** Try significantly reducing `"batch_size"` in `config.json` (e.g., to 1 or 2) to see if the error is sensitive to batch memory.
        *   **Inspect Tensor Sizes:** Carefully examine all tensor creation and manipulation steps, especially those involving `.to(device_str)`:
            *   In `neuron_self_report.py` during `all_input_ids` creation and padding. The `max_len` (default 512) * batch_size * embedding_dim could be large.
            *   Within `ActivationDatasetGenerator` if it's involved before the error.
            *   Within `ActivationPredictor.forward`.
            *   Within `PredictorTrainer.train_epoch` when moving batch data to device.
        *   The error `total bytes of NDArray > 2**32` (4GB) is a strong clue. Identify which tensor allocation is causing this. Print shapes and dtypes of tensors being moved to MPS or created on MPS.
        *   Consider if any large intermediate tensors (e.g., full vocabulary logits for all sequence positions if not handled carefully) are being unnecessarily kept in memory or processed on the MPS device.
    *   **Goal:** Identify the specific tensor and operation causing the excessive memory allocation on MPS and implement a fix. This might involve changing how data is batched, processed, or ensuring only necessary data is on the MPS device at any time.

3.  **Verify Test Suite:**
    *   After resolving the MPS error (or if working on a non-MPS device to bypass the issue temporarily for other development), run `pytest`.
    *   Address any new test failures that might have been introduced or uncovered.

4.  **Re-evaluate "Bouncing Loss" (Post-MPS Fix):**
    *   **Context:** The user previously observed that for neuron L8N481, the training/validation loss was "bouncing around" rather than clearly decreasing. They confirmed that this neuron *should* be learnable with a linear probe.
    *   **Current Mitigations:**
        *   Regression targets are now normalized in `neuron_self_report.py`.
        *   Default learning rate in `config.json` is `1e-4`.
    *   **Further Steps (if bouncing persists after MPS fix and with the above mitigations):**
        *   Experiment with even lower learning rates (e.g., `5e-5`, `1e-5`) in `config.json`.
        *   Consider adding basic input feature normalization for the features fed into the regression/classification head in `ActivationPredictor.forward`. This would involve standardizing `token_features`.
        *   Log the mean, std, min, max of `token_features` before they are passed to the head to understand their scale and distribution.
        *   If L8N481 remains difficult, try a different neuron (perhaps one identified by `NeuronScanner` if it was run successfully on a CPU or after the MPS fix) to isolate if the issue is general to the training setup or specific to that neuron's characteristics with the current feature extraction.

5.  **Proceed with Tasks from `plan-next-steps.md`:**
    *   Once the MPS error is resolved and there's reasonable confidence in the basic learning setup (i.e., loss shows some trend), consult `plan-next-steps.md`.
    *   Based on that list and current project needs, "Detailed Evaluation Metrics in `PredictorTrainer`" (Task 2 from the list) or "Implement Robust Token ID Mapping for Token-Based Targets" (Task 1) are strong candidates for the next development focus.

## IV. Important Notes for the Next Instance of Cline

*   **Pylint/Pylance `pytest` import errors:** These are known static analysis quirks. If `pytest` is installed in the environment, these can generally be ignored. Comments have been added to attempt to silence them.
*   **`wandb` Pylance errors:** Similar to `pytest`, these are likely static analysis issues with the optional `wandb` import. The runtime code correctly guards against `wandb` usage if it's not available or not enabled.
*   **Configuration (`config.json`):** This is now the single source of truth for all run parameters and defaults.
*   **User's Primary Environment:** macOS with an MPS (Apple Silicon) GPU. MPS-specific issues are a priority.
*   **Code Style & Structure:** Prioritize readability and clarity, even if it means some minor code duplication, as per earlier discussions.

This plan should provide a solid starting point for the next session.
