# Neuron Activation Projection Analysis Plan

## Overview

This notebook will analyze the relationship between ground truth neuron activations and four projection directions:
1. The prediction head direction
2. The neuron input weight direction
3. The neuron output weight direction
4. The final residual stream

Given an output directory from a previous run, the notebook will load model outputs and run a correlation analysis between these different directions.

## Implementation Plan

### 1. Setup and Imports
- Import libraries (torch, numpy, matplotlib, pandas, etc.)
- Set up utility functions for loading data from output directories

### 2. Data Loading
- Parse the output directory path to identify target layer and neuron
- Load the dataset containing input texts and ground truth activations
- Load the trained model head from the model directory
- Load the base transformer model (gpt2) to extract neuron weights

### 3. Extract Weight Directions
- Extract the linear prediction weights from the trained head
- Extract the MLP input weights for the target neuron
- Extract the MLP output weights for the target neuron
- Normalize all three weight vectors to unit length

### 4. Run Input Texts Through Models
- For each input text in the dataset:
  - Get the hidden states from the model at multiple points:
    - Residual stream immediately before the MLP layer (for input weight projection)
    - Residual stream immediately after the MLP layer (for output weight projection)
    - Final layer residual stream (for final output projection)
  - Calculate the projection of each hidden state onto its respective weight direction:
    - Project pre-MLP hidden state onto input weights
    - Project post-MLP hidden state onto output weights
    - Project final hidden state onto head weights
    - Project final hidden state onto normalized "neuron direction" (for comparison)
  - Store these projections along with the ground truth activation

### 5. Correlation Analysis
- Calculate Pearson correlation between:
  - Ground truth and prediction head projection
  - Ground truth and input weight projection
  - Ground truth and output weight projection
  - Ground truth and final residual stream neuron direction projection
- Calculate coefficient of determination (R²) for each relationship
- Generate a correlation matrix showing relationships between all projections
- Perform statistical significance testing for the correlations

### 6. Visualization
- Create a 4-panel scatter plot showing:
  1. Ground truth activations vs. head weight projections
  2. Ground truth activations vs. input weight projections
  3. Ground truth activations vs. output weight projections
  4. Ground truth activations vs. final residual stream neuron direction projections
- Include correlation values, p-values, and regression lines in each panel
- Add marginal histograms to show the distribution of each variable
- Create a correlation heatmap showing relationships between all variables
- Add a title showing neuron and layer information

### 7. Save and Report
- Save the visualization to the output directory
- Compute and display a summary of the findings
- Save the correlation results to a JSON file

## Technical Considerations

- Use transformer_lens for accessing model internals and extracting hidden states
- Use the existing utility functions from linear_weights_analysis.ipynb where applicable
- Perform projections in two ways:
  - Primary analysis: Use raw (non-normalized) weights to preserve magnitude information and maintain consistency with the rest of the project
  - Secondary analysis: Calculate cosine similarities and normalized projections to isolate directional effects
  - Compare both approaches to see if magnitude information contributes to the correlations
- Use appropriate hooks with transformer_lens to extract activations at precise points:
  - Pre-MLP hook for input weight projections
  - Post-MLP hook for output weight projections
  - Final layer hook for endpoint analysis
- Handle potential numerical issues:
  - Check for NaN/Inf values
  - Apply appropriate scaling for numerical stability
- Use a sufficient sample size from the dataset for reliable correlation analysis
- Handle potential errors and edge cases gracefully
- Include documentation and comments for clarity

## Expected Outcomes

This analysis will help us understand:
1. Whether the regression head is actually learning the neuron's input weights, output weights, or a different direction entirely
2. If the neuron's activation is directly reflected in the final residual stream
3. Which representation space best captures the neuron's behavior
4. Potential mechanisms for how the model is making its predictions