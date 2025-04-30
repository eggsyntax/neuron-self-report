# Understanding Neural Network Projections: A Beginner's Guide

Neural networks are complex systems with millions of parameters, making it difficult to understand what's happening inside them. This notebook helps us peek inside by examining how a specific neuron relates to different mathematical directions in the network.

## What is a Projection?

In this context, a "projection" is like measuring how much a point extends in a particular direction. Imagine you're standing in a room and measuring the shadow a stick casts in different directions when you shine a light on it. The length of the shadow tells you how much the stick "projects" in that direction.

In neural networks, we're measuring how much an activation pattern aligns with specific weight directions. This helps us understand what the network is "paying attention to" or "looking for" when it processes information.

## Comparing a Scalar (Neuron Activation) with Vectors

It's important to understand a key point: we're comparing a single number (the neuron's activation value) with projections calculated from high-dimensional vectors (768 dimensions in our GPT-2 model).

Here's how this works:

1. Each input text creates a **hidden state** in the model - a 768-dimensional vector that represents what the model "thinks" at that point.

2. The **neuron's activation** is a single scalar value (just one number) that represents how strongly this particular neuron fired for that input.

3. To compare these different objects, we **project** the hidden state onto each of our directions of interest. This means we calculate the dot product between the hidden state and a weight vector, which gives us a single number.

4. Now we can directly compare: for each input text, we have the neuron's activation (a number) and the projection value for each direction (also a number).

5. We calculate the correlation between these sets of numbers across many input texts to see how closely they relate.

## The Four Directions We're Examining

Our notebook analyzes four important directions:

1. **Prediction Head Direction** (head_projection): These are the weights learned by our external prediction model that tries to guess what the neuron will do. This direction shows what features in the final layer's representation are most useful for predicting the neuron's activation.

2. **Neuron Input Weight Direction** (input_projection): These are the weights that connect the neuron to the previous layer. They determine what patterns make the neuron activate. Think of these as the "features" the neuron is looking for.

3. **Neuron Output Weight Direction** (output_projection): These are the weights that connect the neuron to the next layer. They show how the neuron's activation affects the rest of the network. These represent the neuron's "influence" on the network.

4. **Final Residual Stream Direction** (neuron_dir_projection): This measures whether the neuron's information persists all the way to the end of the network in its original form, or if it gets transformed along the way.

Each of these directions is a 768-dimensional vector that represents a specific "direction" in the network's representation space.

## How to Interpret the Results

The notebook calculates correlations between the neuron's actual activation (ground truth) and each of these projections. These correlations tell us:

- **High correlation with input weights**: The neuron's activation directly reflects information already present in the network before the neuron activates. The neuron is primarily "reading" existing information.

- **High correlation with output weights**: The neuron's information gets clearly expressed in the next layer. The neuron is effectively "writing" its information to the network.

- **High correlation with prediction head**: Our external prediction model has successfully learned to track what the neuron is doing, possibly by learning one of the other directions.

- **High correlation with final layer direction**: The neuron's information persists throughout the network without major transformations.

The visualization shows scatter plots where each point represents an input text. If points form a straight line, there's a strong relationship; if they're scattered randomly, there's little relationship.

The correlation matrix shows how all these directions relate to each other. If two directions have high correlation, they might be representing similar information.

## Why This Matters

This analysis helps us answer fundamental questions about what computation is happening inside the neural network:

1. Is our regression head actually learning to approximate the neuron's input weights, output weights, or something else entirely?

2. Does the neuron's activation directly influence the final output, or is its information transformed?

3. Which representation space best captures what this neuron is doing?

4. What is the mechanism by which the model makes its predictions?

By understanding these relationships, we can gain insights into how information flows through the neural network and how different components relate to each other. This kind of analysis is a key part of "mechanistic interpretability" - the effort to understand exactly how neural networks process information.

The goal isn't just to know that the network works, but to understand *how* it works by tracing the flow of information through specific neurons and weights. This analysis gives us a window into the inner workings of what is otherwise a complex black box.