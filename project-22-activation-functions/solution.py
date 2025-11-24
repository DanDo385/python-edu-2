"""
Project 22: Activation Functions and Neural Network Basics - SOLUTION

Complete solution with detailed comments explaining activation functions
and why they're essential for neural networks.
"""

import numpy as np


def relu(x):
    """
    Rectified Linear Unit (ReLU) activation.
    
    ReLU is the most popular activation function in deep learning.
    It's simple, fast, and helps address the vanishing gradient problem.
    
    Formula: ReLU(x) = max(0, x)
    
    Properties:
    - Output: [0, ∞) - unbounded above, bounded below at 0
    - Simple: Just threshold at 0
    - Fast: No expensive exponentials
    - No saturation: For x > 0, gradient is always 1 (no vanishing gradients!)
    - Dead neurons: For x < 0, output is 0 (neuron never activates)
    
    Why ReLU is popular:
    - Solves vanishing gradient problem (for positive inputs)
    - Computationally efficient
    - Empirically works well in practice
    
    Args:
        x (np.ndarray): Input values
    
    Returns:
        np.ndarray: ReLU output, same shape as input
    
    Example:
        relu([-2, -1, 0, 1, 2])  # Returns [0, 0, 0, 1, 2]
    """
    # ReLU: max(0, x) element-wise
    # np.maximum compares element-wise and returns maximum
    # For each element: if x > 0, return x; else return 0
    return np.maximum(0, x)
    
    # Alternative: x * (x > 0) also works (uses boolean indexing)
    # But np.maximum is clearer and more efficient


def relu_derivative(x):
    """
    Derivative of ReLU function.
    
    Used in backpropagation to compute gradients.
    The derivative tells us how the output changes with input.
    
    Formula:
    dReLU/dx = 1 if x > 0
            = 0 if x <= 0
    
    Note: At x = 0, derivative is technically undefined, but we use 0.
    
    Properties:
    - Constant gradient (1) for positive inputs (good for learning!)
    - Zero gradient for negative inputs (dead neurons don't learn)
    
    Args:
        x (np.ndarray): Input values
    
    Returns:
        np.ndarray: Derivative values, same shape as input
    
    Example:
        relu_derivative([-2, -1, 0, 1, 2])  # Returns [0, 0, 0, 1, 1]
    """
    # Derivative is 1 where x > 0, 0 otherwise
    # (x > 0) creates boolean array, converting to int gives 0 or 1
    return (x > 0).astype(float)
    
    # This is used in backpropagation:
    # gradient = relu_derivative(x) * upstream_gradient


def tanh(x):
    """
    Hyperbolic tangent activation.
    
    Tanh is like sigmoid but output range is (-1, 1) instead of (0, 1).
    This zero-centering can help with training.
    
    Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Properties:
    - Output range: (-1, 1) - zero-centered (unlike sigmoid)
    - Smooth and differentiable
    - S-shaped curve (like sigmoid)
    - Still saturates at extremes (gradients can vanish)
    
    Comparison to sigmoid:
    - Tanh: zero-centered, range (-1, 1)
    - Sigmoid: not zero-centered, range (0, 1)
    - Zero-centering can help with gradient flow
    
    Args:
        x (np.ndarray): Input values
    
    Returns:
        np.ndarray: Tanh output, same shape as input
    
    Example:
        tanh(0)   # Returns 0.0
        tanh(10)  # Returns ~1.0
        tanh(-10) # Returns ~-1.0
    """
    # NumPy provides optimized tanh implementation
    return np.tanh(x)
    
    # Manual implementation (for understanding):
    # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    # But np.tanh() is faster and more numerically stable


def softmax(x):
    """
    Softmax activation function.
    
    Softmax converts logits (raw scores) to probabilities.
    It's used in the output layer for multi-class classification.
    
    Formula: softmax(x)[i] = exp(x[i]) / sum(exp(x))
    
    Properties:
    - Output sums to 1 (probability distribution)
    - All outputs are positive (probabilities)
    - Largest input gets highest probability
    - Smooth and differentiable
    
    Why softmax?
    - Converts arbitrary scores to valid probabilities
    - Largest score gets highest probability (makes sense!)
    - Used with cross-entropy loss for classification
    
    Numerical stability:
    - Subtracting max before exp prevents overflow
    - Doesn't change result (softmax is shift-invariant)
    
    Args:
        x (np.ndarray): Input logits, shape (n,) or (batch, n)
    
    Returns:
        np.ndarray: Probabilities, same shape as input
    
    Example:
        x = [1, 2, 3]
        softmax(x)  # Returns probabilities that sum to 1
        # Largest value (3) gets highest probability
    """
    # For numerical stability, subtract max before exponentiating
    # This doesn't change the result (softmax is shift-invariant)
    # But prevents exp(large_number) from overflowing
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    
    # Compute exponentials
    exp_x = np.exp(x_shifted)
    
    # Normalize by sum to get probabilities
    # Sum along last axis (classes dimension)
    probabilities = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    return probabilities
    
    # This is used in neural networks for multi-class classification:
    # logits = model(input)  # Raw scores
    # probabilities = softmax(logits)  # Convert to probabilities
    # prediction = argmax(probabilities)  # Choose class with highest probability


def forward_pass_simple_network(x, w1, b1, w2, b2):
    """
    Forward pass through a simple 2-layer network.
    
    This demonstrates how activation functions enable non-linearity.
    Without ReLU, this would just be a linear transformation.
    
    Architecture:
    Layer 1: h = ReLU(w1*x + b1)  (hidden layer with ReLU)
    Layer 2: y = w2*h + b2        (output layer, linear)
    
    Why ReLU in layer 1?
    - Without it: y = w2*(w1*x + b1) + b2 = (w2*w1)*x + constant
    - This is still linear! Multiple layers = single layer.
    - With ReLU: Non-linear transformation, can learn complex patterns!
    
    Args:
        x (np.ndarray): Input, shape (n,)
        w1 (float): Weight for layer 1
        b1 (float): Bias for layer 1
        w2 (float): Weight for layer 2
        b2 (float): Bias for layer 2
    
    Returns:
        np.ndarray: Network output
    
    Example:
        x = [1, 2]
        w1, b1, w2, b2 = 1.0, 0.0, 1.0, 0.0
        forward_pass_simple_network(x, w1, b1, w2, b2)
    """
    # Layer 1: Linear transformation + ReLU activation
    # This introduces non-linearity
    z1 = w1 * x + b1  # Linear combination
    h = relu(z1)      # Non-linear activation
    
    # Layer 2: Linear transformation (no activation)
    # This is the output layer
    y = w2 * h + b2
    
    return y
    
    # This simple network can learn non-linear patterns!
    # The ReLU activation is what makes it powerful.
    # Without ReLU, it would just be: y = (w2*w1)*x + (w2*b1 + b2)
    # Which is still linear (useless for complex problems).


