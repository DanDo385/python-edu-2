"""
Project 22: Activation Functions and Neural Network Basics

Complete the functions below according to their docstrings.
Run pytest test.py -v to test your solutions.
"""

import numpy as np


def relu(x):
    """
    Rectified Linear Unit (ReLU) activation.
    
    ReLU(x) = max(0, x)
    
    Most popular activation in deep learning.
    Simple, fast, and helps with vanishing gradient problem.
    
    Args:
        x (np.ndarray): Input values
    
    Returns:
        np.ndarray: ReLU output, same shape as input
    """
    # TODO: Implement ReLU
    return None


def relu_derivative(x):
    """
    Derivative of ReLU function.
    
    dReLU/dx = 1 if x > 0, else 0
    
    Used in backpropagation to compute gradients.
    
    Args:
        x (np.ndarray): Input values
    
    Returns:
        np.ndarray: Derivative values, same shape as input
    """
    # TODO: Implement ReLU derivative
    return None


def tanh(x):
    """
    Hyperbolic tangent activation.
    
    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Output range: (-1, 1)
    Zero-centered (unlike sigmoid).
    
    Args:
        x (np.ndarray): Input values
    
    Returns:
        np.ndarray: Tanh output, same shape as input
    """
    # TODO: Implement tanh
    # Hint: Use np.tanh() or implement manually
    return None


def softmax(x):
    """
    Softmax activation function.
    
    Converts logits to probabilities.
    softmax(x)[i] = exp(x[i]) / sum(exp(x))
    
    Output sums to 1 (probability distribution).
    Used for multi-class classification.
    
    Args:
        x (np.ndarray): Input logits, shape (n,) or (batch, n)
    
    Returns:
        np.ndarray: Probabilities, same shape as input
    """
    # TODO: Implement softmax
    # Hint: Subtract max for numerical stability: exp(x - max(x))
    return None


def forward_pass_simple_network(x, w1, b1, w2, b2):
    """
    Forward pass through a simple 2-layer network.
    
    Layer 1: h = ReLU(w1*x + b1)
    Layer 2: y = w2*h + b2
    
    Args:
        x (np.ndarray): Input, shape (n,)
        w1 (float): Weight for layer 1
        b1 (float): Bias for layer 1
        w2 (float): Weight for layer 2
        b2 (float): Bias for layer 2
    
    Returns:
        np.ndarray: Network output
    """
    # TODO: Implement forward pass
    # Layer 1: linear transformation + ReLU
    # Layer 2: linear transformation (no activation)
    return None


