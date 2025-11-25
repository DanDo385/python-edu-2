"""
Project 21: Logistic Regression for Classification

Complete the functions below according to their docstrings.
Run pytest test.py -v to test your solutions.
"""

import numpy as np


def sigmoid(x):
    """
    Compute sigmoid (logistic) function.
    
    sigmoid(x) = 1 / (1 + exp(-x))
    
    Maps any real number to (0, 1) range.
    Used to convert linear outputs to probabilities.
    
    Args:
        x (np.ndarray or float): Input value(s)
    
    Returns:
        np.ndarray or float: Sigmoid output(s), same shape as input
    """
    # TODO: Implement sigmoid function
    # Hint: Use np.exp() for exponential
    return None


def predict_proba(x, w, b):
    """
    Predict probability of class 1 using logistic regression.
    
    p = sigmoid(w*x + b)
    
    Args:
        x (np.ndarray): Input features, shape (n,)
        w (float): Weight
        b (float): Bias
    
    Returns:
        np.ndarray: Probabilities, shape (n,)
    """
    # TODO: Compute probabilities using sigmoid
    return None


def predict(x, w, b, threshold=0.5):
    """
    Predict class labels (0 or 1) using threshold.
    
    If probability >= threshold: predict 1
    Otherwise: predict 0
    
    Args:
        x (np.ndarray): Input features
        w (float): Weight
        b (float): Bias
        threshold (float): Decision threshold (default 0.5)
    
    Returns:
        np.ndarray: Class predictions (0 or 1)
    """
    # TODO: Get probabilities, then apply threshold
    return None


def binary_cross_entropy_loss(y_pred_proba, y_true):
    """
    Compute binary cross-entropy loss.
    
    BCE = -mean(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
    
    Args:
        y_pred_proba (np.ndarray): Predicted probabilities, shape (n,)
        y_true (np.ndarray): True labels (0 or 1), shape (n,)
    
    Returns:
        float: Binary cross-entropy loss
    """
    # TODO: Compute BCE loss
    # Hint: Use np.log() and handle numerical stability (add small epsilon)
    # Avoid log(0) by clipping probabilities: np.clip(y_pred_proba, 1e-15, 1-1e-15)
    return None


def compute_gradients_logistic(x, y_pred_proba, y_true):
    """
    Compute gradients for logistic regression.
    
    Gradients:
    dLoss/dw = mean(x * (y_pred - y_true))
    dLoss/db = mean(y_pred - y_true)
    
    Args:
        x (np.ndarray): Input features
        y_pred_proba (np.ndarray): Predicted probabilities
        y_true (np.ndarray): True labels
    
    Returns:
        tuple: (grad_w, grad_b)
    """
    # TODO: Compute gradients
    grad_w = None
    grad_b = None
    return grad_w, grad_b


def train_logistic_regression(x, y, learning_rate=0.01, epochs=1000):
    """
    Train logistic regression model.
    
    Args:
        x (np.ndarray): Input features
        y (np.ndarray): True labels (0 or 1)
        learning_rate (float): Learning rate
        epochs (int): Number of epochs
    
    Returns:
        tuple: (w, b, loss_history)
    """
    # TODO: Initialize w and b
    w = None
    b = None
    
    loss_history = []
    
    # TODO: Training loop
    # For each epoch:
    #   1. Compute probabilities
    #   2. Compute loss, append to history
    #   3. Compute gradients
    #   4. Update w and b
    
    return w, b, loss_history




