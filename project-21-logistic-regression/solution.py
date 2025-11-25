"""
Project 21: Logistic Regression for Classification - SOLUTION

Complete solution with detailed comments explaining classification,
sigmoid function, and cross-entropy loss.
"""

import numpy as np


def sigmoid(x):
    """
    Compute sigmoid (logistic) function.
    
    The sigmoid function maps any real number to the range (0, 1).
    This makes it perfect for converting linear model outputs to probabilities.
    
    Formula: sigmoid(x) = 1 / (1 + exp(-x))
    
    Properties:
    - Output range: (0, 1) - perfect for probabilities
    - S-shaped curve (sigmoid = "S-shaped")
    - Smooth and differentiable everywhere
    - Symmetric around (0, 0.5)
    
    Why sigmoid for classification?
    - Linear regression outputs unbounded values (-∞ to +∞)
    - Probabilities must be in [0, 1]
    - Sigmoid provides smooth, differentiable mapping
    
    Args:
        x (np.ndarray or float): Input value(s)
    
    Returns:
        np.ndarray or float: Sigmoid output(s), same shape as input
    
    Example:
        sigmoid(0)    # Returns 0.5
        sigmoid(5)    # Returns ~0.993 (high probability)
        sigmoid(-5)   # Returns ~0.007 (low probability)
    """
    # Compute sigmoid using vectorized operations
    # np.exp() works element-wise on arrays
    # For large negative x, exp(-x) is large, so 1/(1+large) ≈ 0
    # For large positive x, exp(-x) ≈ 0, so 1/(1+0) = 1
    return 1 / (1 + np.exp(-x))
    
    # Note: For numerical stability with very negative x, we could use:
    # return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))
    # But the simple version works fine for most cases


def predict_proba(x, w, b):
    """
    Predict probability of class 1 using logistic regression.
    
    This is the forward pass for logistic regression:
    1. Compute linear combination: z = w*x + b
    2. Apply sigmoid: p = sigmoid(z)
    
    The output is a probability: p ∈ (0, 1)
    - p close to 1: high confidence in class 1
    - p close to 0: high confidence in class 0
    - p = 0.5: uncertain (decision boundary)
    
    Args:
        x (np.ndarray): Input features, shape (n,)
        w (float): Weight
        b (float): Bias
    
    Returns:
        np.ndarray: Probabilities of class 1, shape (n,)
    
    Example:
        x = [0, 1, 2]
        w = 1.0, b = 0.0
        predict_proba(x, w, b)  # Returns probabilities
    """
    # Step 1: Compute linear combination (same as linear regression)
    z = w * x + b
    
    # Step 2: Apply sigmoid to get probabilities
    # This converts unbounded z values to probabilities in (0, 1)
    probabilities = sigmoid(z)
    
    return probabilities


def predict(x, w, b, threshold=0.5):
    """
    Predict class labels (0 or 1) using threshold.
    
    Converts probabilities to hard class predictions.
    This is the final step: probability → class label.
    
    Decision rule:
    - If probability >= threshold: predict class 1
    - Otherwise: predict class 0
    
    Default threshold is 0.5 (equal probability), but can be adjusted
    based on precision/recall trade-offs.
    
    Args:
        x (np.ndarray): Input features
        w (float): Weight
        b (float): Bias
        threshold (float): Decision threshold (default 0.5)
    
    Returns:
        np.ndarray: Class predictions (0 or 1)
    
    Example:
        x = [-2, 0, 2]
        w = 1.0, b = 0.0
        predict(x, w, b)  # Returns [0, 0 or 1, 1] depending on probabilities
    """
    # Get probabilities first
    probabilities = predict_proba(x, w, b)
    
    # Convert probabilities to class labels using threshold
    # probabilities >= threshold returns boolean array
    # Converting to int gives 0 or 1
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions


def binary_cross_entropy_loss(y_pred_proba, y_true):
    """
    Compute binary cross-entropy loss.
    
    Cross-entropy is the standard loss function for classification.
    It measures the difference between predicted probabilities and true labels.
    
    Formula:
    BCE = -mean(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
    
    Why cross-entropy?
    - Appropriate for probability predictions
    - Heavily penalizes confident wrong predictions
    - Well-behaved gradient (good for optimization)
    
    Interpretation:
    - If y_true=1 and y_pred=0.9: small loss (good prediction)
    - If y_true=1 and y_pred=0.1: large loss (confidently wrong!)
    - If y_true=1 and y_pred=0.5: moderate loss (uncertain)
    
    Args:
        y_pred_proba (np.ndarray): Predicted probabilities, shape (n,)
        y_true (np.ndarray): True labels (0 or 1), shape (n,)
    
    Returns:
        float: Binary cross-entropy loss
    
    Example:
        y_pred = [0.9, 0.1, 0.8]
        y_true = [1, 0, 1]
        binary_cross_entropy_loss(y_pred, y_true)  # Computes BCE
    """
    # Clip probabilities to avoid numerical issues
    # log(0) is undefined, so we avoid probabilities exactly 0 or 1
    # Small epsilon (1e-15) prevents log(0) while maintaining accuracy
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    
    # Compute cross-entropy loss
    # For each sample:
    # - If y_true=1: loss = -log(y_pred)
    # - If y_true=0: loss = -log(1-y_pred)
    # This is computed element-wise, then averaged
    loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    
    return loss


def compute_gradients_logistic(x, y_pred_proba, y_true):
    """
    Compute gradients for logistic regression.
    
    These gradients tell us how to adjust w and b to reduce loss.
    
    Mathematical derivation (simplified):
    Loss = BCE = -mean(y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z)))
    where z = w*x + b
    
    After simplification (sigmoid derivative cancels nicely):
    dLoss/dw = mean(x * (y_pred - y_true))
    dLoss/db = mean(y_pred - y_true)
    
    Notice: These are similar to linear regression gradients!
    The difference is y_pred is a probability (sigmoid output), not raw value.
    
    Args:
        x (np.ndarray): Input features
        y_pred_proba (np.ndarray): Predicted probabilities
        y_true (np.ndarray): True labels
    
    Returns:
        tuple: (grad_w, grad_b) gradients
    """
    # Compute errors: difference between predictions and true labels
    errors = y_pred_proba - y_true
    
    # Gradient w.r.t. w: mean(x * errors)
    # Similar to linear regression, but errors are probability differences
    grad_w = np.mean(x * errors)
    
    # Gradient w.r.t. b: mean(errors)
    # Average error across all samples
    grad_b = np.mean(errors)
    
    return grad_w, grad_b


def train_logistic_regression(x, y, learning_rate=0.01, epochs=1000):
    """
    Train logistic regression model using gradient descent.
    
    This is the complete training loop for classification.
    Same structure as linear regression, but with:
    - Sigmoid activation (probabilities)
    - Cross-entropy loss (instead of MSE)
    - Different gradient formulas
    
    Args:
        x (np.ndarray): Input features
        y (np.ndarray): True labels (0 or 1)
        learning_rate (float): Learning rate
        epochs (int): Number of epochs
    
    Returns:
        tuple: (w, b, loss_history)
    """
    # Initialize parameters
    np.random.seed(42)
    w = np.random.randn() * 0.01
    b = np.random.randn() * 0.01
    
    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass: compute probabilities
        y_pred_proba = predict_proba(x, w, b)
        
        # Compute loss
        loss = binary_cross_entropy_loss(y_pred_proba, y)
        loss_history.append(loss)
        
        # Compute gradients
        grad_w, grad_b = compute_gradients_logistic(x, y_pred_proba, y)
        
        # Update parameters
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
    
    return w, b, loss_history




