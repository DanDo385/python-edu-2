"""
Project 26: Model Evaluation and Data Splitting

Complete the functions below according to their docstrings.
Run pytest test.py -v to test your solutions.

Note: You'll need to install numpy: pip install numpy
"""

import numpy as np


def split_data(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split dataset into training, validation, and test sets with shuffling.
    
    Args:
        X (np.ndarray): Features, shape (n_samples, n_features)
        y (np.ndarray): Labels, shape (n_samples,)
        train_ratio (float): Proportion for training set (default 0.7)
        val_ratio (float): Proportion for validation set (default 0.15)
        test_ratio (float): Proportion for test set (default 0.15)
        random_seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    
    Requirements:
        - Ratios should sum to 1.0 (or approximately 1.0)
        - Data should be shuffled before splitting
        - Sets should be disjoint (no overlap)
        - Use random_seed for reproducibility
    """
    # TODO: Implement data splitting
    # 1. Set random seed: np.random.seed(random_seed)
    # 2. Create shuffled indices: np.random.permutation(len(X))
    # 3. Calculate split points based on ratios
    # 4. Split X and y using indices
    # 5. Return all splits
    return None, None, None, None, None, None


def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.
    
    Accuracy = (Number of correct predictions) / (Total predictions)
    
    Args:
        y_true (np.ndarray): True labels, shape (n_samples,)
        y_pred (np.ndarray): Predicted labels, shape (n_samples,)
    
    Returns:
        float: Accuracy between 0.0 and 1.0
    """
    # TODO: Compute accuracy
    # Hint: Count how many predictions match true labels, divide by total
    return None


def compute_confusion_matrix(y_true, y_pred, num_classes=2):
    """
    Compute confusion matrix for classification.
    
    Confusion matrix structure (for binary classification):
        [[TN, FP],
         [FN, TP]]
    
    Args:
        y_true (np.ndarray): True labels, shape (n_samples,)
        y_pred (np.ndarray): Predicted labels, shape (n_samples,)
        num_classes (int): Number of classes (default 2 for binary)
    
    Returns:
        np.ndarray: Confusion matrix, shape (num_classes, num_classes)
    
    Note:
        - Rows represent true labels
        - Columns represent predicted labels
        - Element [i, j] = count of samples with true label i and predicted label j
    """
    # TODO: Build confusion matrix
    # Hint: Initialize matrix of zeros, then iterate and count
    # For each sample, increment matrix[y_true[i], y_pred[i]]
    return None


def compute_precision(y_true, y_pred, positive_class=1):
    """
    Compute precision for binary classification.
    
    Precision = TP / (TP + FP)
    
    Precision measures: "Of all positive predictions, how many were correct?"
    
    Args:
        y_true (np.ndarray): True labels (binary: 0 or 1)
        y_pred (np.ndarray): Predicted labels (binary: 0 or 1)
        positive_class (int): Which class is considered positive (default 1)
    
    Returns:
        float: Precision score
    """
    # TODO: Compute precision
    # Hint: Use confusion matrix or count TP and FP directly
    return None


def compute_recall(y_true, y_pred, positive_class=1):
    """
    Compute recall for binary classification.
    
    Recall = TP / (TP + FN)
    
    Recall measures: "Of all actual positives, how many did we find?"
    
    Args:
        y_true (np.ndarray): True labels (binary: 0 or 1)
        y_pred (np.ndarray): Predicted labels (binary: 0 or 1)
        positive_class (int): Which class is considered positive (default 1)
    
    Returns:
        float: Recall score
    """
    # TODO: Compute recall
    # Hint: Use confusion matrix or count TP and FN directly
    return None


def compute_mse(y_true, y_pred):
    """
    Compute Mean Squared Error for regression.
    
    MSE = mean((y_true - y_pred)^2)
    
    Args:
        y_true (np.ndarray): True target values, shape (n_samples,)
        y_pred (np.ndarray): Predicted target values, shape (n_samples,)
    
    Returns:
        float: Mean Squared Error
    """
    # TODO: Compute MSE
    # Hint: Compute squared differences, then take mean
    return None


def compute_mae(y_true, y_pred):
    """
    Compute Mean Absolute Error for regression.
    
    MAE = mean(|y_true - y_pred|)
    
    Args:
        y_true (np.ndarray): True target values, shape (n_samples,)
        y_pred (np.ndarray): Predicted target values, shape (n_samples,)
    
    Returns:
        float: Mean Absolute Error
    """
    # TODO: Compute MAE
    # Hint: Compute absolute differences, then take mean
    return None


def verify_splits_disjoint(X_train, X_val, X_test):
    """
    Verify that training, validation, and test sets are disjoint (no overlap).
    
    This is critical - overlapping sets would cause data leakage!
    
    Args:
        X_train (np.ndarray): Training features
        X_val (np.ndarray): Validation features
        X_test (np.ndarray): Test features
    
    Returns:
        bool: True if sets are disjoint, False otherwise
    
    Note:
        - Compare arrays row by row to check for duplicates
        - Can use np.array_equal for row comparison
    """
    # TODO: Check that no row appears in multiple sets
    # Hint: Iterate through rows and check if any row from one set
    #       appears in another set
    return None


def compare_train_test_performance(y_train_true, y_train_pred, y_test_true, y_test_pred):
    """
    Compare model performance on training vs test data.
    
    This function helps detect overfitting:
    - If train accuracy >> test accuracy → overfitting
    - If train accuracy ≈ test accuracy → good generalization
    
    Args:
        y_train_true (np.ndarray): Training true labels
        y_train_pred (np.ndarray): Training predictions
        y_test_true (np.ndarray): Test true labels
        y_test_pred (np.ndarray): Test predictions
    
    Returns:
        dict: Dictionary with keys 'train_accuracy', 'test_accuracy', 'generalization_gap'
    
    Note:
        generalization_gap = train_accuracy - test_accuracy
        Positive gap indicates overfitting
    """
    # TODO: Compute accuracies and gap
    # Hint: Use compute_accuracy for both sets
    return {
        'train_accuracy': None,
        'test_accuracy': None,
        'generalization_gap': None
    }
