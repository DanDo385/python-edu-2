"""
Project 26: Model Evaluation and Data Splitting - SOLUTION

Complete solution with detailed comments explaining proper evaluation practices.
"""

import numpy as np


def split_data(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split dataset into training, validation, and test sets with shuffling.
    
    This is a fundamental ML practice - we must never evaluate on training data alone!
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get number of samples
    n_samples = len(X)
    
    # Create shuffled indices
    indices = np.random.permutation(n_samples)
    
    # Calculate split points
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # Split indices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Split data using indices
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.
    
    Accuracy is the simplest metric: what fraction of predictions are correct?
    """
    # Count correct predictions (where true == predicted)
    correct = np.sum(y_true == y_pred)
    
    # Divide by total number of predictions
    accuracy = correct / len(y_true)
    
    return accuracy


def compute_confusion_matrix(y_true, y_pred, num_classes=2):
    """
    Compute confusion matrix for classification.
    
    Confusion matrix provides detailed breakdown:
    - True Negatives (TN): Correctly predicted negatives
    - False Positives (FP): Incorrectly predicted as positive
    - False Negatives (FN): Incorrectly predicted as negative
    - True Positives (TP): Correctly predicted positives
    """
    # Initialize matrix with zeros
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Count occurrences of each (true_label, pred_label) pair
    for i in range(len(y_true)):
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])
        cm[true_label, pred_label] += 1
    
    return cm


def compute_precision(y_true, y_pred, positive_class=1):
    """
    Compute precision for binary classification.
    
    Precision = TP / (TP + FP)
    
    "Of all positive predictions, how many were actually positive?"
    High precision means few false positives.
    """
    # Count TP and FP
    tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
    fp = np.sum((y_true != positive_class) & (y_pred == positive_class))
    
    # Avoid division by zero
    if tp + fp == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    return precision


def compute_recall(y_true, y_pred, positive_class=1):
    """
    Compute recall for binary classification.
    
    Recall = TP / (TP + FN)
    
    "Of all actual positives, how many did we find?"
    High recall means few false negatives.
    """
    # Count TP and FN
    tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
    fn = np.sum((y_true == positive_class) & (y_pred != positive_class))
    
    # Avoid division by zero
    if tp + fn == 0:
        return 0.0
    
    recall = tp / (tp + fn)
    return recall


def compute_mse(y_true, y_pred):
    """
    Compute Mean Squared Error for regression.
    
    MSE = mean((y_true - y_pred)^2)
    
    MSE penalizes large errors more than small errors (due to squaring).
    This makes it sensitive to outliers.
    """
    # Compute squared differences
    squared_errors = (y_true - y_pred) ** 2
    
    # Take mean
    mse = np.mean(squared_errors)
    
    return mse


def compute_mae(y_true, y_pred):
    """
    Compute Mean Absolute Error for regression.
    
    MAE = mean(|y_true - y_pred|)
    
    MAE treats all errors equally (no squaring).
    More robust to outliers than MSE.
    """
    # Compute absolute differences
    absolute_errors = np.abs(y_true - y_pred)
    
    # Take mean
    mae = np.mean(absolute_errors)
    
    return mae


def verify_splits_disjoint(X_train, X_val, X_test):
    """
    Verify that training, validation, and test sets are disjoint.
    
    This is critical - overlapping sets would cause data leakage!
    Data leakage means the model sees test data during training, leading to
    overly optimistic performance estimates.
    """
    # Check train vs val
    for row in X_train:
        # Check if this row appears in val or test
        if np.any(np.all(X_val == row, axis=1)) or np.any(np.all(X_test == row, axis=1)):
            return False
    
    # Check val vs test
    for row in X_val:
        if np.any(np.all(X_test == row, axis=1)):
            return False
    
    return True


def compare_train_test_performance(y_train_true, y_train_pred, y_test_true, y_test_pred):
    """
    Compare model performance on training vs test data.
    
    This function helps detect overfitting:
    - Large gap (train >> test) → overfitting
    - Small gap (train ≈ test) → good generalization
    
    The generalization gap reveals whether the model memorized training data
    or learned generalizable patterns.
    """
    # Compute accuracies
    train_accuracy = compute_accuracy(y_train_true, y_train_pred)
    test_accuracy = compute_accuracy(y_test_true, y_test_pred)
    
    # Compute gap
    generalization_gap = train_accuracy - test_accuracy
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'generalization_gap': generalization_gap
    }


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample dataset
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    y = np.random.randint(0, 2, 1000)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Splits are disjoint: {verify_splits_disjoint(X_train, X_val, X_test)}")
    
    # Simulate predictions (in practice, these come from a trained model)
    # For demo: perfect on train (overfitting), worse on test
    y_train_pred = y_train.copy()  # Perfect predictions
    y_test_pred = np.random.choice([0, 1], size=len(y_test))  # Random (poor)
    
    # Compute metrics
    train_acc = compute_accuracy(y_train, y_train_pred)
    test_acc = compute_accuracy(y_test, y_test_pred)
    
    print(f"\nTraining accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Generalization gap: {train_acc - test_acc:.3f}")
    
    # Confusion matrix
    cm = compute_confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion matrix:\n{cm}")
    
    # Precision and recall
    precision = compute_precision(y_test, y_test_pred)
    recall = compute_recall(y_test, y_test_pred)
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
