"""
Test suite for Project 26: Model Evaluation and Data Splitting

Run with: pytest test.py -v
"""

import pytest
import numpy as np
from exercise import (
    split_data,
    compute_accuracy,
    compute_confusion_matrix,
    compute_precision,
    compute_recall,
    compute_mse,
    compute_mae,
    verify_splits_disjoint,
    compare_train_test_performance,
)


class TestDataSplitting:
    """Test data splitting functions."""
    
    def test_split_data_ratios(self):
        """Test that split produces correct ratios."""
        X = np.random.randn(1000, 5)
        y = np.random.randint(0, 2, 1000)
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        # Check ratios are approximately correct
        assert len(X_train) == 700
        assert len(X_val) == 150
        assert len(X_test) == 150
        
        # Check that all data is accounted for
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
    
    def test_split_data_disjoint(self):
        """Test that splits are disjoint (no overlap)."""
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Verify no overlap
        assert verify_splits_disjoint(X_train, X_val, X_test)
    
    def test_split_data_reproducible(self):
        """Test that same seed produces same split."""
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)
        
        # First split
        X_train1, _, _, _, _, _ = split_data(X, y, random_seed=42)
        
        # Second split with same seed
        X_train2, _, _, _, _, _ = split_data(X, y, random_seed=42)
        
        # Should be identical
        assert np.array_equal(X_train1, X_train2)
    
    def test_split_data_shuffled(self):
        """Test that data is shuffled (not in original order)."""
        # Create ordered data
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        
        X_train, _, _, _, _, _ = split_data(X, y, random_seed=42)
        
        # First few training samples should not be [0, 1, 2, ...]
        # (unless by chance, but very unlikely)
        first_few = X_train[:5].flatten()
        assert not np.array_equal(first_few, [0, 1, 2, 3, 4])


class TestClassificationMetrics:
    """Test classification metric functions."""
    
    def test_compute_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 1.0
    
    def test_compute_accuracy_partial(self):
        """Test accuracy with partial correctness."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1])
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 0.5  # 2 out of 4 correct
    
    def test_compute_confusion_matrix_binary(self):
        """Test confusion matrix for binary classification."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=2)
        
        # Expected:
        # True=0, Pred=0: 1 (first sample)
        # True=0, Pred=1: 1 (last sample)
        # True=1, Pred=0: 1 (third sample)
        # True=1, Pred=1: 2 (second and fifth samples)
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 2  # TP
    
    def test_compute_precision(self):
        """Test precision computation."""
        y_true = np.array([1, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 0, 0])
        
        # TP=2, FP=1, so precision = 2/(2+1) = 2/3
        precision = compute_precision(y_true, y_pred, positive_class=1)
        assert abs(precision - 2/3) < 0.001
    
    def test_compute_recall(self):
        """Test recall computation."""
        y_true = np.array([1, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 0, 0])
        
        # TP=2, FN=1, so recall = 2/(2+1) = 2/3
        recall = compute_recall(y_true, y_pred, positive_class=1)
        assert abs(recall - 2/3) < 0.001


class TestRegressionMetrics:
    """Test regression metric functions."""
    
    def test_compute_mse_perfect(self):
        """Test MSE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        
        mse = compute_mse(y_true, y_pred)
        assert mse == 0.0
    
    def test_compute_mse_errors(self):
        """Test MSE with prediction errors."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 2.5])
        
        # Errors: [0.5, 0.0, 0.5]
        # Squared: [0.25, 0.0, 0.25]
        # Mean: 0.5/3 ≈ 0.167
        mse = compute_mse(y_true, y_pred)
        expected = np.mean((y_true - y_pred) ** 2)
        assert abs(mse - expected) < 0.001
    
    def test_compute_mae_perfect(self):
        """Test MAE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        mae = compute_mae(y_true, y_pred)
        assert mae == 0.0
    
    def test_compute_mae_errors(self):
        """Test MAE with prediction errors."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 2.5])
        
        # Errors: [0.5, 0.0, 0.5]
        # Mean: 1.0/3 ≈ 0.333
        mae = compute_mae(y_true, y_pred)
        expected = np.mean(np.abs(y_true - y_pred))
        assert abs(mae - expected) < 0.001


class TestGeneralization:
    """Test functions that detect overfitting."""
    
    def test_compare_train_test_performance(self):
        """Test comparison of train vs test performance."""
        # Simulate overfitting: perfect on train, worse on test
        y_train_true = np.array([0, 1, 1, 0, 1])
        y_train_pred = np.array([0, 1, 1, 0, 1])  # Perfect
        
        y_test_true = np.array([0, 1, 0, 1])
        y_test_pred = np.array([0, 0, 0, 1])  # 75% accuracy
        
        result = compare_train_test_performance(
            y_train_true, y_train_pred,
            y_test_true, y_test_pred
        )
        
        assert result['train_accuracy'] == 1.0
        assert result['test_accuracy'] == 0.75
        assert result['generalization_gap'] == 0.25
    
    def test_verify_splits_disjoint(self):
        """Test that verify_splits_disjoint works correctly."""
        # Create disjoint sets
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_val = np.array([[7, 8], [9, 10]])
        X_test = np.array([[11, 12], [13, 14]])
        
        assert verify_splits_disjoint(X_train, X_val, X_test) == True
        
        # Create overlapping sets (should fail)
        X_train_overlap = np.array([[1, 2], [3, 4]])
        X_val_overlap = np.array([[3, 4], [5, 6]])  # Overlaps with train
        X_test_overlap = np.array([[7, 8]])
        
        assert verify_splits_disjoint(X_train_overlap, X_val_overlap, X_test_overlap) == False


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_compute_accuracy_empty(self):
        """Test accuracy with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        
        # Should handle gracefully (might return 0 or raise error)
        # For this test, we'll check it doesn't crash
        try:
            accuracy = compute_accuracy(y_true, y_pred)
            # If it returns, that's fine (implementation choice)
        except (ValueError, ZeroDivisionError):
            # Also acceptable - empty input is edge case
            pass
    
    def test_split_data_small_dataset(self):
        """Test splitting with very small dataset."""
        X = np.random.randn(10, 2)
        y = np.random.randint(0, 2, 10)
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Should still produce valid splits
        assert len(X_train) > 0
        assert len(X_val) >= 0  # Might be 0 for very small datasets
        assert len(X_test) >= 0
