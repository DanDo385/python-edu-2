"""
Test suite for Project 28: Hyperparameter Tuning

Run with: pytest test.py -v
"""

import pytest
import numpy as np
from exercise import (
    grid_search_hyperparameters,
    step_decay_schedule,
    exponential_decay_schedule,
    run_experiment,
    find_best_hyperparameters,
    compare_experiments,
    track_learning_rate_history,
    verify_reproducibility,
)


class TestGridSearch:
    """Test grid search functionality."""
    
    def test_grid_search_basic(self):
        """Test basic grid search."""
        grid = {
            'learning_rate': [0.01, 0.001],
            'batch_size': [32, 64]
        }
        
        combinations = grid_search_hyperparameters(grid)
        
        assert len(combinations) == 4  # 2 * 2 = 4 combinations
        
        # Check all combinations are present
        expected = [
            {'learning_rate': 0.01, 'batch_size': 32},
            {'learning_rate': 0.01, 'batch_size': 64},
            {'learning_rate': 0.001, 'batch_size': 32},
            {'learning_rate': 0.001, 'batch_size': 64},
        ]
        
        for exp in expected:
            assert exp in combinations
    
    def test_grid_search_single_param(self):
        """Test grid search with single parameter."""
        grid = {'learning_rate': [0.01, 0.001, 0.0001]}
        combinations = grid_search_hyperparameters(grid)
        
        assert len(combinations) == 3
        assert {'learning_rate': 0.01} in combinations
        assert {'learning_rate': 0.001} in combinations
        assert {'learning_rate': 0.0001} in combinations


class TestLearningRateSchedules:
    """Test learning rate scheduling."""
    
    def test_step_decay_basic(self):
        """Test step decay schedule."""
        initial_lr = 0.01
        decay_interval = 5
        decay_factor = 0.5
        
        # Epochs 0-4: 0.01
        assert abs(step_decay_schedule(initial_lr, 0, decay_interval, decay_factor) - 0.01) < 0.001
        assert abs(step_decay_schedule(initial_lr, 4, decay_interval, decay_factor) - 0.01) < 0.001
        
        # Epochs 5-9: 0.005
        assert abs(step_decay_schedule(initial_lr, 5, decay_interval, decay_factor) - 0.005) < 0.001
        assert abs(step_decay_schedule(initial_lr, 9, decay_interval, decay_factor) - 0.005) < 0.001
        
        # Epochs 10-14: 0.0025
        assert abs(step_decay_schedule(initial_lr, 10, decay_interval, decay_factor) - 0.0025) < 0.001
    
    def test_exponential_decay(self):
        """Test exponential decay schedule."""
        initial_lr = 0.01
        decay_rate = 0.95
        
        # Epoch 0: 0.01
        assert abs(exponential_decay_schedule(initial_lr, 0, decay_rate) - 0.01) < 0.001
        
        # Epoch 1: 0.01 * 0.95 = 0.0095
        assert abs(exponential_decay_schedule(initial_lr, 1, decay_rate) - 0.0095) < 0.001
        
        # Epoch 2: 0.01 * 0.95^2 ≈ 0.009025
        expected = 0.01 * (0.95 ** 2)
        assert abs(exponential_decay_schedule(initial_lr, 2, decay_rate) - expected) < 0.001


class TestExperiments:
    """Test experiment running and comparison."""
    
    def test_run_experiment(self):
        """Test running a single experiment."""
        # Create dummy data
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randint(0, 2, 20)
        
        hyperparams = {'learning_rate': 0.001, 'batch_size': 32}
        
        result = run_experiment(None, X_train, y_train, X_val, y_val, hyperparams, random_seed=42)
        
        assert 'hyperparams' in result
        assert 'val_loss' in result
        assert 'val_accuracy' in result
        assert result['hyperparams'] == hyperparams
        assert result['val_loss'] > 0
    
    def test_find_best_hyperparameters(self):
        """Test finding best hyperparameters."""
        results = [
            {'hyperparams': {'lr': 0.01}, 'val_loss': 0.5, 'val_accuracy': 0.5},
            {'hyperparams': {'lr': 0.001}, 'val_loss': 0.3, 'val_accuracy': 0.7},
            {'hyperparams': {'lr': 0.0001}, 'val_loss': 0.4, 'val_accuracy': 0.6},
        ]
        
        # Find best by val_loss (minimize)
        best = find_best_hyperparameters(results, metric='val_loss', minimize=True)
        assert best['lr'] == 0.001  # Lowest loss
        
        # Find best by val_accuracy (maximize)
        best_acc = find_best_hyperparameters(results, metric='val_accuracy', minimize=False)
        assert best_acc['lr'] == 0.001  # Highest accuracy
    
    def test_compare_experiments(self):
        """Test comparing multiple experiments."""
        results = [
            {'hyperparams': {'lr': 0.01}, 'val_loss': 0.5},
            {'hyperparams': {'lr': 0.001}, 'val_loss': 0.3},
            {'hyperparams': {'lr': 0.0001}, 'val_loss': 0.4},
        ]
        
        summary = compare_experiments(results)
        
        assert summary['best_val_loss'] == 0.3
        assert summary['worst_val_loss'] == 0.5
        assert abs(summary['mean_val_loss'] - 0.4) < 0.01
        assert summary['best_hyperparams']['lr'] == 0.001


class TestLearningRateTracking:
    """Test learning rate history tracking."""
    
    def test_track_step_decay(self):
        """Test tracking step decay schedule."""
        lr_history = track_learning_rate_history(
            initial_lr=0.01,
            num_epochs=15,
            schedule_type='step',
            decay_interval=5,
            decay_factor=0.5
        )
        
        assert len(lr_history) == 15
        assert lr_history[0] == 0.01
        assert lr_history[4] == 0.01  # Still 0.01 at epoch 4
        assert lr_history[5] == 0.005  # Decayed at epoch 5
        assert lr_history[10] == 0.0025  # Decayed again at epoch 10
    
    def test_track_exponential_decay(self):
        """Test tracking exponential decay schedule."""
        lr_history = track_learning_rate_history(
            initial_lr=0.01,
            num_epochs=5,
            schedule_type='exponential',
            decay_rate=0.95
        )
        
        assert len(lr_history) == 5
        assert abs(lr_history[0] - 0.01) < 0.001
        assert abs(lr_history[1] - 0.0095) < 0.001
        assert abs(lr_history[2] - 0.01 * (0.95**2)) < 0.001


class TestReproducibility:
    """Test experiment reproducibility."""
    
    def test_reproducibility_same_seed(self):
        """Test that same seed produces same results."""
        def dummy_experiment(hyperparams, seed):
            np.random.seed(seed)
            return {'result': np.random.randn()}
        
        hyperparams = {'lr': 0.01}
        is_reproducible = verify_reproducibility(
            dummy_experiment, hyperparams, seed1=42, seed2=42
        )
        
        assert is_reproducible == True
    
    def test_reproducibility_different_seed(self):
        """Test that different seeds produce different results."""
        def dummy_experiment(hyperparams, seed):
            np.random.seed(seed)
            return {'result': np.random.randn()}
        
        hyperparams = {'lr': 0.01}
        is_reproducible = verify_reproducibility(
            dummy_experiment, hyperparams, seed1=42, seed2=43
        )
        
        assert is_reproducible == False
