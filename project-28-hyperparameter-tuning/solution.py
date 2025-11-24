"""
Project 28: Hyperparameter Tuning - SOLUTION

Complete solution demonstrating systematic hyperparameter tuning.
"""

import numpy as np
from itertools import product


def grid_search_hyperparameters(hyperparameter_grid):
    """
    Generate all combinations of hyperparameters for grid search.
    """
    # Get parameter names and value lists
    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())
    
    # Generate all combinations
    combinations = []
    for combo in product(*param_values):
        # Create dictionary mapping names to values
        combo_dict = dict(zip(param_names, combo))
        combinations.append(combo_dict)
    
    return combinations


def step_decay_schedule(initial_lr, epoch, decay_interval=10, decay_factor=0.5):
    """
    Implement step decay learning rate schedule.
    """
    # Calculate number of decay steps that have occurred
    num_steps = epoch // decay_interval
    
    # Apply decay: multiply by decay_factor for each step
    current_lr = initial_lr * (decay_factor ** num_steps)
    
    return current_lr


def exponential_decay_schedule(initial_lr, epoch, decay_rate=0.95):
    """
    Implement exponential decay learning rate schedule.
    """
    # Exponential decay: initial_lr * (decay_rate ^ epoch)
    current_lr = initial_lr * (decay_rate ** epoch)
    
    return current_lr


def run_experiment(model_class, X_train, y_train, X_val, y_val, hyperparams, random_seed=42):
    """
    Run a single training experiment with given hyperparameters.
    
    For this exercise, we simulate results. In real ML, this would train the model.
    """
    np.random.seed(random_seed)
    
    # Extract hyperparameters
    lr = hyperparams.get('learning_rate', 0.01)
    batch_size = hyperparams.get('batch_size', 32)
    
    # Simulate validation loss based on hyperparameters
    # Better hyperparameters → lower loss
    base_loss = 0.5
    lr_factor = abs(np.log10(lr + 1e-6))  # Log scale
    bs_factor = 1.0 / (batch_size / 32.0)  # Larger batch → lower loss
    
    val_loss = base_loss * (1.0 / (lr_factor + 1)) * bs_factor + np.random.normal(0, 0.05)
    val_loss = max(0.1, val_loss)  # Clamp minimum
    
    val_accuracy = 1.0 - val_loss  # Inverse relationship
    val_accuracy = np.clip(val_accuracy, 0.5, 0.99)
    
    return {
        'hyperparams': hyperparams,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'final_train_loss': val_loss * 0.8  # Train loss usually lower
    }


def find_best_hyperparameters(experiment_results, metric='val_loss', minimize=True):
    """
    Find the best hyperparameters from experiment results.
    """
    if minimize:
        # Find minimum value
        best_result = min(experiment_results, key=lambda x: x[metric])
    else:
        # Find maximum value
        best_result = max(experiment_results, key=lambda x: x[metric])
    
    return best_result['hyperparams']


def compare_experiments(experiment_results):
    """
    Compare multiple experiments and return summary statistics.
    """
    # Extract validation losses
    val_losses = [r['val_loss'] for r in experiment_results]
    
    # Find best and worst
    best_val_loss = min(val_losses)
    worst_val_loss = max(val_losses)
    mean_val_loss = np.mean(val_losses)
    
    # Find best hyperparameters
    best_result = min(experiment_results, key=lambda x: x['val_loss'])
    best_hyperparams = best_result['hyperparams']
    
    return {
        'best_val_loss': best_val_loss,
        'worst_val_loss': worst_val_loss,
        'mean_val_loss': mean_val_loss,
        'best_hyperparams': best_hyperparams
    }


def track_learning_rate_history(initial_lr, num_epochs, schedule_type='step', **schedule_kwargs):
    """
    Track learning rate over training epochs using a schedule.
    """
    lr_history = []
    
    for epoch in range(num_epochs):
        if schedule_type == 'step':
            lr = step_decay_schedule(
                initial_lr,
                epoch,
                schedule_kwargs.get('decay_interval', 10),
                schedule_kwargs.get('decay_factor', 0.5)
            )
        elif schedule_type == 'exponential':
            lr = exponential_decay_schedule(
                initial_lr,
                epoch,
                schedule_kwargs.get('decay_rate', 0.95)
            )
        else:
            lr = initial_lr  # Constant learning rate
        
        lr_history.append(lr)
    
    return lr_history


def verify_reproducibility(experiment_func, hyperparams, seed1, seed2):
    """
    Verify that experiments are reproducible with same random seed.
    """
    # Run experiment with first seed
    result1 = experiment_func(hyperparams, seed1)
    
    # Run experiment with second seed
    result2 = experiment_func(hyperparams, seed2)
    
    # Check if results are identical
    if seed1 == seed2:
        # Should be identical
        return result1 == result2
    else:
        # Should be different
        return result1 != result2


# Example usage
if __name__ == "__main__":
    # Define hyperparameter grid
    grid = {
        'learning_rate': [0.01, 0.001, 0.0001],
        'batch_size': [32, 64, 128]
    }
    
    # Generate all combinations
    combinations = grid_search_hyperparameters(grid)
    print(f"Generated {len(combinations)} hyperparameter combinations")
    
    # Track learning rate schedule
    lr_history = track_learning_rate_history(
        initial_lr=0.01,
        num_epochs=20,
        schedule_type='step',
        decay_interval=5,
        decay_factor=0.5
    )
    
    print(f"\nLearning rate schedule (first 10 epochs):")
    for i, lr in enumerate(lr_history[:10]):
        print(f"  Epoch {i}: {lr:.6f}")
