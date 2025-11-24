# Solution Explanation: Hyperparameter Tuning

## Overview

Hyperparameter tuning is the process of finding the best settings for your model. Unlike model parameters (weights) which are learned from data, hyperparameters must be chosen by the practitioner. This project teaches systematic approaches to hyperparameter tuning.

## Key Concepts Explained

### What are Hyperparameters?

**Hyperparameters control the learning process:**
- Learning rate: How big steps to take during optimization
- Batch size: How many examples per update
- Number of epochs: How long to train
- Network architecture: Layers, neurons per layer
- Regularization strength: Lambda, dropout rate

**Why They Matter:**
- Same architecture, different hyperparameters → vastly different performance
- Poor hyperparameters → slow convergence or no convergence
- Good hyperparameters → fast convergence and good generalization

### Grid Search

**The Approach:**
- Try all combinations of specified values
- Systematic but computationally expensive
- Good when you have few hyperparameters

**Example:**
```python
grid = {
    'learning_rate': [0.01, 0.001],
    'batch_size': [32, 64]
}
# Generates 4 combinations: (0.01, 32), (0.01, 64), (0.001, 32), (0.001, 64)
```

**When to Use:**
- Small number of hyperparameters (2-3)
- Small number of values per hyperparameter
- When you want exhaustive search

### Random Search

**The Approach:**
- Sample random combinations from hyperparameter space
- More efficient than grid search
- Often finds better values

**When to Use:**
- Many hyperparameters
- Large search space
- Limited computational budget

### Learning Rate Scheduling

**The Problem:**
- Fixed learning rate may be too large (overshoot) or too small (slow convergence)

**The Solution:**
- Start with larger learning rate
- Gradually decrease over time
- Common schedules: Step decay, exponential decay

**Step Decay:**
- Decrease LR by factor every N epochs
- Example: Every 10 epochs, multiply by 0.5
- Simple and effective

**Exponential Decay:**
- Decrease LR exponentially: `lr = initial_lr * (decay_rate ^ epoch)`
- Smooth decrease
- Good for long training

### Validation Set for Hyperparameter Selection

**Critical Rule:** Never use test set for hyperparameter tuning!

**Workflow:**
1. Train model on training set
2. Evaluate on validation set
3. Use validation performance to choose hyperparameters
4. Final evaluation on test set (only once!)

**Why:**
- Test set should remain unseen until final evaluation
- Using test set for tuning → data leakage → overly optimistic results

## Implementation Details

### Grid Search Implementation

```python
from itertools import product

def grid_search_hyperparameters(hyperparameter_grid):
    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())
    
    combinations = []
    for combo in product(*param_values):
        combo_dict = dict(zip(param_names, combo))
        combinations.append(combo_dict)
    
    return combinations
```

**Key Points:**
- Use `itertools.product` to generate all combinations
- Create dictionaries mapping names to values
- Return list of all combinations

### Learning Rate Scheduling

**Step Decay:**
```python
num_steps = epoch // decay_interval
current_lr = initial_lr * (decay_factor ** num_steps)
```

**Exponential Decay:**
```python
current_lr = initial_lr * (decay_rate ** epoch)
```

### Experiment Tracking

**Track Results:**
- Store hyperparameters and performance metrics
- Compare experiments to find best
- Use validation performance (not training!)

**Finding Best:**
```python
best = min(experiment_results, key=lambda x: x['val_loss'])
best_hyperparams = best['hyperparams']
```

## Common Pitfalls

1. **Using test set for tuning**: Data leakage!
2. **Too many hyperparameters**: Grid search becomes intractable
3. **Not tracking experiments**: Can't compare or reproduce
4. **Ignoring validation performance**: Training performance doesn't indicate generalization
5. **Not using random seeds**: Experiments not reproducible

## Real-World Application

**In Practice:**
- Start with reasonable defaults
- Use validation set to tune hyperparameters
- Try grid search for 2-3 hyperparameters
- Use random search for more hyperparameters
- Use learning rate scheduling for long training
- Track all experiments (use tools like MLflow, Weights & Biases)
- Always use random seeds for reproducibility

**In PyTorch (upcoming projects):**
- Learning rate schedulers: `torch.optim.lr_scheduler.StepLR`
- Hyperparameter tuning libraries: Optuna, Ray Tune
- Experiment tracking: TensorBoard, MLflow

## Why This Matters

Hyperparameter tuning is **essential** for getting good model performance:
- Same model, different hyperparameters → 10-50% performance difference
- Poor hyperparameters → wasted training time
- Good hyperparameters → production-ready models

This project establishes hyperparameter tuning practices that will be used in every ML project going forward.
