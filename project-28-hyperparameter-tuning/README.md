# Project 28: Hyperparameter Tuning and Experimentation

## Learning Objectives

- Understand what hyperparameters are and why they matter
- Learn systematic approaches to hyperparameter tuning (grid search, random search)
- Understand the role of validation set in hyperparameter selection
- Implement learning rate scheduling
- Track and compare multiple training experiments
- Understand the importance of reproducibility in ML experiments

## Problem Description

Hyperparameters are settings that control the learning process itself - they're not learned from data but must be chosen by the practitioner. Choosing good hyperparameters is crucial for model performance, but it requires systematic experimentation.

**What are Hyperparameters?**
- Learning rate: How big steps to take during optimization
- Number of epochs: How long to train
- Batch size: How many examples per update
- Network architecture: Number of layers, neurons per layer
- Regularization strength: Lambda for L2, dropout rate
- Optimizer choice: SGD, Adam, etc.

**Why Hyperparameter Tuning Matters:**
- Same model architecture, different hyperparameters → vastly different performance
- Poor hyperparameters → slow convergence or no convergence
- Good hyperparameters → fast convergence and good generalization

## Key Concepts

### Hyperparameter Types

```
┌─────────────────────────────────────────┐
│         Hyperparameter Categories        │
├─────────────────────────────────────────┤
│  Learning Process:                      │
│    - Learning rate                      │
│    - Batch size                         │
│    - Number of epochs                   │
│                                         │
│  Architecture:                          │
│    - Hidden layer sizes                 │
│    - Number of layers                   │
│    - Activation functions                │
│                                         │
│  Regularization:                        │
│    - L2 lambda                         │
│    - Dropout rate                       │
│                                         │
│  Optimization:                          │
│    - Optimizer type                     │
│    - Momentum                           │
└─────────────────────────────────────────┘
```

### Grid Search vs Random Search

**Grid Search:**
- Try all combinations of specified values
- Systematic but computationally expensive
- Good when you have few hyperparameters

**Random Search:**
- Sample random combinations
- More efficient, often finds better values
- Better when you have many hyperparameters

### Learning Rate Scheduling

**The Problem:**
- Fixed learning rate: May be too large (overshoot) or too small (slow)

**The Solution:**
- Start with larger learning rate
- Gradually decrease over time
- Common schedules: Step decay, exponential decay, cosine annealing

### Validation Set for Hyperparameter Selection

```
Training Set → Train model
Validation Set → Evaluate hyperparameters (choose best)
Test Set → Final evaluation (only once!)
```

**Critical Rule:** Never use test set for hyperparameter tuning!

## Solution Approach

### Experimentation Workflow

1. **Define hyperparameter space**: What values to try?
2. **Set up experiments**: Create combinations to test
3. **Train and evaluate**: Run each experiment, track results
4. **Compare results**: Which hyperparameters performed best?
5. **Select best**: Choose hyperparameters with best validation performance

### Learning Rate Scheduling

**Step Decay:**
```python
if epoch % decay_interval == 0:
    learning_rate *= decay_factor
```

**Exponential Decay:**
```python
learning_rate = initial_lr * (decay_rate ** epoch)
```

## How Python Uniquely Solves This

### 1. Easy Experimentation

```python
# Iterate over hyperparameter combinations
for lr in [0.01, 0.001, 0.0001]:
    for batch_size in [32, 64, 128]:
        train_model(lr, batch_size)
```

### 2. Rich Libraries

```python
# Tools like itertools for combinations
from itertools import product
for lr, bs in product(learning_rates, batch_sizes):
    train_model(lr, bs)
```

### 3. Easy Tracking

```python
# Dictionaries for tracking results
results = {
    'learning_rate': [],
    'batch_size': [],
    'validation_loss': []
}
```

## Deliverables

Complete functions for:
1. Grid search over hyperparameter combinations
2. Learning rate scheduling (step decay, exponential decay)
3. Experiment tracking and comparison
4. Finding best hyperparameters based on validation performance
5. Reproducible experiments with random seeds

## Testing

Run: `pytest test.py -v`

Tests verify:
- Grid search generates correct combinations
- Learning rate schedules update correctly
- Best hyperparameters are selected correctly
- Experiments are reproducible with fixed seeds

## Next Steps

After completing this project, you'll understand:
- How to systematically tune hyperparameters
- Why validation set is crucial for hyperparameter selection
- How learning rate scheduling improves training
- The importance of tracking experiments

This knowledge is essential for training any ML model - you'll always need to tune hyperparameters, whether training simple models or complex deep networks.
