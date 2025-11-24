# Project 29: Mini-Batch vs Stochastic Gradient Descent

## Learning Objectives

- Understand the difference between batch, mini-batch, and stochastic gradient descent
- Learn the trade-offs between different batch sizes
- Implement mini-batch processing in training loops
- Understand how batch size affects convergence and training dynamics
- Learn about momentum and advanced optimizers conceptually
- Understand why mini-batch GD is preferred in practice

## Problem Description

Gradient descent can be performed in different ways depending on how much data we use to compute each gradient update. Understanding these variants is crucial for efficient training.

**Three Variants:**
1. **Batch Gradient Descent**: Use entire dataset for each update
2. **Stochastic Gradient Descent (SGD)**: Use one example at a time
3. **Mini-Batch Gradient Descent**: Use small batches (compromise)

**Why Mini-Batch?**
- Batch GD: Stable but slow (one update per epoch)
- SGD: Fast updates but noisy (many updates per epoch)
- Mini-Batch: Balance of stability and speed

## Key Concepts

### Gradient Descent Variants

```
Batch GD:        Update after processing ALL data
                 Stable, slow, memory-intensive

SGD:             Update after EACH example
                 Fast, noisy, many updates

Mini-Batch GD:   Update after processing BATCH of examples
                 Balanced: stable enough, fast enough
```

### Batch Size Trade-offs

**Small Batch (SGD-like):**
- More updates per epoch
- Noisier gradients
- Faster per update
- Less stable convergence

**Large Batch:**
- Fewer updates per epoch
- Smoother gradients
- Slower per update
- More stable convergence

**Mini-Batch (32-256):**
- Good balance
- Standard in practice
- Works well with GPU parallelism

### Momentum

**Concept**: Accumulate velocity in direction of persistent gradients.

**Effect**: Helps overcome local minima and speeds convergence.

**Mathematical Formulation:**
```
velocity = momentum * velocity + learning_rate * gradient
weights = weights - velocity
```

## Solution Approach

### Implementing Mini-Batch Processing

1. **Shuffle data**: Randomize order
2. **Create batches**: Split into mini-batches
3. **Process batches**: Compute gradients per batch
4. **Update weights**: After each batch (not after full epoch)

### Batch Size Selection

- **Small datasets**: Use batch size = dataset size (batch GD)
- **Large datasets**: Use batch size = 32-256 (mini-batch)
- **Memory constraints**: Use smaller batches
- **GPU available**: Use larger batches (parallelism)

## How Python Uniquely Solves This

### 1. Easy Batch Creation

```python
# Python - simple batch creation
for i in range(0, len(X), batch_size):
    batch_X = X[i:i+batch_size]
    batch_y = y[i:i+batch_size]
    # Process batch
```

### 2. NumPy Slicing

```python
# Efficient batch processing with NumPy
batches = [X[i:i+batch_size] for i in range(0, len(X), batch_size)]
```

## Comparison with Other Languages

### Go
- **More verbose**: Batch creation requires more code
- **Less ML tooling**: Fewer libraries for batch processing

### TypeScript
- **Limited ML support**: Not designed for scientific computing

### Rust
- **Performance**: Fast but batch processing more complex

## Deliverables

Complete functions for:
1. Creating mini-batches from dataset
2. Training loop with mini-batch processing
3. Comparing different batch sizes
4. Implementing simple momentum update
5. Tracking updates per epoch for different batch sizes

## Testing

Run: `pytest test.py -v`

Tests verify:
- Batches are created correctly
- All data is processed
- Batch size affects number of updates
- Momentum accumulates correctly

## Next Steps

After completing this project, you'll understand:
- Why mini-batch GD is standard in practice
- How batch size affects training
- How to implement efficient batch processing

This knowledge is essential - PyTorch DataLoaders use mini-batch processing by default, and understanding why helps you choose appropriate batch sizes.
