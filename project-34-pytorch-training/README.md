# Project 34: Training a PyTorch Model (MLP on MNIST)

## Learning Objectives

- Learn the complete PyTorch training loop
- Use DataLoader for batch processing
- Use optimizers (SGD, Adam) for weight updates
- Define and use loss functions
- Train a model end-to-end on MNIST
- Evaluate model performance
- Understand the training loop pattern used in all PyTorch projects

## Problem Description

This project brings together everything: tensors, autograd, modules, and now training! You'll train a neural network on MNIST using PyTorch's training utilities.

**The Training Loop:**
```
1. Load data (DataLoader)
2. Define model, loss, optimizer
3. For each epoch:
   a. For each batch:
      - Forward pass
      - Compute loss
      - Backward pass (autograd!)
      - Update weights (optimizer)
   b. Evaluate on validation set
4. Test final model
```

**Why This Matters:**
- This is the standard PyTorch training pattern
- Used in all deep learning projects
- Much simpler than pure NumPy (Project 30)

## Key Concepts

### DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in dataloader:
    # Process batch
```

**Benefits:**
- Automatic batching
- Shuffling
- Parallel data loading

### Optimizer

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# or
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Usage:**
```python
optimizer.zero_grad()  # Zero gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update weights
```

### Loss Functions

```python
criterion = nn.CrossEntropyLoss()  # For classification

loss = criterion(predictions, targets)
```

### Training Loop Pattern

```python
for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        # Forward
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Solution Approach

### Complete Training Pipeline

1. **Load data**: Use DataLoader
2. **Define model**: Use nn.Module
3. **Define loss**: Use nn.CrossEntropyLoss
4. **Define optimizer**: Use torch.optim
5. **Train**: Standard loop
6. **Evaluate**: Compute accuracy

## How Python Uniquely Solves This

### 1. Clean Training Loop

```python
# Python - clean and readable
for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
```

### 2. Built-in Utilities

- DataLoader: Automatic batching
- Optimizers: Pre-implemented algorithms
- Loss functions: Ready to use

## Comparison with Other Languages

### Go
- **No PyTorch equivalent**: Would need to build everything
- **Much more complex**: Manual implementation required

### TypeScript
- **Limited ML support**: No mature training frameworks

### Rust
- **Less mature**: Fewer ML training utilities

## Deliverables

Complete implementation:
1. Data loading with DataLoader
2. Model definition
3. Loss function
4. Optimizer setup
5. Training loop
6. Evaluation function
7. Accuracy computation

## Testing

Run: `pytest test.py -v`

Tests verify:
- Training loop runs without errors
- Loss decreases over epochs
- Model achieves reasonable accuracy (≥90% on MNIST)
- Evaluation works correctly

## Next Steps

After completing this project, you'll understand:
- The complete PyTorch training pipeline
- How to use DataLoader, optimizers, loss functions
- The standard training loop pattern

This is the foundation for all PyTorch training - you'll use this pattern in every project!
