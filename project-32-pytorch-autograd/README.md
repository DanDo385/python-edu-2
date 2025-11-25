# Project 32: PyTorch Autograd – Automatic Differentiation in Practice

## Learning Objectives

- Understand PyTorch's automatic differentiation (autograd) system
- Learn how to enable gradient computation with `requires_grad=True`
- Compute gradients using `.backward()`
- Access gradients via `.grad` attribute
- Understand when to use `torch.no_grad()` context
- Link back to custom autodiff from Project 24

## Problem Description

Automatic differentiation is PyTorch's superpower - it computes gradients automatically! This eliminates the need for manual gradient computation (like we did in Project 25).

**What is Autograd?**
- Automatically computes gradients of any computation
- Builds computation graph on the fly
- Enables training neural networks without manual backprop

**Why Autograd Matters:**
- No more manual gradient computation!
- Works for any differentiable function
- Essential for training neural networks

## Key Concepts

### Enabling Gradient Computation

```python
# Create tensor with gradient tracking
x = torch.tensor(2.0, requires_grad=True)

# Perform operations
y = x ** 2 + 3 * x + 1

# Compute gradient
y.backward()

# Access gradient
print(x.grad)  # dy/dx at x=2
```

### Computation Graph

PyTorch builds a computation graph automatically:

```
x (requires_grad=True)
  ↓
x**2
  ↓
x**2 + 3*x + 1
  ↓
y
```

When you call `y.backward()`, PyTorch traverses this graph to compute gradients.

### Gradient Accumulation

**Important:** Gradients accumulate! You must zero them between steps:

```python
optimizer.zero_grad()  # Zero gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update weights
```

### Disabling Gradient Tracking

**When to disable:**
- During inference (no need for gradients)
- When updating weights (avoid tracking weight updates)

```python
with torch.no_grad():
    # Operations here won't track gradients
    predictions = model(x)
```

## Solution Approach

### Understanding Autograd

1. **Enable tracking**: Set `requires_grad=True`
2. **Perform operations**: Build computation graph
3. **Compute gradients**: Call `.backward()`
4. **Access gradients**: Use `.grad` attribute
5. **Zero gradients**: Reset for next iteration

### Gradient Computation

- **Scalar output**: Call `.backward()` directly
- **Vector output**: Need to provide gradient vector (or use `.backward(torch.ones_like(output))`)

## How Python Uniquely Solves This

### 1. Seamless Integration

```python
# Automatic differentiation
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x
y.backward()
print(x.grad)  # 3*x^2 + 2 = 14 at x=2
```

### 2. Context Managers

```python
# Easy to disable gradients
with torch.no_grad():
    # No gradient tracking
    result = model(x)
```

## Deliverables

Complete functions for:
1. Computing gradients of simple functions
2. Computing gradients of multivariate functions
3. Using `torch.no_grad()` context
4. Zeroing gradients
5. Verifying gradient correctness

## Testing

Run: `pytest test.py -v`

Tests verify:
- Gradients compute correctly for known functions
- Multiple gradients work (multivariate)
- `no_grad` context disables tracking
- Gradients accumulate correctly

## Next Steps

After completing this project, you'll understand:
- How PyTorch computes gradients automatically
- When to enable/disable gradient tracking
- How to use gradients for training

This is the foundation for training models - all PyTorch training uses autograd!
