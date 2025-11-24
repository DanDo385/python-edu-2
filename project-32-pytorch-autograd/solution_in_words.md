# Solution Explanation: PyTorch Autograd

## Overview

PyTorch's automatic differentiation (autograd) is its superpower. It computes gradients automatically, eliminating the need for manual gradient computation that we did in pure NumPy.

## Key Concepts Explained

### What is Autograd?

**Automatic Differentiation:**
- Computes gradients of any computation automatically
- Builds computation graph on the fly
- No manual backprop needed!

**How it Works:**
1. Track operations when `requires_grad=True`
2. Build computation graph automatically
3. Compute gradients when `.backward()` is called
4. Access gradients via `.grad` attribute

### Enabling Gradient Tracking

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x
y.backward()
print(x.grad)  # Gradient computed automatically!
```

**Key Points:**
- `requires_grad=True` enables tracking
- Operations create computation graph
- `.backward()` computes gradients
- `.grad` contains gradient values

### Computation Graph

PyTorch builds a graph automatically:

```
x (requires_grad=True)
  ↓
x**3
  ↓
x**3 + 2*x
  ↓
y
```

When you call `y.backward()`, PyTorch:
1. Traverses graph backward
2. Applies chain rule
3. Computes gradients
4. Stores in `.grad`

### Gradient Accumulation

**Important:** Gradients accumulate!

```python
x = torch.tensor(2.0, requires_grad=True)

y1 = x ** 2
y1.backward()  # x.grad = 4

y2 = x ** 2
y2.backward()  # x.grad = 8 (accumulated!)
```

**Solution:** Zero gradients between steps:
```python
x.grad.zero_()  # Reset gradients
```

### Disabling Gradient Tracking

**When to disable:**
- During inference (no gradients needed)
- When updating weights (avoid tracking updates)

```python
with torch.no_grad():
    predictions = model(x)  # No gradient tracking
```

### Vector Outputs

**For vector outputs, provide gradient vector:**

```python
x = torch.tensor(1.0, requires_grad=True)
y = torch.stack([x**2, x**3])  # Vector output

# Option 1: Sum then backward
y.sum().backward()

# Option 2: Provide gradient vector
y.backward(torch.ones_like(y))
```

## Implementation Details

### Simple Function Gradient

```python
def compute_gradient_simple_function(x_value):
    x = torch.tensor(x_value, requires_grad=True)
    y = x ** 3 + 2 * x
    y.backward()
    return x.grad.item()
```

**Key Points:**
- Create tensor with `requires_grad=True`
- Perform operations (graph built automatically)
- Call `.backward()` to compute gradients
- Access via `.grad`

### Multivariate Gradients

```python
def compute_gradient_multivariate(a_value, b_value):
    a = torch.tensor(a_value, requires_grad=True)
    b = torch.tensor(b_value, requires_grad=True)
    z = 3 * a + 4 * b ** 2
    z.backward()
    return a.grad.item(), b.grad.item()
```

**Key Points:**
- Multiple tensors can have `requires_grad=True`
- Single `.backward()` computes all gradients
- Each tensor's `.grad` contains its gradient

### No Grad Context

```python
with torch.no_grad():
    # Operations here don't track gradients
    result = model(x)
```

**Key Points:**
- Use context manager `torch.no_grad()`
- Disables gradient tracking inside context
- Useful for inference and weight updates

## Common Pitfalls

1. **Forgetting requires_grad**: No gradients computed
2. **Not zeroing gradients**: Gradients accumulate incorrectly
3. **Backward on non-scalar**: Need to provide gradient vector
4. **Using no_grad incorrectly**: Disables tracking when needed

## Real-World Application

**In Training:**
```python
# Training loop
for batch in dataloader:
    optimizer.zero_grad()      # Zero gradients
    predictions = model(x)    # Forward pass
    loss = criterion(predictions, y)  # Compute loss
    loss.backward()            # Compute gradients (autograd!)
    optimizer.step()           # Update weights
```

**Key Pattern:**
1. Zero gradients
2. Forward pass (builds graph)
3. Compute loss
4. Backward pass (computes gradients)
5. Update weights

## Why This Matters

Autograd is **essential** for PyTorch:
- Enables training without manual gradients
- Works for any differentiable function
- Much easier than manual backprop
- Industry standard approach

**Comparison to Project 24:**
- Project 24: Built autodiff from scratch (educational)
- Project 32: Use PyTorch's autograd (practical)
- Same concept, but PyTorch does it automatically!

This project establishes autograd understanding, which is used in every PyTorch training loop!
