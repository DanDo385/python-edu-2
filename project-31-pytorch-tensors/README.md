# Project 31: PyTorch Tensors and GPU Fundamentals

## Learning Objectives

- Understand PyTorch tensors and how they differ from NumPy arrays
- Learn basic tensor operations (creation, indexing, reshaping)
- Understand how to move tensors to GPU for acceleration
- Learn tensor type conversions and shape manipulation
- Set up reproducible experiments with random seeds
- Compare CPU vs GPU performance for large operations

## Problem Description

PyTorch is the industry-standard deep learning framework. It builds on NumPy concepts but adds:
- **Automatic differentiation** (coming in Project 32)
- **GPU acceleration** (this project)
- **Neural network modules** (coming in Project 33)
- **Much less code** than pure NumPy

**PyTorch Tensors vs NumPy Arrays:**
- Similar API (familiar if you know NumPy)
- Can run on GPU (much faster!)
- Support automatic differentiation
- Seamless conversion between NumPy and PyTorch

## Key Concepts

### Tensor Creation

```python
import torch

# From Python list
tensor = torch.tensor([1, 2, 3, 4, 5])

# Pre-filled tensors
zeros = torch.zeros(5)
ones = torch.ones(5)
rand = torch.randn(5)  # Random values
```

### GPU Acceleration

```python
# Check if GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move tensor to GPU
tensor_gpu = tensor.to(device)
# or
tensor_gpu = tensor.cuda()  # Shorthand
```

### Tensor Operations

```python
# Element-wise operations (same as NumPy)
result = tensor1 + tensor2
result = tensor1 * 2
result = tensor1 ** 2

# Matrix multiplication
result = tensor1 @ tensor2
# or
result = torch.matmul(tensor1, tensor2)
```

### Shape Manipulation

```python
# Get shape
shape = tensor.shape  # or tensor.size()

# Reshape
reshaped = tensor.reshape(2, 3)
reshaped = tensor.view(2, 3)  # Similar, but shares memory

# Transpose
transposed = tensor.T
```

## Solution Approach

### Understanding Tensors

1. **Create tensors**: From lists, arrays, or using constructors
2. **Perform operations**: Element-wise, matrix ops, etc.
3. **Move to GPU**: For acceleration (if available)
4. **Compare performance**: CPU vs GPU timing

### GPU Usage

- **Check availability**: `torch.cuda.is_available()`
- **Move tensors**: `.to(device)` or `.cuda()`
- **Operations on GPU**: Automatically faster for large tensors
- **Move back to CPU**: `.cpu()` for NumPy conversion

## How Python Uniquely Solves This

### 1. Seamless GPU Integration

```python
# Python - easy GPU usage
tensor = torch.randn(1000, 1000)
tensor_gpu = tensor.cuda()  # Move to GPU
result = tensor_gpu @ tensor_gpu  # Fast GPU computation
```

### 2. NumPy Interoperability

```python
# Easy conversion
numpy_array = tensor.numpy()  # PyTorch → NumPy
tensor = torch.from_numpy(numpy_array)  # NumPy → PyTorch
```

### 3. Intuitive API

PyTorch API is similar to NumPy, making transition easy.

## Comparison with Other Languages

### Go
- **No PyTorch equivalent**: Would need to use C libraries
- **GPU programming**: Much more complex

### TypeScript
- **Limited ML support**: Not designed for deep learning
- **No GPU acceleration**: Limited to CPU

### Rust
- **Performance**: Fast but less mature ML ecosystem
- **GPU**: More complex than PyTorch

## Deliverables

Complete functions for:
1. Creating tensors from various sources
2. Basic tensor operations
3. Shape manipulation
4. GPU operations (if available)
5. Performance comparison (CPU vs GPU)

## Testing

Run: `pytest test.py -v`

Tests verify:
- Tensor creation works correctly
- Operations produce expected results
- GPU operations work (if GPU available)
- Shape manipulation is correct

## Next Steps

After completing this project, you'll understand:
- How PyTorch tensors work
- How to use GPU acceleration
- Basic tensor operations

This foundation is essential - all PyTorch projects use tensors!
