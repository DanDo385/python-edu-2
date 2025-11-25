# Project 24: Building an Autodiff Engine from Scratch

## Learning Objectives

- Understand automatic differentiation (how PyTorch/TensorFlow compute gradients)
- Build a minimal autograd system
- Learn computational graph representation
- Implement forward and backward passes
- Understand how frameworks automate gradient computation

## Problem Description

Automatic differentiation (autodiff) is what makes modern deep learning possible. Instead of manually computing gradients (like in Project 23), autodiff automatically computes them for any computation. This project builds a mini version of what PyTorch does.

**The Core Idea:**
- Build a computation graph as operations happen
- Store how to compute gradients for each operation
- Traverse graph backward to compute all gradients

## Key Concepts

### Computational Graph

```
x → * → z
y ↗   ↘
      + → result
```

### Autograd Components

```
Tensor: Stores value + gradient + operation
Operation: Knows how to compute forward + backward
Graph: Tracks dependencies between tensors
```

## Solution Approach

### Building Blocks

1. **Tensor class**: Stores value, gradient, and operation
2. **Operations**: Addition, multiplication with backward functions
3. **Backward pass**: Traverse graph, compute gradients
4. **Gradient accumulation**: Handle multiple paths to same tensor

## How Python Uniquely Solves This

Python's object-oriented nature and operator overloading make autodiff natural. We can override `__add__`, `__mul__` to build the graph automatically.

## Deliverables

Complete autodiff engine:
1. Tensor class with value and grad
2. Operations (add, multiply) that build graph
3. Backward method to compute gradients
4. Test on simple computations

## Testing

Run: `pytest test.py -v`



