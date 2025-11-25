# Project 22: Activation Functions and Neural Network Basics

## Learning Objectives

- Understand why activation functions are essential
- Implement common activation functions (ReLU, Tanh, Softmax)
- Learn properties of each activation function
- Understand how activations enable non-linearity
- Visualize activation functions and their derivatives

## Problem Description

Activation functions are what make neural networks powerful. Without them, multiple layers would collapse into a single linear transformation. Activations introduce non-linearity, allowing networks to learn complex patterns.

**Why Activations Matter:**
- **Without activation**: Multiple layers = single linear layer (useless!)
- **With activation**: Each layer adds non-linearity (powerful!)
- **Different activations**: Different properties (gradients, outputs, use cases)

## Key Concepts

### Common Activation Functions

```
Sigmoid:  1 / (1 + e^(-x))     → (0, 1)   - Saturated, smooth
Tanh:     tanh(x)              → (-1, 1)  - Zero-centered, smooth  
ReLU:     max(0, x)            → [0, ∞)   - Simple, no saturation for x>0
Softmax:  exp(x) / sum(exp(x)) → (0, 1)   - Probabilities, sums to 1
```

### Why Non-Linearity is Essential

```
Without activation:
Layer1: y1 = W1*x + b1
Layer2: y2 = W2*y1 + b2 = W2*(W1*x + b1) + b2 = (W2*W1)*x + (W2*b1 + b2)
Result: Still linear! Multiple layers = single layer.

With activation:
Layer1: y1 = ReLU(W1*x + b1)  (non-linear!)
Layer2: y2 = ReLU(W2*y1 + b2)  (non-linear!)
Result: Non-linear transformation! Can learn complex patterns.
```

## Solution Approach

### Understanding Each Activation

1. **Sigmoid**: Bounded, smooth, but saturates (gradients vanish)
2. **Tanh**: Zero-centered, smooth, but still saturates
3. **ReLU**: Simple, no saturation for positive inputs, but dead neurons
4. **Softmax**: Converts logits to probabilities (for multi-class)

## How Python Uniquely Solves This

NumPy provides efficient implementations of these functions, and they're easy to visualize with matplotlib.

## Deliverables

Complete activation function implementations:
1. ReLU and its derivative
2. Tanh and its derivative
3. Softmax (for multi-class)
4. Forward pass through simple network
5. Visualization of activations

## Testing

Run: `pytest test.py -v`



