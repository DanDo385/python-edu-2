# Project 25: Training a Shallow Neural Network (MLP) from Scratch

## Learning Objectives

- Combine all previous knowledge into a complete neural network
- Implement multi-layer perceptron (MLP) architecture
- Train network using gradient descent and backpropagation
- Understand forward pass, loss computation, backward pass, weight updates
- Solve a non-linear problem (XOR) that linear models can't solve

## Problem Description

This project brings together everything: NumPy arrays, linear algebra, activation functions, gradient descent, and backpropagation. You'll build and train a neural network from scratch to solve the XOR problem - a classic demonstration that neural networks can learn non-linear patterns.

**The XOR Problem:**
- Input: (0,0), (0,1), (1,0), (1,1)
- Output: 0, 1, 1, 0 (XOR truth table)
- **Not linearly separable!** Requires a hidden layer.

## Key Concepts

### MLP Architecture

```
Input → Hidden Layer (ReLU) → Output Layer (Sigmoid) → Prediction
```

### Training Process

```
1. Forward pass: Compute predictions
2. Compute loss: Measure error
3. Backward pass: Compute gradients
4. Update weights: Gradient descent
5. Repeat until converged
```

## Solution Approach

### Understanding the Complete Pipeline

1. **Initialize weights**: Random small values
2. **Forward pass**: Input → Hidden → Output
3. **Loss**: Binary cross-entropy
4. **Backward pass**: Compute gradients for all weights
5. **Update**: Adjust weights to reduce loss

## How Python Uniquely Solves This

NumPy makes the implementation clean and vectorized. The entire network can be implemented in pure NumPy, demonstrating all concepts clearly.

## Comparison with Other Languages

Similar advantages - Python's clarity helps understand the complete system.

## Deliverables

Complete neural network implementation:
1. MLP class with forward pass
2. Loss computation
3. Backward pass (gradient computation)
4. Training loop
5. Solve XOR problem

## Testing

Run: `pytest test.py -v`

## Next Steps

After completing this project, you'll have built a complete neural network from scratch! This is a major milestone. Next, we'll use PyTorch to make this easier and scale to larger networks.


