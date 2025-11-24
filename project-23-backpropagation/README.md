# Project 23: Manual Backpropagation in Neural Networks

## Learning Objectives

- Understand the chain rule of calculus
- Manually compute gradients for a simple neural network
- Trace how errors propagate backward through layers
- Understand the connection between forward and backward passes
- Master gradient computation for multi-layer networks

## Problem Description

Backpropagation is the algorithm that makes neural networks learn. It computes gradients efficiently using the chain rule, allowing us to train deep networks. Understanding it manually is crucial for truly understanding how neural networks work.

**The Core Idea:**
1. Forward pass: Compute predictions
2. Compute loss
3. Backward pass: Propagate errors backward, computing gradients
4. Update weights using gradients

## Key Concepts

### Chain Rule

```
If y = f(g(x)), then:
dy/dx = (dy/dg) * (dg/dx)

In neural networks:
Loss depends on output, output depends on hidden, hidden depends on input
Gradients chain backward through layers
```

### Gradient Flow

```
Output layer: dLoss/dw_output
Hidden layer: dLoss/dw_hidden = dLoss/doutput * doutput/dhidden * dhidden/dw_hidden
Input layer: dLoss/dw_input = ... (chains further back)
```

## Solution Approach

### Understanding from First Principles

1. **Forward pass**: Compute all intermediate values
2. **Loss computation**: Measure error
3. **Backward pass**: Apply chain rule layer by layer
4. **Gradient accumulation**: Combine gradients from multiple paths

## How Python Uniquely Solves This

Python's clear syntax makes the chain rule implementation readable. NumPy handles the computations efficiently.

## Comparison with Other Languages

Similar advantages - Python's clarity helps understand the algorithm.

## Deliverables

Complete backpropagation implementation:
1. Forward pass through 2-layer network
2. Loss computation
3. Backward pass (manual gradient computation)
4. Weight updates
5. Verify with numerical gradients

## Testing

Run: `pytest test.py -v`


