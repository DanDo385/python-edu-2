# Project 19: Gradient Descent Basics

## Learning Objectives

- Understand optimization: finding minimum of a function
- Learn what gradients are and how they point to steepest ascent
- Master gradient descent: moving opposite to gradient to minimize
- Understand learning rate and its critical role
- Implement gradient descent from scratch for simple functions

## Problem Description

Gradient descent is the optimization algorithm that powers all machine learning. Every neural network, every model training, uses gradient descent (or variants) to learn optimal parameters.

**The Core Idea:**
1. Start with random parameters
2. Compute gradient (direction of steepest increase)
3. Move opposite to gradient (toward minimum)
4. Repeat until converged

**Why It Works:**
- Gradient points "uphill" (direction of increase)
- Moving opposite (downhill) decreases the function
- With small steps, we find the minimum

## Key Concepts

### Functions and Derivatives

```
Function: f(x) = x²
Derivative: f'(x) = 2x
Gradient: Points in direction of steepest increase
```

### Gradient Descent Algorithm

```
1. Initialize: x = random value
2. Loop:
   a. Compute gradient: grad = df/dx
   b. Update: x = x - learning_rate * grad
   c. Check convergence
3. Return optimal x
```

### Learning Rate

```
Learning rate (α): Step size
- Too small: Slow convergence
- Too large: May overshoot, diverge
- Just right: Fast, stable convergence
```

## Solution Approach

### Understanding from First Principles

1. **Functions have minima**: Points where function is lowest
2. **Derivatives show direction**: Positive = increasing, Negative = decreasing
3. **Gradient = derivative**: Points toward increase
4. **Move opposite**: To find minimum, go opposite to gradient
5. **Small steps**: Learning rate controls step size

### Visual Thinking

```
Function: f(x) = (x-3)² + 4
Minimum at x = 3

Gradient descent:
Start: x = 0
Step 1: grad = -6, x = 0 - 0.1*(-6) = 0.6
Step 2: grad = -4.8, x = 0.6 - 0.1*(-4.8) = 1.08
...
Converges to x ≈ 3
```

## How Python Uniquely Solves This

### 1. Numerical Differentiation

Python can compute gradients numerically:

```python
# Compute gradient numerically
def gradient(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)
```

### 2. Clean Implementation

```python
# Readable gradient descent
for epoch in range(epochs):
    grad = compute_gradient(f, x)
    x = x - learning_rate * grad
```

## Deliverables

Complete functions for:
1. Computing gradients (numerical and analytical)
2. Gradient descent for 1D functions
3. Visualizing gradient descent path
4. Understanding learning rate effects
5. Handling convergence criteria

## Testing

Run: `pytest test.py -v`

## Next Steps

After completing this project, you'll understand:
- How optimization works in ML
- Why gradient descent is fundamental
- How learning rate affects training

This directly applies to training neural networks - gradient descent is how they learn!



