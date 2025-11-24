# Project 20: Linear Regression from Scratch

## Learning Objectives

- Implement linear regression from first principles
- Understand the relationship between data and model
- Master Mean Squared Error (MSE) loss function
- Compute gradients for linear regression
- Train a model using gradient descent
- Make predictions and evaluate performance

## Problem Description

Linear regression is the simplest machine learning algorithm, but it demonstrates all the core concepts:
- **Model**: y = w*x + b (line)
- **Loss**: Mean Squared Error (MSE)
- **Optimization**: Gradient descent
- **Training**: Minimize loss to find best w and b

This project brings together everything learned so far: NumPy arrays, linear algebra, and gradient descent, to build a complete ML model from scratch.

## Key Concepts

### Linear Model

```
y = w * x + b

Where:
- x: input feature
- y: predicted output
- w: weight (slope)
- b: bias (y-intercept)
```

### Mean Squared Error (MSE)

```
MSE = (1/n) * sum((y_pred - y_true)²)

Measures average squared difference between predictions and true values.
```

### Gradients

```
dMSE/dw = (2/n) * sum(x * (y_pred - y_true))
dMSE/db = (2/n) * sum(y_pred - y_true)
```

### Training Loop

```
1. Initialize w, b randomly
2. For each epoch:
   a. Make predictions: y_pred = w*x + b
   b. Compute loss: MSE
   c. Compute gradients
   d. Update: w = w - lr * dMSE/dw
              b = b - lr * dMSE/db
3. Return trained w, b
```

## Solution Approach

### Understanding from First Principles

1. **Model represents relationship**: Line that fits data
2. **Loss measures error**: How wrong are predictions?
3. **Gradients show direction**: How to improve w and b?
4. **Gradient descent optimizes**: Find best w and b
5. **Training converges**: Loss decreases, model improves

### Visual Thinking

```
Data points: (x1, y1), (x2, y2), ...
Initial line: y = random_w * x + random_b (poor fit)
After training: y = learned_w * x + learned_b (good fit)

Training process:
Epoch 1: Loss = 10.5, w=0.5, b=1.0
Epoch 2: Loss = 8.2,  w=0.7, b=0.8
Epoch 3: Loss = 6.1,  w=0.9, b=0.6
...
Epoch 100: Loss = 0.3, w=2.0, b=1.0 (converged!)
```

## How Python Uniquely Solves This

### 1. NumPy Vectorization

Python/NumPy makes vectorized operations natural:

```python
# Python - vectorized and clean
y_pred = w * x + b  # Works on entire array at once
loss = np.mean((y_pred - y_true)**2)  # Vectorized MSE

# vs. Other languages - need explicit loops
```

### 2. Clean Training Loop

```python
# Python - readable training loop
for epoch in range(epochs):
    y_pred = w * x + b
    loss = compute_loss(y_pred, y_true)
    grad_w, grad_b = compute_gradients(x, y_pred, y_true)
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
```

## Comparison with Other Languages

### Go
- **More verbose**: Need explicit loops for vectorization
- **Type safety**: Explicit types required throughout

### TypeScript
- **Limited**: Not designed for numerical computing
- **Web-focused**: Better for other applications

### Rust
- **Performance**: Excellent but more complex
- **Ecosystem**: Less mature for ML

## Deliverables

Complete linear regression implementation:
1. Prediction function (forward pass)
2. Loss computation (MSE)
3. Gradient computation
4. Training loop with gradient descent
5. Evaluation metrics

## Testing

Run: `pytest test.py -v`

## Next Steps

After completing this project, you'll understand:
- How ML models learn from data
- The complete training process
- How to implement algorithms from scratch

This is a major milestone - you've built your first ML model! This foundation applies to all neural networks.


