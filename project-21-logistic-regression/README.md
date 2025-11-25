# Project 21: Logistic Regression for Classification

## Learning Objectives

- Understand classification vs regression
- Learn the sigmoid (logistic) function and why it's used
- Master binary cross-entropy loss for classification
- Implement logistic regression from scratch
- Understand decision boundaries and probability interpretation

## Problem Description

Logistic regression extends linear regression to classification. Instead of predicting continuous values, we predict probabilities (0 to 1) and classify based on a threshold (usually 0.5).

**Key Difference from Linear Regression:**
- **Linear regression**: y = w*x + b (unbounded output)
- **Logistic regression**: p = sigmoid(w*x + b) (bounded 0-1, probability)

## Key Concepts

### Sigmoid Function

```
sigmoid(x) = 1 / (1 + e^(-x))

Properties:
- Output range: (0, 1)
- S-shaped curve
- Smooth and differentiable
```

### Binary Cross-Entropy Loss

```
BCE = -mean(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))

Penalizes confident wrong predictions heavily.
```

### Decision Boundary

```
If p >= 0.5: predict class 1
If p < 0.5: predict class 0

The decision boundary is where p = 0.5, i.e., w*x + b = 0
```

## Solution Approach

### Understanding Classification

1. **Probability interpretation**: Model outputs probability of class 1
2. **Sigmoid activation**: Maps unbounded values to (0, 1)
3. **Cross-entropy loss**: Appropriate for probability predictions
4. **Decision threshold**: Convert probabilities to class predictions

## How Python Uniquely Solves This

Python/NumPy makes probability computations natural with vectorized operations and built-in functions like `np.exp()` for sigmoid.

## Deliverables

Complete logistic regression implementation:
1. Sigmoid function
2. Prediction function (with probability output)
3. Binary cross-entropy loss
4. Gradient computation
5. Training loop

## Testing

Run: `pytest test.py -v`



