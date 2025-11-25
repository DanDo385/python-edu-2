# Solution in Words: Logistic Regression

## How to Think About This Problem

### Understanding Classification

Think of classification like sorting items into two bins. Instead of predicting a number, we predict which bin an item belongs to.

**Mental Model:**
```
Regression: Predict exact value (e.g., price = $150)
Classification: Predict category (e.g., spam or not spam)
```

### Understanding Sigmoid

**What we're doing:** Converting unbounded values to probabilities (0 to 1).

**How to think about it:**
- Linear model gives unbounded output (could be -∞ to +∞)
- We need probability (0 to 1)
- Sigmoid squashes values to (0, 1) range
- S-shaped curve: steep in middle, flat at extremes

**Example thought process:**
- Linear output: z = 5 (large positive)
- Sigmoid: sigmoid(5) ≈ 0.993 (high probability of class 1)
- Linear output: z = -5 (large negative)
- Sigmoid: sigmoid(-5) ≈ 0.007 (low probability of class 1)

### Understanding Cross-Entropy Loss

**What we're doing:** Measuring how wrong our probability predictions are.

**How to think about it:**
- If true label is 1 and we predict 0.9 → small error
- If true label is 1 and we predict 0.1 → large error (confidently wrong!)
- Cross-entropy heavily penalizes confident wrong predictions

### Problem-Solving Strategy

1. **Apply linear model**: z = w*x + b
2. **Apply sigmoid**: p = sigmoid(z) (probability)
3. **Compute loss**: Cross-entropy
4. **Compute gradients**: How to improve
5. **Update parameters**: Gradient descent




