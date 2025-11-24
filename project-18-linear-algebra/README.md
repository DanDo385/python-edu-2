# Project 18: Linear Algebra Essentials for ML

## Learning Objectives

- Understand vectors and matrices as fundamental ML building blocks
- Master dot products and their role in neural networks
- Learn matrix-vector multiplication (the core of neural network layers)
- Understand linear systems and how to solve them
- Grasp how linear algebra underpins all ML model calculations

## Problem Description

Linear algebra is the mathematical foundation of machine learning. Every neural network layer, every prediction, every optimization step involves linear algebra operations. Understanding these operations from first principles is essential.

**Why Linear Algebra Matters:**
- **Neural networks** = sequences of matrix multiplications
- **Predictions** = input × weights (matrix multiplication)
- **Gradients** = computed using linear algebra
- **Data** = represented as vectors and matrices

## Key Concepts

### Vectors

```
Vector: [1, 2, 3]  → A point in n-dimensional space
        Can represent: features, weights, biases, gradients
```

### Dot Product

```
Dot product: a · b = sum(a[i] * b[i])
            Measures similarity, computes weighted sums
```

### Matrix-Vector Multiplication

```
Matrix × Vector = Vector
(m, n) × (n,) = (m,)

This is how neural networks compute outputs!
```

### Linear Systems

```
Ax = b  → Solve for x
Used in: least squares, optimization, solving for optimal weights
```

## Solution Approach

### Understanding from First Principles

1. **Vectors as Data**: Each data point is a vector of features
2. **Matrices as Transformations**: Weights transform inputs to outputs
3. **Dot Products as Similarity**: Measure how aligned vectors are
4. **Matrix Multiplication as Composition**: Chain transformations

### Visual Thinking

```
Input vector:    [x1, x2, x3]     (features)
Weight matrix:   [[w11, w12],     (learned parameters)
                  [w21, w22],
                  [w31, w32]]
Output vector:   [y1, y2]          (predictions)

y1 = x1*w11 + x2*w21 + x3*w31  (weighted sum!)
y2 = x1*w12 + x2*w22 + x3*w32
```

## How Python Uniquely Solves This

### 1. NumPy's Linear Algebra Module

Python provides `numpy.linalg` with optimized linear algebra operations:

```python
# Python - clean and efficient
result = np.linalg.solve(A, b)  # Solve Ax = b
eigenvals = np.linalg.eig(A)    # Eigenvalues

# vs. Other languages - more verbose or less optimized
```

### 2. Intuitive Matrix Operations

```python
# Python - reads like math
y = X @ w + b  # Matrix multiplication + bias

# Clear and concise, matches mathematical notation
```

## Comparison with Other Languages

### Go
- **No built-in linear algebra**: Must use external libraries
- **More verbose**: Explicit type handling required

### TypeScript
- **Limited support**: Not designed for scientific computing
- **Web-focused**: Better for frontend applications

### Rust
- **Performance**: Excellent but requires more code
- **Ecosystem**: Less mature for linear algebra

## Deliverables

Complete functions for:
1. Dot product computation
2. Matrix-vector multiplication
3. Solving linear systems
4. Vector projections
5. Computing matrix properties (determinant, inverse for small matrices)

## Testing

Run: `pytest test.py -v`

## Next Steps

After completing this project, you'll understand:
- How neural networks are fundamentally linear algebra operations
- Why matrix multiplication is the core of ML
- How to reason about shapes and dimensions

This knowledge directly applies to implementing neural networks from scratch.


