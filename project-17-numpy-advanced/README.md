# Project 17: Advanced NumPy – Broadcasting and Matrix Operations

## Learning Objectives

- Master NumPy broadcasting rules and patterns
- Understand matrix multiplication and linear algebra operations
- Learn to eliminate loops using broadcasting
- Perform efficient matrix operations
- Understand when broadcasting applies vs when it doesn't

## Problem Description

Broadcasting is NumPy's powerful feature that allows operations between arrays of different shapes. Understanding broadcasting is crucial for writing efficient ML code - it's how we apply operations across batches, add biases to matrices, and more.

**What is Broadcasting?**
Broadcasting automatically expands smaller arrays to match larger arrays' shapes, enabling element-wise operations without explicit loops or reshaping.

## Key Concepts

### Broadcasting Rules

NumPy broadcasts arrays by:
1. **Aligning dimensions** from the right
2. **Expanding dimensions** of size 1
3. **Repeating values** along expanded dimensions

```
Example:
Array A: (5, 3)      → Shape: 5 rows, 3 columns
Array B: (3,)         → Shape: 3 elements (treated as 1 row, 3 columns)
Result:  (5, 3)      → B is "broadcast" to match A
```

### Matrix Multiplication

```
Matrix multiplication (dot product):
A @ B  or  np.dot(A, B)

Rules:
- A: (m, n)
- B: (n, p)  
- Result: (m, p)

Each element (i, j) = sum of A[i, :] * B[:, j]
```

### Common Broadcasting Patterns

```python
# Pattern 1: Add vector to each row of matrix
matrix + vector  # Vector broadcast across rows

# Pattern 2: Multiply matrix by scalar
matrix * scalar  # Scalar broadcast to all elements

# Pattern 3: Operations between compatible shapes
(5, 3) + (1, 3)  # Second array broadcast to (5, 3)
```

## Solution Approach

### Understanding Broadcasting Step-by-Step

1. **Identify shapes**: What are the dimensions?
2. **Check compatibility**: Can they be broadcast?
3. **Visualize expansion**: How does the smaller array expand?
4. **Apply operation**: Perform element-wise operation

### Matrix Operations

Matrix multiplication is the foundation of neural networks:
- Input data × weights = predictions
- Gradients flow through matrix multiplications
- Understanding shapes is critical

## How Python Uniquely Solves This

### 1. Automatic Broadcasting

NumPy automatically handles broadcasting:

```python
# Automatic broadcasting
matrix = np.array([[1, 2], [3, 4]])  # Shape (2, 2)
vector = np.array([10, 20])          # Shape (2,)
result = matrix + vector              # Automatically broadcasts!
```

### 2. Intuitive Matrix Operations

```python
# Clean syntax for matrix multiplication
result = A @ B  # Matrix multiplication
```

## Deliverables

Complete functions for:
1. Broadcasting operations (add vector to matrix rows)
2. Matrix multiplication
3. Transpose operations
4. Computing distances using broadcasting
5. Advanced array manipulations

## Testing

Run: `pytest test.py -v`

## Next Steps

After completing this project, you'll understand:
- How broadcasting eliminates the need for loops
- Matrix operations fundamental to neural networks
- Shape compatibility and how to reason about it

This knowledge is essential for implementing neural networks from scratch in upcoming projects.

