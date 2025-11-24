# Project 16: NumPy 101 – Arrays and Vectorized Operations

## Learning Objectives

- Understand why NumPy exists and how it differs from Python lists
- Create and manipulate NumPy arrays
- Learn vectorized operations (operating on entire arrays without loops)
- Understand array shapes and dimensions
- Master basic array operations (indexing, slicing, reshaping)

## Problem Description

NumPy (Numerical Python) is the foundation of scientific computing in Python. While Python lists are flexible, they're slow for numerical operations. NumPy provides efficient, vectorized operations on arrays, making it essential for machine learning and data science.

**Why NumPy?**
- **Speed**: Operations are implemented in C, 10-100x faster than Python loops
- **Memory efficiency**: Arrays store data contiguously in memory
- **Vectorization**: Operate on entire arrays at once, no loops needed
- **Mathematical operations**: Built-in support for linear algebra, statistics, etc.

## Key Concepts

### NumPy Arrays vs Python Lists

```
Python List:
[1, 2, 3, 4, 5]  → Each element is a Python object (slow, flexible)

NumPy Array:
[1 2 3 4 5]      → Homogeneous data, contiguous memory (fast, efficient)
```

### Array Creation

```python
import numpy as np

# From Python list
arr = np.array([1, 2, 3, 4, 5])

# Pre-filled arrays
zeros = np.zeros(5)        # [0. 0. 0. 0. 0.]
ones = np.ones(5)         # [1. 1. 1. 1. 1.]
range_arr = np.arange(5)   # [0 1 2 3 4]
```

### Shape and Dimensions

```
1D Array (vector):     [1 2 3 4]           → shape: (4,)
2D Array (matrix):     [[1 2]              → shape: (2, 2)
                        [3 4]]
3D Array:              [[[1 2] [3 4]]]      → shape: (1, 2, 2)
```

### Vectorized Operations

```python
# Instead of looping:
result = []
for x in list1:
    result.append(x * 2)

# NumPy vectorization:
result = arr * 2  # Operates on entire array at once!
```

## Solution Approach

### Understanding Vectorization

Vectorization means applying operations to entire arrays simultaneously, rather than element-by-element. This is possible because:

1. **Homogeneous data**: All elements are the same type
2. **Contiguous memory**: Data stored sequentially
3. **Optimized C code**: Operations compiled to machine code

### Array Operations

```
┌─────────────────────────────────────────┐
│         NumPy Array Operations          │
├─────────────────────────────────────────┤
│  Creation: np.array(), np.zeros(), etc. │
│  Indexing: arr[0], arr[1:3]             │
│  Math: arr + 5, arr * 2, arr ** 2      │
│  Shape: arr.shape, arr.reshape()       │
│  Stats: np.mean(), np.sum(), np.max()  │
└─────────────────────────────────────────┘
```

## How Python Uniquely Solves This

### 1. Seamless Integration

NumPy arrays integrate naturally with Python:

```python
# Python - easy conversion
import numpy as np
python_list = [1, 2, 3]
numpy_array = np.array(python_list)  # Convert easily

# vs. Go - more verbose
// Need explicit type conversions and manual memory management

# vs. TypeScript - similar but less mature ecosystem
const arr = new Float64Array([1, 2, 3]);

# vs. Rust - powerful but more complex
let arr = Array1::from_vec(vec![1.0, 2.0, 3.0]);
```

### 2. Broadcasting

NumPy automatically handles operations between arrays of different shapes:

```python
# Python - automatic broadcasting
arr = np.array([[1, 2], [3, 4]])
arr + 10  # Adds 10 to every element automatically

# Other languages require explicit loops or special functions
```

### 3. Expressive Syntax

NumPy operations read like mathematical notation:

```python
# Python - reads like math
result = (arr - mean) / std  # Normalization formula

# vs. explicit loops in other languages
```

## Comparison with Other Languages

### Go
- **No built-in NumPy equivalent**: Must use libraries or implement manually
- **Explicit types**: More verbose but type-safe
- **Performance**: Fast but requires more code

### TypeScript
- **Limited numerical computing**: Not designed for scientific computing
- **Web-focused**: Better for browser/Node.js applications
- **Type safety**: Optional types help catch errors

### Rust
- **Performance**: Can match or exceed NumPy performance
- **Complexity**: Steeper learning curve, more verbose
- **Ecosystem**: Less mature scientific computing libraries

## Deliverables

Complete functions for:
1. Creating arrays from lists and using array constructors
2. Array indexing and slicing
3. Vectorized arithmetic operations
4. Array reshaping and shape manipulation
5. Basic statistical operations (mean, sum, max, min)

## Testing

Run: `pytest test.py -v`

## Next Steps

After completing this project, you'll understand:
- How NumPy arrays differ from Python lists
- Why vectorization is crucial for performance
- Basic array manipulation operations

This foundation is essential for all machine learning work - every ML library (PyTorch, TensorFlow, scikit-learn) builds on NumPy concepts.

