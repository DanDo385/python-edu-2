# Project 04: Data Structures I – Lists and Tuples

## Learning Objectives

- Understand Python's sequence data structures (lists and tuples)
- Learn indexing, slicing, and iteration
- Master list methods (append, pop, sort, etc.)
- Understand when to use lists vs tuples
- Use list comprehensions for concise code

## Problem Description

Lists and tuples are Python's primary sequence types. Lists are mutable (can be changed), while tuples are immutable (cannot be changed after creation). Understanding when to use each is crucial for writing efficient Python code.

## Key Concepts

### Lists vs Tuples

```
┌─────────────┬──────────────┬─────────────────┐
│ Feature     │ List         │ Tuple           │
├─────────────┼──────────────┼─────────────────┤
│ Mutability  │ Mutable      │ Immutable       │
│ Syntax      │ [1, 2, 3]    │ (1, 2, 3)       │
│ Use Case    │ Collections  │ Fixed records   │
│ Performance │ Slower       │ Faster          │
└─────────────┴──────────────┴─────────────────┘
```

### Indexing and Slicing

```python
my_list = [10, 20, 30, 40, 50]
my_list[0]      # 10 (first element)
my_list[-1]     # 50 (last element)
my_list[1:4]    # [20, 30, 40] (slice)
my_list[:3]     # [10, 20, 30] (start to index 3)
my_list[2:]     # [30, 40, 50] (index 2 to end)
```

## How Python Uniquely Solves This

### 1. Negative Indexing

Python allows negative indices to count from the end:

```python
# Python - intuitive negative indexing
my_list[-1]  # Last element

# vs. Go - need length calculation
mySlice[len(mySlice)-1]

# vs. TypeScript - similar to Python
myArray[myArray.length - 1]

# vs. Rust - need explicit calculation
my_vec[my_vec.len() - 1]
```

### 2. List Comprehensions

Python's concise way to create lists:

```python
# Python - elegant and readable
squares = [x**2 for x in range(10)]

# vs. Go - verbose
var squares []int
for x := 0; x < 10; x++ {
    squares = append(squares, x*x)
}

# vs. TypeScript - similar with map
const squares = Array.from({length: 10}, (_, i) => i * i);

# vs. Rust - iterator-based
let squares: Vec<i32> = (0..10).map(|x| x * x).collect();
```

## Comparison with Other Languages

### Go
- **Slices**: Similar to lists but with fixed capacity
- **Arrays**: Fixed-size, different from slices
- **No negative indexing**: Must calculate manually

### TypeScript
- **Arrays**: Similar to Python lists
- **Readonly arrays**: Similar to tuples
- **Array methods**: map, filter, reduce similar to comprehensions

### Rust
- **Vectors (Vec)**: Similar to lists
- **Arrays**: Fixed-size, stack-allocated
- **Ownership**: Important consideration when manipulating

## Deliverables

Complete functions for:
1. Finding min/max/average in lists
2. Merging sorted lists
3. List manipulation (append, pop, sort)
4. List comprehensions
5. Tuple operations

## Testing

Run: `pytest test.py -v`

