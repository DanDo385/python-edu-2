# Project 02: Control Flow and Loops

## Learning Objectives

- Master decision-making in code using conditional statements
- Understand repetitive execution using loops
- Learn loop control statements (break, continue)
- Apply control flow to solve algorithmic problems

## Problem Description

Control flow determines the order in which statements are executed in a program. Python provides powerful constructs for making decisions (`if/elif/else`) and repeating operations (`for` and `while` loops). These are fundamental to implementing algorithms and solving problems programmatically.

This project introduces conditional branching and iteration, which are essential for implementing any non-trivial algorithm.

## Key Concepts

### 1. Conditional Statements (if/elif/else)

Python uses indentation to define code blocks:

```python
if condition:
    # Execute if condition is True
elif another_condition:
    # Execute if first is False and this is True
else:
    # Execute if all conditions are False
```

**Python's Indentation-Based Syntax:**
- Uses indentation (spaces/tabs) instead of braces `{}`
- Forces consistent code style
- Makes code more readable

### 2. Comparison Operators

```
┌──────────┬──────────────────┬──────────────┐
│ Operator │ Meaning          │ Example      │
├──────────┼──────────────────┼──────────────┤
│ ==       │ Equal to         │ 5 == 5 → True│
│ !=       │ Not equal        │ 5 != 3 → True│
│ <        │ Less than        │ 3 < 5 → True │
│ >        │ Greater than     │ 5 > 3 → True │
│ <=       │ Less or equal    │ 3 <= 3 → True│
│ >=       │ Greater or equal │ 5 >= 3 → True│
└──────────┴──────────────────┴──────────────┘
```

### 3. Logical Operators

```
┌──────────┬──────────────┬─────────────────────────────┐
│ Operator │ Meaning      │ Example                     │
├──────────┼──────────────┼─────────────────────────────┤
│ and      │ Both true    │ True and False → False      │
│ or       │ Either true   │ True or False → True         │
│ not      │ Negation      │ not True → False            │
└──────────┴──────────────┴─────────────────────────────┘
```

### 4. For Loops

Iterate over sequences (lists, strings, ranges):

```python
for item in sequence:
    # Process item
```

**Range Function:**
```python
range(stop)              # 0 to stop-1
range(start, stop)       # start to stop-1
range(start, stop, step) # start to stop-1, incrementing by step
```

### 5. While Loops

Repeat while a condition is true:

```python
while condition:
    # Execute while condition is True
```

**Important:** Must ensure condition eventually becomes False to avoid infinite loops!

### 6. Loop Control Statements

```
┌──────────┬─────────────────────────────────────────────┐
│ Statement│ Effect                                      │
├──────────┼─────────────────────────────────────────────┤
│ break    │ Exit the loop immediately                   │
│ continue │ Skip to next iteration                       │
│ pass     │ Do nothing (placeholder)                    │
└──────────┴─────────────────────────────────────────────┘
```

## Solution Approach

### FizzBuzz Problem

A classic programming problem that combines conditionals and loops:

```
For numbers 1 to N:
- If divisible by 3: print "Fizz"
- If divisible by 5: print "Buzz"
- If divisible by both: print "FizzBuzz"
- Otherwise: print the number
```

**Visual Flow:**
```
Start → i = 1
  ↓
Check: i divisible by 3?
  ├─ Yes → Check: also divisible by 5?
  │         ├─ Yes → "FizzBuzz"
  │         └─ No → "Fizz"
  └─ No → Check: divisible by 5?
            ├─ Yes → "Buzz"
            └─ No → print i
  ↓
i = i + 1
  ↓
i <= N? → Yes → Loop back
  └─ No → End
```

### Summing Numbers

Calculate sum of numbers 1 to N:

```
sum = 0
for each number from 1 to N:
    sum = sum + number
return sum
```

**Mathematical Formula:** sum = N × (N + 1) / 2 (but we'll use loops!)

## How Python Uniquely Solves This

### 1. Indentation-Based Blocks

Python uses indentation instead of braces, making code cleaner:

```python
# Python - clean and readable
if x > 0:
    print("Positive")
    if x > 10:
        print("Large")

# vs. Go - braces required
if x > 0 {
    fmt.Println("Positive")
    if x > 10 {
        fmt.Println("Large")
    }
}

# vs. TypeScript - braces required
if (x > 0) {
    console.log("Positive");
    if (x > 10) {
        console.log("Large");
    }
}

# vs. Rust - braces required
if x > 0 {
    println!("Positive");
    if x > 10 {
        println!("Large");
    }
}
```

### 2. For-Each Style Loops

Python's `for` loops iterate over sequences directly:

```python
# Python - iterate directly
for item in [1, 2, 3]:
    print(item)

# vs. Go - need index or range
for i := 0; i < len(items); i++ {
    fmt.Println(items[i])
}
// Or: for i, item := range items { ... }

# vs. TypeScript - similar to Python
for (const item of [1, 2, 3]) {
    console.log(item);
}

# vs. Rust - similar iterator pattern
for item in vec![1, 2, 3] {
    println!("{}", item);
}
```

### 3. Range Function

Python's `range()` is versatile and memory-efficient:

```python
# Python - generates numbers on-demand
for i in range(10):  # 0 to 9
    print(i)

# vs. Go - similar range syntax
for i := range 10 {
    fmt.Println(i)
}

# vs. TypeScript - need to create array
for (let i = 0; i < 10; i++) {
    console.log(i);
}

# vs. Rust - iterator-based
for i in 0..10 {
    println!("{}", i);
}
```

## Comparison with Other Languages

### Go
- **Explicit braces**: Uses `{}` for all blocks
- **No while loop**: Uses `for` for both counting and conditional loops
- **Switch statements**: More powerful than Python's (can use expressions)

### TypeScript
- **C-style syntax**: Familiar to C/Java programmers
- **Optional braces**: Can omit braces for single statements
- **For-of loops**: Similar to Python's iteration style

### Rust
- **Pattern matching**: Powerful `match` expressions (like switch but better)
- **Iterator methods**: Functional-style operations (map, filter, etc.)
- **Ownership**: Loop variables have ownership semantics

## Deliverables

Complete the functions in `test/exercise.py`:

1. **Conditional logic** - Implement if/elif/else branching
2. **FizzBuzz** - Classic problem combining conditionals and loops
3. **Sum calculation** - Use loops to sum numbers
4. **Loop control** - Use break and continue appropriately
5. **Nested loops** - Handle multiple levels of iteration

## Testing

Run the tests with:

```bash
cd test
pytest test.py -v
```

## Next Steps

After completing this project, you'll understand:
- How to make decisions in code
- How to repeat operations efficiently
- How to control loop execution

These skills are essential for implementing algorithms in upcoming projects.

