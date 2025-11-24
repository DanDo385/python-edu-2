# Project 03: Functions and Modular Programming

## Learning Objectives

- Learn to write reusable functions with parameters and return values
- Understand variable scope and the call stack
- Import and use modules (including Python's standard library)
- Write proper documentation strings (docstrings)
- Organize code into modules

## Problem Description

Functions are the building blocks of larger programs. They allow you to:
- **Reuse code** without duplication
- **Organize logic** into manageable pieces
- **Test components** independently
- **Abstract complexity** behind simple interfaces

Python's function system is flexible, supporting default arguments, keyword arguments, variable-length arguments, and more.

## Key Concepts

### 1. Function Definition

```python
def function_name(parameter1, parameter2):
    """Docstring describing what the function does."""
    # Function body
    return result
```

**Components:**
- `def` keyword starts function definition
- Function name follows naming conventions (snake_case)
- Parameters in parentheses (can be empty)
- Docstring describes the function
- `return` statement sends value back (optional)

### 2. Function Call Stack

```
┌─────────────────────┐
│   main()            │ ← Top of stack (currently executing)
├─────────────────────┤
│   function_a()      │ ← Called by main
├─────────────────────┤
│   function_b()      │ ← Called by function_a
└─────────────────────┘ ← Bottom of stack
```

When a function calls another, the new function is "pushed" onto the stack. When it returns, it's "popped" off.

### 3. Variable Scope

```
Global Scope (module level)
    ↓
Function Scope (local variables)
    ↓
Nested Function Scope (if applicable)
```

**Rules:**
- Local variables shadow global ones
- Can read globals, but need `global` keyword to modify
- Parameters are local to the function

### 4. Importing Modules

```python
import math                    # Import entire module
from math import sqrt          # Import specific function
from math import sqrt as sq    # Import with alias
```

**Common Modules:**
- `math`: Mathematical functions
- `random`: Random number generation
- `datetime`: Date and time operations
- `os`: Operating system interface

## Solution Approach

### Factorial Function

Calculate n! = n × (n-1) × ... × 2 × 1

**Iterative approach:**
```
result = 1
for i in range(1, n+1):
    result = result * i
return result
```

### Prime Checker

Check if a number is prime (reusable from Project 02).

### Modular Design

Break problems into smaller functions:
- One function = one responsibility
- Functions call other functions
- Easier to test and debug

## How Python Uniquely Solves This

### 1. Flexible Function Arguments

Python supports multiple argument styles:

```python
# Python - flexible arguments
def func(a, b=10, *args, **kwargs):
    pass

func(1)                    # a=1, b=10
func(1, 2)                 # a=1, b=2
func(1, 2, 3, 4, x=5)      # a=1, b=2, args=(3,4), kwargs={'x':5}

# vs. Go - fixed signatures
func(a int, b int) { }      // Must provide all args

# vs. TypeScript - optional with ?
func(a: number, b?: number) { }  // Similar but less flexible

# vs. Rust - explicit patterns
fn func(a: i32, b: Option<i32>) { }  // More verbose
```

### 2. First-Class Functions

Functions are objects in Python:

```python
# Python - functions are values
def square(x): return x**2
my_func = square
result = my_func(5)

# vs. Go - function values exist but syntax differs
square := func(x int) int { return x * x }

# vs. TypeScript - similar
const square = (x: number) => x * x;

# vs. Rust - closures/closures
let square = |x| x * x;
```

### 3. Docstrings

Python's built-in documentation system:

```python
def my_func():
    """This is a docstring."""
    pass

help(my_func)  # Shows docstring
```

## Comparison with Other Languages

### Go
- **Multiple return values**: Functions can return (val, error) tuples natively
- **No optional parameters**: Must use variadic functions or structs
- **Explicit error handling**: Return errors, no exceptions

### TypeScript
- **Type annotations**: Optional but encouraged for type safety
- **Arrow functions**: Concise syntax for small functions
- **Overloads**: Can define multiple signatures for same function

### Rust
- **Ownership**: Function parameters have ownership semantics
- **Pattern matching**: Can destructure in function parameters
- **No exceptions**: Uses Result<T, E> for error handling

## Deliverables

Complete the functions in `test/exercise.py`:

1. **Basic functions** - Write functions with parameters and returns
2. **Factorial** - Calculate factorial iteratively
3. **Prime checker** - Reusable prime checking function
4. **Module usage** - Import and use standard library modules
5. **Helper functions** - Break problems into smaller functions

## Testing

Run the tests with:

```bash
cd test
pytest test.py -v
```

## Next Steps

After completing this project, you'll understand:
- How to write reusable functions
- How to organize code into modules
- How Python's function system works

These skills are essential for all upcoming projects.

