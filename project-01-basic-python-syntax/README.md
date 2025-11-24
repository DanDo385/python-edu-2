# Project 01: Basic Python Syntax and Variables

## Learning Objectives

- Introduce Python syntax and the interactive environment
- Understand variables and basic I/O operations
- Learn primitive data types (integers, floats, strings, booleans)
- Master variable assignment and naming rules
- Perform basic arithmetic operations

## Problem Description

Python is a dynamically-typed, interpreted language that emphasizes readability and simplicity. Unlike statically-typed languages, Python determines variable types at runtime, making it flexible but requiring careful attention to type behavior.

This project introduces the fundamental building blocks of Python programming: how to store data in variables, perform calculations, and interact with the program through input and output.

## Key Concepts

### 1. Printing Output

Python's `print()` function is the primary way to display information:

```python
print("Hello, World!")
print(42)
print("The answer is", 42)
```

**How Python is Unique:**
- Python 3's `print()` is a function (not a statement like Python 2)
- Automatically converts values to strings
- Supports multiple arguments with automatic spacing
- Can redirect to files easily: `print("text", file=f)`

### 2. Reading User Input

The `input()` function reads text from the user:

```python
name = input("Enter your name: ")
```

**Python's Approach:**
- Always returns a string (must convert for numbers)
- Blocks execution until user presses Enter
- Can provide a prompt string as argument

### 3. Primitive Data Types

Python has several built-in types:

```
┌─────────────────────────────────────────┐
│         Python Data Types               │
├─────────────────────────────────────────┤
│  int      → 42, -10, 0                  │
│  float    → 3.14, -0.5, 2.0             │
│  str      → "hello", 'world', """doc""" │
│  bool     → True, False                 │
│  NoneType → None                        │
└─────────────────────────────────────────┘
```

**Type Checking:**
```python
type(42)        # <class 'int'>
isinstance(42, int)  # True
```

### 4. Variable Assignment

Variables are created by assignment (no declaration needed):

```python
x = 10
y = "hello"
z = 3.14
```

**Python's Dynamic Typing:**
- Variables can change type: `x = 10` then `x = "hello"` is valid
- No type declarations required
- Type is determined by the value assigned

### 5. Arithmetic Operations

Python supports standard mathematical operations:

```
┌──────────┬──────────────┬─────────────────┐
│ Operator │ Operation    │ Example         │
├──────────┼──────────────┼─────────────────┤
│ +        │ Addition     │ 3 + 2 = 5       │
│ -        │ Subtraction  │ 5 - 2 = 3       │
│ *        │ Multiplication│ 3 * 4 = 12     │
│ /        │ Division     │ 10 / 3 = 3.333  │
│ //       │ Floor div    │ 10 // 3 = 3     │
│ %        │ Modulo       │ 10 % 3 = 1      │
│ **       │ Exponentiation│ 2 ** 3 = 8     │
└──────────┴──────────────┴─────────────────┘
```

**Python's Division Behavior:**
- `/` always returns float (even for integers)
- `//` performs floor division (returns int for int operands)
- `%` returns remainder (useful for cycles and patterns)

## Solution Approach

### Step 1: Variable Declaration and Types

Create variables of different types and understand their properties:

```python
# Integer
age = 25

# Float
temperature = 98.6

# String
name = "Alice"

# Boolean
is_student = True
```

### Step 2: Basic Arithmetic

Perform calculations using arithmetic operators:

```python
# Addition
sum_result = 10 + 5

# Multiplication
product = 3 * 4

# Division (returns float)
quotient = 15 / 3

# Exponentiation
power = 2 ** 8
```

### Step 3: Type Conversion

Convert between types when needed:

```python
# String to int
num_str = "42"
num_int = int(num_str)

# Int to string
age = 25
age_str = str(age)

# Float to int (truncates)
pi_int = int(3.14)  # 3
```

### Step 4: String Operations

Manipulate strings with operators and methods:

```python
# Concatenation
full_name = "John" + " " + "Doe"

# Repetition
separator = "-" * 20

# String formatting (multiple ways)
message = f"Hello, {name}!"
message = "Hello, {}!".format(name)
```

## How Python Uniquely Solves This

### 1. Dynamic Typing
Python doesn't require type declarations, making code more concise:

```python
# Python - simple and flexible
x = 10
x = "now a string"  # Valid!

# vs. Go - explicit types required
var x int = 10
x = "string"  // Compile error!

# vs. TypeScript - types optional but encouraged
let x: number = 10;
x = "string";  // Type error if strict mode

# vs. Rust - strict ownership and types
let mut x: i32 = 10;
x = "string";  // Compile error!
```

### 2. Print Function Flexibility
Python's `print()` handles multiple types seamlessly:

```python
# Python - automatic conversion
print(42, "hello", 3.14)  # Works!

# Go - needs format strings
fmt.Println(42, "hello", 3.14)  // Similar but different syntax

# TypeScript - console.log similar
console.log(42, "hello", 3.14);  // Similar behavior

# Rust - needs macros for formatting
println!("{} {} {}", 42, "hello", 3.14);  // More verbose
```

### 3. Division Behavior
Python 3's division always returns floats, preventing integer division surprises:

```python
# Python 3 - intuitive
10 / 3  # 3.3333333333333335

# Go - integer division by default
10 / 3  // 3 (integer division)
10.0 / 3.0  // 3.333... (float division)

# TypeScript - similar to Python
10 / 3  // 3.3333333333333335

# Rust - explicit types matter
10 / 3  // 3 (integer division)
10.0 / 3.0  // 3.333... (float division)
```

## Comparison with Other Languages

### Go
- **Statically typed**: Must declare types explicitly
- **Compiled**: Faster execution, but requires compilation step
- **Multiple return values**: Functions can return multiple values natively
- **No classes**: Uses structs and interfaces instead

### TypeScript
- **Optional typing**: Can use types or not (JavaScript compatibility)
- **Compiled to JavaScript**: Runs in browsers/Node.js
- **Class-based OOP**: Similar to Python but with type annotations
- **Strict null checking**: Helps prevent null/undefined errors

### Rust
- **Ownership system**: Memory safety without garbage collection
- **Zero-cost abstractions**: High-level features with C-like performance
- **Pattern matching**: Powerful destructuring and matching
- **Compile-time guarantees**: Catches many errors before runtime

## Deliverables

Complete the functions in `test/exercise.py`:

1. **Variable declarations** - Create variables of different types
2. **Arithmetic operations** - Perform calculations
3. **Type conversions** - Convert between types
4. **String operations** - Concatenate and format strings
5. **Input/Output** - Read input and format output

## Testing

Run the tests with:

```bash
cd test
pytest test.py -v
```

All tests should pass when your implementation is correct.

## Next Steps

After completing this project, you'll understand:
- How Python handles variables and types
- Basic I/O operations
- Arithmetic and string operations

This foundation is essential for all subsequent projects, as every program uses variables and basic operations.

