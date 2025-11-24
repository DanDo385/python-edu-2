# Project 07: Python OOP Advanced (Inheritance & Exceptions)

## Learning Objectives

- Understand class inheritance and method overriding
- Learn polymorphism and the `super()` function
- Master exception handling with try/except blocks
- Create custom exception classes
- Understand when to use inheritance vs composition

## Problem Description

Inheritance allows classes to reuse code from parent classes. Exceptions provide a clean way to handle errors. Together, they enable robust, maintainable object-oriented code.

## Key Concepts

### Inheritance

```python
class Parent:
    def method(self):
        return "parent"

class Child(Parent):
    def method(self):
        return "child"  # Override parent method
```

### Exception Handling

```python
try:
    # Risky code
    result = 10 / 0
except ZeroDivisionError:
    # Handle error
    result = None
```

## How Python Uniquely Solves This

Python's multiple inheritance and MRO (Method Resolution Order) provide flexible inheritance. Exception handling is built into the language with try/except/finally.

## Comparison with Other Languages

- **Go**: No inheritance, uses composition. Error handling via return values
- **TypeScript**: Single inheritance with interfaces. Try/catch similar
- **Rust**: Traits instead of inheritance. Result<T, E> for errors

## Deliverables

Complete class hierarchies and exception handling:
1. Shape hierarchy (Rectangle, Circle inherit from Shape)
2. Custom exceptions (OverdrawError, InvalidInputError)
3. Exception handling in methods

