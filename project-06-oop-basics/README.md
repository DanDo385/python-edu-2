# Project 06: Python OOP Basics

## Learning Objectives

- Understand object-oriented programming concepts in Python
- Learn to define classes with attributes and methods
- Master the `__init__` constructor and `self` reference
- Create objects and call methods
- Understand encapsulation and instance vs class variables

## Problem Description

Object-Oriented Programming (OOP) organizes code into classes and objects. Python's OOP system is flexible and powerful, allowing you to model real-world entities and their behaviors.

## Key Concepts

### Class Definition

```python
class ClassName:
    def __init__(self, param1, param2):
        self.attribute1 = param1
        self.attribute2 = param2
    
    def method_name(self):
        return self.attribute1
```

### Key Components

```
┌─────────────────────────────────────┐
│         Python Class Structure      │
├─────────────────────────────────────┤
│  class BankAccount:                 │
│      def __init__(self, balance):   │ ← Constructor
│          self.balance = balance     │ ← Instance attribute
│                                      │
│      def deposit(self, amount):     │ ← Method
│          self.balance += amount     │
│                                      │
│      def withdraw(self, amount):    │
│          self.balance -= amount     │
└─────────────────────────────────────┘
```

### Self Reference

- `self` refers to the instance of the class
- Must be first parameter in instance methods
- Used to access instance attributes and methods

### Instance vs Class Variables

- **Instance variables**: Unique to each object (`self.attribute`)
- **Class variables**: Shared by all instances (`ClassName.variable`)

## How Python Uniquely Solves This

### 1. Explicit Self

Python requires explicit `self` parameter:

```python
# Python - explicit self
class MyClass:
    def method(self, arg):
        return self.attribute + arg

# vs. Go - no classes, uses structs and methods
type MyStruct struct {
    Attribute int
}
func (m MyStruct) Method(arg int) int {
    return m.Attribute + arg
}

# vs. TypeScript - implicit this
class MyClass {
    method(arg: number): number {
        return this.attribute + arg;
    }
}

# vs. Rust - no self keyword, uses &self
impl MyStruct {
    fn method(&self, arg: i32) -> i32 {
        self.attribute + arg
    }
}
```

### 2. Dynamic Attributes

Python allows adding attributes at runtime:

```python
obj = MyClass()
obj.new_attribute = "value"  # Valid in Python!
```

### 3. Duck Typing

Python doesn't require explicit interfaces—if it walks like a duck, it's a duck.

## Comparison with Other Languages

### Go
- **No classes**: Uses structs with methods
- **Interfaces**: Implicit interface satisfaction
- **No inheritance**: Uses composition instead

### TypeScript
- **Class syntax**: Similar to Python but with types
- **Access modifiers**: public, private, protected
- **Interfaces**: Explicit interface definitions

### Rust
- **Structs and impl**: Similar concept but different syntax
- **Ownership**: Important consideration for methods
- **Traits**: Similar to interfaces

## Deliverables

Complete the class definitions in `exercise.py`:
1. BankAccount class with deposit/withdraw methods
2. Point class with distance calculation
3. Rectangle class with area/perimeter methods
4. Student class with grade management

## Testing

Run: `pytest test.py -v`

