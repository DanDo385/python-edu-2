# Project 08: Recursion and Divide-and-Conquer

## Learning Objectives

- Understand recursion: functions calling themselves
- Learn base cases and recursive cases
- Master classic recursive algorithms (factorial, Fibonacci, GCD)
- Understand the call stack and recursion depth
- Apply divide-and-conquer strategies

## Problem Description

Recursion is a powerful technique where a function solves a problem by calling itself on smaller subproblems. Divide-and-conquer breaks problems into smaller pieces, solves them recursively, and combines results.

## Key Concepts

### Recursion Structure

```
Base Case: Simple case that doesn't need recursion
Recursive Case: Break problem into smaller subproblems
```

### Call Stack Visualization

```
factorial(3)
  → factorial(2)
    → factorial(1)
      → factorial(0) [returns 1]
    [returns 1]
  [returns 2]
[returns 6]
```

## How Python Uniquely Solves This

Python's recursion limit (default 1000) protects against infinite recursion. Tail recursion optimization isn't automatic, but recursion is natural for tree/graph problems.

## Comparison with Other Languages

- **Go**: Similar recursion, but tail call optimization possible
- **TypeScript**: Similar, but can hit stack limits easily
- **Rust**: Recursion similar, but ownership can complicate things

## Deliverables

Complete recursive functions:
1. Factorial (recursive)
2. Fibonacci (recursive)
3. GCD using Euclidean algorithm
4. Binary search (divide-and-conquer)
5. List sum (recursive)

