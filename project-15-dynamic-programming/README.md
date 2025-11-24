# Project 15: Dynamic Programming Fundamentals

## Learning Objectives

- Understand overlapping subproblems and optimal substructure
- Implement memoization (top-down DP)
- Implement tabulation (bottom-up DP)
- Solve classic DP problems (Fibonacci, coin change)
- Recognize when to use dynamic programming

## Problem Description

Dynamic Programming optimizes recursive solutions by storing results of subproblems. It's essential for solving optimization problems efficiently.

## Key Concepts

### DP Characteristics

- **Overlapping subproblems**: Same subproblems computed multiple times
- **Optimal substructure**: Optimal solution contains optimal solutions to subproblems

### Approaches

- **Memoization**: Top-down, cache results as you compute
- **Tabulation**: Bottom-up, build table from base cases

## How Python Uniquely Solves This

Python dictionaries make memoization natural. List comprehensions help with tabulation. Decorators can automate memoization.

## Comparison with Other Languages

- **Go**: Similar approach with maps for memoization
- **TypeScript**: Similar, with optional type annotations
- **Rust**: Ownership makes DP implementations more complex

## Deliverables

Complete DP solutions:
1. Fibonacci with memoization
2. Fibonacci with tabulation
3. Coin change problem
4. Compare performance with naive recursion

