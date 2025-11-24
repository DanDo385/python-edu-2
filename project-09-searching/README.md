# Project 09: Searching Algorithms

## Learning Objectives

- Implement linear search (O(n) time)
- Implement binary search (O(log n) time)
- Understand when to use each algorithm
- Learn time complexity analysis basics
- Handle edge cases in search algorithms

## Problem Description

Searching is fundamental to computer science. Different algorithms have different time complexities. Understanding when to use each is crucial for writing efficient code.

## Key Concepts

### Linear Search

```
Check each element sequentially until found or end reached.
Time: O(n) - must check every element in worst case
Space: O(1) - no extra space needed
```

### Binary Search

```
Requires sorted array. Compare with middle, eliminate half.
Time: O(log n) - eliminates half each step
Space: O(1) iterative, O(log n) recursive
```

## How Python Uniquely Solves This

Python's `in` operator uses linear search for lists but hash tables for sets/dicts (O(1) average). Built-in `bisect` module provides binary search utilities.

## Comparison with Other Languages

- **Go**: Similar implementations, but explicit type requirements
- **TypeScript**: Array methods like `find()` use linear search
- **Rust**: Iterator methods provide efficient searching

## Deliverables

Complete search functions:
1. Linear search
2. Binary search (iterative and recursive)
3. Search with early exit
4. Handle edge cases (empty list, not found)

