# Project 10: Sorting Algorithms

## Learning Objectives

- Implement insertion sort (O(n²))
- Implement merge sort (O(n log n))
- Understand time complexity differences
- Learn in-place vs out-of-place sorting
- Compare algorithm performance

## Problem Description

Sorting is one of the most fundamental operations in computer science. Different algorithms have different time complexities and use cases.

## Key Concepts

### Insertion Sort

```
Build sorted array one element at a time.
Time: O(n²) worst case, O(n) best case (already sorted)
Space: O(1) - in-place
```

### Merge Sort

```
Divide array in half, sort each half, merge results.
Time: O(n log n) - always
Space: O(n) - needs extra space for merging
```

## How Python Uniquely Solves This

Python's built-in `sorted()` uses Timsort (hybrid of merge sort and insertion sort), which is O(n log n) worst case but O(n) for nearly-sorted data.

## Comparison with Other Languages

- **Go**: Similar implementations, `sort` package uses quicksort
- **TypeScript**: Array.sort() uses implementation-dependent algorithm
- **Rust**: Iterator methods provide efficient sorting

## Deliverables

Complete sorting functions:
1. Insertion sort
2. Merge sort
3. Compare performance
4. Handle edge cases

