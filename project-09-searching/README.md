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

## Detailed Problem Analysis

### Linear Search - Step by Step

**Algorithm:**
1. Start from the first element
2. Compare each element with target
3. If found, return index
4. If end reached without finding, return -1

**Time Complexity Analysis:**
- Best case: O(1) - target is first element
- Average case: O(n/2) ≈ O(n) - target is in middle
- Worst case: O(n) - target is last or not present

**Space Complexity:** O(1) - only using a few variables

**When to Use:**
- Unsorted data
- Small datasets
- Need to find first occurrence
- Simple implementation needed

### Binary Search - Step by Step

**Prerequisites:** Array must be sorted

**Algorithm (Iterative):**
1. Set left = 0, right = len(arr) - 1
2. While left <= right:
   - Calculate mid = (left + right) // 2
   - If arr[mid] == target: return mid
   - If arr[mid] < target: left = mid + 1 (search right half)
   - If arr[mid] > target: right = mid - 1 (search left half)
3. Return -1 if not found

**Algorithm (Recursive):**
1. Base case: If left > right, return -1
2. Calculate mid = (left + right) // 2
3. If found at mid, return mid
4. Recursively search appropriate half

**Time Complexity Analysis:**
- Best case: O(1) - target is middle element
- Average case: O(log n) - eliminates half each step
- Worst case: O(log n) - must go to leaf

**Space Complexity:**
- Iterative: O(1) - constant space
- Recursive: O(log n) - call stack depth

**When to Use:**
- Sorted data
- Large datasets
- Need fast search
- Can afford sorting cost upfront

### Edge Cases to Handle

1. **Empty list**: Return -1 immediately
2. **Single element**: Check if it matches
3. **Target not present**: Return -1
4. **Duplicate elements**: Decide which index to return (first/last/any)
5. **Target at boundaries**: First or last element
6. **Very large arrays**: Ensure no integer overflow in mid calculation

### Common Mistakes

1. **Off-by-one errors**: Using `<=` vs `<` in binary search
2. **Not checking empty list**: Will cause index errors
3. **Forgetting to sort**: Binary search requires sorted data
4. **Integer overflow**: `(left + right) // 2` vs `left + (right - left) // 2`

## Deliverables

Complete search functions with thorough testing:

1. **Linear search** - Handle all edge cases
2. **Binary search (iterative)** - Implement with proper bounds checking
3. **Binary search (recursive)** - Understand recursion vs iteration trade-offs
4. **Search with early exit** - Optimize linear search when possible
5. **Find first/last occurrence** - Handle duplicates correctly
6. **Search in rotated sorted array** - Advanced binary search variant

**Additional Challenges:**
- Implement binary search to find insertion point
- Search in 2D sorted matrix
- Find peak element in array (binary search variant)

