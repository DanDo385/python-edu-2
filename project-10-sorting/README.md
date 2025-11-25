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

## Detailed Problem Analysis

### Insertion Sort - Deep Dive

**Algorithm:**
1. Start with second element (index 1)
2. Compare with elements before it
3. Shift larger elements right
4. Insert element in correct position
5. Repeat for all elements

**Step-by-Step Example:**
```
Array: [5, 2, 4, 6, 1, 3]
Step 1: [2, 5, 4, 6, 1, 3]  (insert 2)
Step 2: [2, 4, 5, 6, 1, 3]  (insert 4)
Step 3: [2, 4, 5, 6, 1, 3]  (6 already in place)
Step 4: [1, 2, 4, 5, 6, 3]  (insert 1)
Step 5: [1, 2, 3, 4, 5, 6]  (insert 3)
```

**Time Complexity:**
- Best case: O(n) - already sorted, one comparison per element
- Average case: O(n²) - roughly n²/4 comparisons
- Worst case: O(n²) - reverse sorted, maximum comparisons

**Space Complexity:** O(1) - in-place sorting

**Stability:** Stable - equal elements maintain relative order

**When to Use:**
- Small datasets (< 50 elements)
- Nearly sorted data
- Simple implementation needed
- Stable sort required

### Merge Sort - Deep Dive

**Algorithm (Divide and Conquer):**
1. **Divide**: Split array into two halves
2. **Conquer**: Recursively sort both halves
3. **Combine**: Merge sorted halves

**Merge Process:**
- Use two pointers, one for each half
- Compare elements, add smaller to result
- Continue until one half exhausted
- Append remaining elements

**Step-by-Step Example:**
```
Array: [5, 2, 4, 6, 1, 3]
Divide: [5, 2, 4] | [6, 1, 3]
Divide: [5, 2] [4] | [6, 1] [3]
Divide: [5] [2] [4] | [6] [1] [3]
Merge: [2, 5] [4] | [1, 6] [3]
Merge: [2, 4, 5] | [1, 3, 6]
Merge: [1, 2, 3, 4, 5, 6]
```

**Time Complexity:**
- Best case: O(n log n) - always divides fully
- Average case: O(n log n) - consistent performance
- Worst case: O(n log n) - guaranteed performance

**Space Complexity:** O(n) - needs auxiliary array for merging

**Stability:** Stable - merge preserves relative order

**When to Use:**
- Large datasets
- Need guaranteed O(n log n) performance
- External sorting (can handle data on disk)
- Stable sort required

### Comparison: Insertion Sort vs Merge Sort

| Aspect | Insertion Sort | Merge Sort |
|--------|---------------|------------|
| Time (best) | O(n) | O(n log n) |
| Time (worst) | O(n²) | O(n log n) |
| Space | O(1) | O(n) |
| Stability | Stable | Stable |
| Best for | Small/nearly sorted | Large datasets |

### Edge Cases to Handle

1. **Empty array**: Return empty array
2. **Single element**: Already sorted, return as-is
3. **Two elements**: Simple swap if needed
4. **Duplicate elements**: Maintain stability
5. **Already sorted**: Insertion sort excels, merge sort still O(n log n)
6. **Reverse sorted**: Worst case for insertion sort

### Common Mistakes

1. **Off-by-one errors**: Incorrect array bounds in merge
2. **Not copying elements**: Modifying original array incorrectly
3. **Infinite recursion**: Not handling base case properly
4. **Memory issues**: Not managing auxiliary arrays efficiently

## Deliverables

Complete sorting implementations with comprehensive testing:

1. **Insertion sort** - In-place implementation with detailed comments
2. **Merge sort** - Both recursive and iterative versions
3. **Performance comparison** - Time both algorithms on various inputs
4. **Edge case handling** - Empty, single element, duplicates, sorted arrays
5. **Stability verification** - Test that equal elements maintain order
6. **Visualization helper** - Print array state at each step (optional)

**Additional Challenges:**
- Implement hybrid sort (use insertion for small subarrays)
- Count inversions using merge sort
- Sort linked list using merge sort
- External merge sort for large files

