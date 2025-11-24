# Project 11: Stack and Queue

## Learning Objectives

- Understand Stack (LIFO) and Queue (FIFO) data structures
- Implement Stack and Queue classes
- Solve problems using stacks (balanced parentheses)
- Solve problems using queues (BFS traversal)
- Understand when to use each structure

## Problem Description

Stacks and queues are fundamental abstract data types. Stacks are LIFO (Last In, First Out) like a stack of plates. Queues are FIFO (First In, First Out) like a line of people.

## Key Concepts

### Stack Operations

```
push(item)  - Add to top
pop()       - Remove from top
peek()      - View top without removing
isEmpty()   - Check if empty
```

### Queue Operations

```
enqueue(item) - Add to back
dequeue()     - Remove from front
peek()        - View front without removing
isEmpty()     - Check if empty
```

## How Python Uniquely Solves This

Python lists can be used as stacks (append/pop). For queues, `collections.deque` provides efficient O(1) operations on both ends.

## Comparison with Other Languages

- **Go**: Similar implementations using slices
- **TypeScript**: Arrays work similarly
- **Rust**: VecDeque provides queue functionality

## Deliverables

Complete Stack and Queue classes:
1. Stack with push/pop/peek
2. Queue with enqueue/dequeue/peek
3. Balanced parentheses checker (using stack)
4. Basic BFS (using queue)

