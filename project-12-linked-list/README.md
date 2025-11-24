# Project 12: Linked List

## Learning Objectives

- Understand linked list structure and nodes
- Implement singly linked list operations
- Learn when to use linked lists vs arrays
- Master pointer/reference concepts in Python
- Implement list reversal

## Problem Description

Linked lists are dynamic data structures where elements are connected via pointers. Unlike arrays, they don't require contiguous memory and can grow/shrink efficiently.

## Key Concepts

### Node Structure

```
Node: [data | next] → Node: [data | next] → None
```

### Operations

- `append(value)` - Add to end
- `prepend(value)` - Add to beginning
- `find(value)` - Search for value
- `remove(value)` - Remove first occurrence
- `reverse()` - Reverse the list

## How Python Uniquely Solves This

Python's references work like pointers. No explicit memory management needed. Linked lists are less common in Python (lists are efficient), but understanding them is important for interviews and understanding data structures.

## Comparison with Other Languages

- **Go**: Similar, but explicit pointer types
- **TypeScript**: Similar reference-based approach
- **Rust**: Ownership makes linked lists more complex

## Deliverables

Complete SinglyLinkedList class:
1. Node class
2. Append/prepend methods
3. Find/remove methods
4. Reverse method

