# Project 13: Trees (Binary Tree Basics)

## Learning Objectives

- Understand tree data structures and terminology
- Implement binary tree node structure
- Master tree traversal algorithms (preorder, inorder, postorder)
- Calculate tree height and size
- Understand Binary Search Tree (BST) properties

## Problem Description

Trees are hierarchical data structures. Binary trees have at most two children per node. Understanding trees is crucial for many algorithms and data structures.

## Key Concepts

### Tree Structure

```
        Root
       /    \
    Left   Right
    /  \    /  \
   L   R   L   R
```

### Traversals

- **Preorder**: Root → Left → Right
- **Inorder**: Left → Root → Right  
- **Postorder**: Left → Right → Root

## How Python Uniquely Solves This

Python's recursive nature makes tree algorithms elegant. No explicit memory management needed for tree structures.

## Comparison with Other Languages

- **Go**: Similar recursive approach
- **TypeScript**: Similar, with optional type annotations
- **Rust**: Ownership makes tree structures more complex

## Deliverables

Complete tree functions:
1. TreeNode class
2. Tree traversal functions
3. Tree height calculation
4. Tree size calculation

