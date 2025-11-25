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

## Detailed Problem Analysis

### Tree Terminology

- **Root**: Topmost node (no parent)
- **Leaf**: Node with no children
- **Internal node**: Node with at least one child
- **Depth**: Distance from root (root depth = 0)
- **Height**: Longest path from node to leaf
- **Level**: Nodes at same depth
- **Subtree**: Tree formed by a node and its descendants

### Binary Tree Properties

- Each node has at most 2 children
- Left child and right child are distinct
- Order matters (left ≠ right)

### Tree Traversals - Deep Dive

**Preorder (Root → Left → Right):**
```
    1
   / \
  2   3
 / \
4   5

Preorder: 1 → 2 → 4 → 5 → 3
Use case: Copy tree, prefix expression evaluation
```

**Inorder (Left → Root → Right):**
```
    1
   / \
  2   3
 / \
4   5

Inorder: 4 → 2 → 5 → 1 → 3
Use case: BST gives sorted order, infix expression
```

**Postorder (Left → Right → Root):**
```
    1
   / \
  2   3
 / \
4   5

Postorder: 4 → 5 → 2 → 3 → 1
Use case: Delete tree, postfix expression evaluation
```

**Level-order (BFS):**
```
    1
   / \
  2   3
 / \
4   5

Level-order: 1 → 2 → 3 → 4 → 5
Use case: Print tree level by level, BFS problems
```

### Binary Search Tree (BST) Properties

**Invariant:** For any node:
- All nodes in left subtree < node value
- All nodes in right subtree > node value
- Both subtrees are also BSTs

**Operations:**
- **Search**: O(log n) average, O(n) worst (unbalanced)
- **Insert**: O(log n) average, O(n) worst
- **Delete**: O(log n) average, O(n) worst

### Tree Height Calculation

**Recursive Approach:**
```
height(node):
  if node is None: return -1 (or 0, depending on definition)
  return 1 + max(height(node.left), height(node.right))
```

**Time Complexity:** O(n) - visit every node
**Space Complexity:** O(h) - recursion stack depth (h = height)

### Tree Size Calculation

**Recursive Approach:**
```
size(node):
  if node is None: return 0
  return 1 + size(node.left) + size(node.right)
```

**Time Complexity:** O(n) - visit every node
**Space Complexity:** O(h) - recursion stack depth

### Edge Cases to Handle

1. **Empty tree**: Root is None
2. **Single node**: Only root exists
3. **Skewed tree**: All nodes on one side (worst case BST)
4. **Perfect tree**: All levels completely filled
5. **Unbalanced tree**: Height much larger than log(n)

### Common Mistakes

1. **Off-by-one in height**: Returning 0 vs -1 for None
2. **Not handling None**: Accessing attributes of None
3. **Incorrect traversal order**: Mixing up left/right/root order
4. **Stack overflow**: Deep recursion on unbalanced trees

## Deliverables

Complete tree implementation with comprehensive functionality:

1. **TreeNode class** - Basic node structure with left/right children
2. **Preorder traversal** - Recursive and iterative versions
3. **Inorder traversal** - Recursive and iterative versions
4. **Postorder traversal** - Recursive and iterative versions
5. **Level-order traversal** - Using queue (BFS)
6. **Tree height** - Calculate maximum depth
7. **Tree size** - Count total nodes
8. **Tree maximum** - Find maximum value in tree
9. **Tree minimum** - Find minimum value in tree
10. **Search in BST** - Efficient search in binary search tree

**Additional Challenges:**
- Check if tree is balanced (height difference ≤ 1)
- Check if tree is valid BST
- Find lowest common ancestor (LCA)
- Serialize and deserialize binary tree
- Invert/mirror binary tree

