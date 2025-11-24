# Project 14: Graphs and Graph Traversal Algorithms

## Learning Objectives

- Understand graph data structures (adjacency list representation)
- Implement Depth-First Search (DFS)
- Implement Breadth-First Search (BFS)
- Understand when to use DFS vs BFS
- Solve problems using graph traversal

## Problem Description

Graphs represent relationships between entities. Graph traversal algorithms (DFS and BFS) are fundamental for exploring graphs and solving many problems.

## Key Concepts

### Graph Representation

```
Adjacency List:
{
  0: [1, 2],
  1: [0, 3],
  2: [0],
  3: [1]
}
```

### DFS vs BFS

- **DFS**: Uses stack (recursion), explores deep before wide
- **BFS**: Uses queue, explores wide before deep

## How Python Uniquely Solves This

Python dictionaries make adjacency lists natural. Recursion makes DFS elegant. Collections.deque provides efficient queue for BFS.

## Comparison with Other Languages

- **Go**: Similar implementations with slices and maps
- **TypeScript**: Similar with objects/arrays
- **Rust**: Ownership makes graph structures more complex

## Deliverables

Complete graph functions:
1. Graph class with adjacency list
2. DFS traversal (recursive and iterative)
3. BFS traversal
4. Path finding using BFS

