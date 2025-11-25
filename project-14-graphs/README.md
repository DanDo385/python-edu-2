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

## Detailed Problem Analysis

### Graph Representations

**1. Adjacency List (Most Common):**
```python
{
  0: [1, 2],
  1: [0, 3],
  2: [0],
  3: [1]
}
```
- Space: O(V + E) - efficient for sparse graphs
- Check edge: O(degree) - need to search list
- Add edge: O(1) - append to list

**2. Adjacency Matrix:**
```
    0  1  2  3
0  [0, 1, 1, 0]
1  [1, 0, 0, 1]
2  [1, 0, 0, 0]
3  [0, 1, 0, 0]
```
- Space: O(V²) - efficient for dense graphs
- Check edge: O(1) - direct lookup
- Add edge: O(1) - set value

### DFS (Depth-First Search) - Deep Dive

**Algorithm (Recursive):**
```
DFS(graph, node, visited):
  1. Mark node as visited
  2. Process node
  3. For each neighbor not visited:
     - Recursively call DFS(neighbor)
```

**Algorithm (Iterative with Stack):**
```
DFS(graph, start):
  1. Create stack, push start
  2. Create visited set
  3. While stack not empty:
     - Pop node
     - If not visited:
       - Mark visited, process
       - Push all neighbors
```

**Time Complexity:** O(V + E) - visit each vertex and edge once
**Space Complexity:** O(V) - visited set + recursion stack/stack

**Use Cases:**
- Topological sorting
- Finding connected components
- Detecting cycles
- Maze solving
- Path finding (not shortest)

### BFS (Breadth-First Search) - Deep Dive

**Algorithm:**
```
BFS(graph, start):
  1. Create queue, enqueue start
  2. Create visited set, mark start visited
  3. While queue not empty:
     - Dequeue node
     - Process node
     - For each unvisited neighbor:
       - Mark visited, enqueue
```

**Time Complexity:** O(V + E) - visit each vertex and edge once
**Space Complexity:** O(V) - visited set + queue

**Use Cases:**
- Shortest path in unweighted graphs
- Level-order traversal
- Finding minimum spanning tree
- Social network analysis (degrees of separation)

### DFS vs BFS Comparison

| Aspect | DFS | BFS |
|--------|-----|-----|
| Data structure | Stack (recursion) | Queue |
| Order | Deep first | Level by level |
| Space | O(h) - height | O(w) - width |
| Shortest path | No guarantee | Yes (unweighted) |
| Best for | Deep trees, backtracking | Level-order, shortest path |

### Path Finding with BFS

**Finding shortest path:**
1. Use BFS to explore level by level
2. Track parent of each node
3. When target found, backtrack using parents
4. Reconstruct path from start to target

**Why BFS finds shortest path:**
- Explores nodes level by level
- First time reaching target = shortest path
- Guarantees minimum number of edges

### Edge Cases to Handle

1. **Disconnected graph**: Multiple components
2. **Self-loops**: Node connected to itself
3. **Multiple edges**: Same edge appears multiple times
4. **Empty graph**: No vertices or edges
5. **Single node**: Graph with one vertex
6. **Cyclic graph**: Contains cycles

### Common Mistakes

1. **Not marking visited before enqueue**: Can add duplicates to queue
2. **Forgetting to check visited**: Revisiting nodes
3. **Wrong data structure**: Using stack for BFS or queue for DFS
4. **Not handling disconnected**: Only traversing one component

## Deliverables

Complete graph implementation with comprehensive functionality:

1. **Graph class** - Adjacency list representation
2. **Add vertex/edge** - Build graph dynamically
3. **DFS (recursive)** - Depth-first traversal
4. **DFS (iterative)** - Using stack
5. **BFS (iterative)** - Using queue
6. **Path finding** - Shortest path using BFS
7. **Connected components** - Find all components using DFS
8. **Cycle detection** - Detect cycles in undirected graph
9. **Print graph** - Visualize adjacency list

**Additional Challenges:**
- Detect cycle in directed graph
- Topological sort using DFS
- Find all paths between two nodes
- Bipartite graph checking
- Strongly connected components (Kosaraju's algorithm)

