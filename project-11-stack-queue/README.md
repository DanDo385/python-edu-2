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

## Detailed Problem Analysis

### Stack - Deep Dive

**LIFO Principle:** Last In, First Out - like a stack of plates

**Operations Complexity:**
- `push(item)`: O(1) - add to top
- `pop()`: O(1) - remove from top
- `peek()`: O(1) - view top element
- `isEmpty()`: O(1) - check if empty
- `size()`: O(1) - get number of elements

**Implementation Choices:**
- **List-based**: Use `append()` and `pop()` (O(1) amortized)
- **Linked list**: More complex but guaranteed O(1)

**Common Applications:**
1. **Function call stack**: Recursion, nested function calls
2. **Expression evaluation**: Infix to postfix conversion
3. **Backtracking**: Undo operations, DFS traversal
4. **Balanced parentheses**: Matching brackets, HTML tags
5. **Browser history**: Back button functionality

**Example: Balanced Parentheses**
```
"()" → Valid
"()[]{}" → Valid
"([)]" → Invalid (mismatched)
"((()))" → Valid
```

**Algorithm:**
1. Traverse string character by character
2. If opening bracket: push to stack
3. If closing bracket: pop from stack and check match
4. At end: stack should be empty

### Queue - Deep Dive

**FIFO Principle:** First In, First Out - like a line of people

**Operations Complexity:**
- `enqueue(item)`: O(1) - add to back
- `dequeue()`: O(1) - remove from front
- `peek()`: O(1) - view front element
- `isEmpty()`: O(1) - check if empty
- `size()`: O(1) - get number of elements

**Implementation Choices:**
- **List-based**: `append()` and `pop(0)` (pop(0) is O(n) - inefficient!)
- **collections.deque**: O(1) operations on both ends (recommended)
- **Linked list**: O(1) operations, more memory overhead

**Common Applications:**
1. **BFS traversal**: Level-order tree traversal, graph exploration
2. **Task scheduling**: Process scheduling, print queue
3. **Breadth-first search**: Finding shortest path in unweighted graphs
4. **Cache implementation**: LRU cache, request queuing
5. **Stream processing**: Processing data in order

**Example: BFS Traversal**
```
    1
   / \
  2   3
 / \
4   5

BFS: 1 → 2 → 3 → 4 → 5
```

**Algorithm:**
1. Start with root node, enqueue it
2. While queue not empty:
   - Dequeue node
   - Process node
   - Enqueue all children

### Edge Cases to Handle

**Stack:**
1. Pop from empty stack → raise exception or return None
2. Peek empty stack → raise exception or return None
3. Very large stack → memory considerations

**Queue:**
1. Dequeue from empty queue → raise exception or return None
2. Peek empty queue → raise exception or return None
3. Circular queue → handle wrap-around for fixed-size queues

### Common Mistakes

1. **Using list.pop(0)**: O(n) operation, use deque instead
2. **Not checking empty**: Will cause errors
3. **Wrong order**: Confusing LIFO vs FIFO
4. **Memory leaks**: Not clearing references

## Deliverables

Complete implementations with comprehensive functionality:

1. **Stack class** - All operations with error handling
2. **Queue class** - Using deque for efficiency
3. **Balanced parentheses checker** - Handle (), [], {}
4. **Expression evaluator** - Evaluate postfix expressions using stack
5. **BFS implementation** - Level-order traversal using queue
6. **Two-stack queue** - Implement queue using two stacks (advanced)
7. **Min stack** - Stack that tracks minimum element in O(1) time

**Additional Challenges:**
- Implement circular queue with fixed size
- Next greater element using stack
- Sliding window maximum using deque
- Design stack with getMin() in O(1)

