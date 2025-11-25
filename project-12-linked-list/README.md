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

## Detailed Problem Analysis

### Linked List vs Array

| Aspect | Array | Linked List |
|--------|-------|-------------|
| Memory | Contiguous | Non-contiguous |
| Access | O(1) random access | O(n) sequential access |
| Insertion (beginning) | O(n) - shift elements | O(1) - update head |
| Insertion (end) | O(1) amortized | O(n) - traverse to end |
| Deletion | O(n) - shift elements | O(n) - find then O(1) |
| Memory overhead | Minimal | Extra pointer per node |

**When to Use Linked Lists:**
- Frequent insertions/deletions at beginning
- Unknown size at creation
- Don't need random access
- Memory fragmentation concerns

### Node Structure Deep Dive

```python
class Node:
    def __init__(self, data):
        self.data = data  # Store the value
        self.next = None  # Reference to next node
```

**Memory Layout:**
```
Node 1: [data: 10 | next: →] → Node 2: [data: 20 | next: →] → None
```

### Operations Complexity Analysis

**Append (add to end):**
- Time: O(n) - must traverse to end
- Space: O(1) - only new node

**Prepend (add to beginning):**
- Time: O(1) - just update head
- Space: O(1) - only new node

**Find:**
- Time: O(n) - worst case traverse entire list
- Space: O(1) - no extra space

**Remove:**
- Time: O(n) - find node + O(1) removal
- Space: O(1) - no extra space

**Reverse:**
- Time: O(n) - must visit every node
- Space: O(1) iterative, O(n) recursive

### Reversing a Linked List

**Iterative Approach:**
```
1. Initialize: prev = None, current = head
2. While current is not None:
   - Store next: next_node = current.next
   - Reverse link: current.next = prev
   - Move forward: prev = current, current = next_node
3. Update head: head = prev
```

**Visual Example:**
```
Original: 1 → A2 A3 A4 None
Step 1:    None ← A1  A2 → A3 → A4 → None
Step 2:    None ← A1 ← A2  A3 → A4 → None
Step 3:    None ← A1 ← A2 ← A3  A4 → None
Step 4:    None ← A1 ← A2 ← A3 ← A4
Result:    A4 → A3 → A2 → A1 → None
```

### Edge Cases to Handle

1. **Empty list**: Head is None
2. **Single node**: Only one element
3. **Removing head**: Special case handling
4. **Removing non-existent value**: Return False or raise exception
5. **Circular reference**: Detect cycles

### Common Mistakes

1. **Losing references**: Not storing next before modifying
2. **Not updating head**: Forgetting to update head in prepend
3. **Off-by-one errors**: Incorrect loop termination
4. **Memory leaks**: Not properly handling removed nodes (in languages with manual memory management)

## Deliverables

Complete SinglyLinkedList implementation with comprehensive functionality:

1. **Node class** - Basic node structure
2. **SinglyLinkedList class** - Complete implementation
3. **Append method** - Add to end with O(n) time
4. **Prepend method** - Add to beginning with O(1) time
5. **Find method** - Search for value, return node or None
6. **Remove method** - Remove first occurrence of value
7. **Reverse method** - Both iterative and recursive versions
8. **Size method** - Count nodes efficiently
9. **To list method** - Convert to Python list for testing

**Additional Challenges:**
- Detect cycle in linked list (Floyd's algorithm)
- Find middle node (two-pointer technique)
- Merge two sorted linked lists
- Remove nth node from end
- Implement doubly linked list

