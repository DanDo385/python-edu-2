# Solution in Words: Stack and Queue

## How to Think About This Problem

### Understanding Stack (LIFO)

Think of a stack like a stack of plates - you add to the top and remove from the top.

**Mental Model:**
```
Push → [top]
        ↓
      [item]
        ↓
      [item]
        ↓
      [bottom]
```

### Understanding Queue (FIFO)

Think of a queue like a line - people join at the back and leave from the front.

**Mental Model:**
```
Front ← [item] [item] [item] ← Back
         ↑                    ↑
      dequeue              enqueue
```

### Stack Thinking

1. **Push**: Add to end of list (top)
2. **Pop**: Remove from end of list (top)
3. **Peek**: Look at last element without removing

### Queue Thinking

1. **Enqueue**: Add to end of list (back)
2. **Dequeue**: Remove from beginning of list (front)
3. **Peek**: Look at first element without removing

### Balanced Parentheses

**Thinking:**
- Use stack to track opening brackets
- When see closing bracket, check if matches top of stack
- If stack empty at end, balanced

### Problem-Solving Strategy

1. **Identify pattern**: LIFO → Stack, FIFO → Queue
2. **Choose structure**: Based on access pattern needed
3. **Implement operations**: Follow the ADT interface

