# Solution in Words: Lists and Tuples

## How to Think About This Problem

### Understanding Lists

Think of a list as a container that holds multiple items in order. You can add, remove, and modify items.

**Mental Model:**
```
List = Ordered Container
[item1, item2, item3, ...]
  ↑      ↑      ↑
index0  index1 index2
```

### Understanding Tuples

Think of a tuple as a fixed record—once created, it cannot be changed. Use tuples for things that shouldn't change (like coordinates, dates).

**Mental Model:**
```
Tuple = Fixed Record
(x, y) = point that never changes
```

### Key Operations

1. **Accessing**: Use index to get value `list[0]`
2. **Slicing**: Get a portion `list[1:4]`
3. **Modifying**: Change values (lists only) `list[0] = new_value`
4. **Iterating**: Loop through items `for item in list:`

### Finding Min/Max

**Thinking:**
- Start with first element as initial min/max
- Compare with each remaining element
- Update if you find smaller/larger value

### Merging Sorted Lists

**Thinking:**
- Use two pointers (indices), one for each list
- Compare elements at current positions
- Add smaller one to result, advance that pointer
- Continue until one list is exhausted
- Add remaining elements from other list

