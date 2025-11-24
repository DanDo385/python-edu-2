# Solution in Words: Dictionaries and Sets

## How to Think About This Problem

### Understanding Dictionaries

Think of a dictionary as a phone book: you look up a name (key) to find a number (value).

**Mental Model:**
```
Dictionary = Lookup Table
{"name": "Alice", "age": 25}
  key    value    key   value
```

### Understanding Sets

Think of a set as a collection of unique items—no duplicates allowed.

**Mental Model:**
```
Set = Unique Collection
{1, 2, 3} - each element appears once
```

### Word Frequency

**Thinking:**
- Split text into words
- For each word, increment its count in dictionary
- Use word as key, count as value

### Set Operations

**Thinking:**
- Union: All elements from both sets
- Intersection: Elements in both sets
- Difference: Elements in first but not second

