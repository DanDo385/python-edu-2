# Project 05: Data Structures II – Dictionaries and Sets

## Learning Objectives

- Master Python's mapping type (dictionaries) for key-value storage
- Understand set operations for unique element handling
- Learn efficient lookups using dictionaries
- Use set operations (union, intersection, difference)
- Apply dictionaries and sets to solve real problems

## Key Concepts

### Dictionaries

```python
# Creation
my_dict = {"key": "value", "name": "Alice"}

# Access
value = my_dict["key"]  # Direct access
value = my_dict.get("key", default)  # Safe access

# Operations
my_dict["new_key"] = "new_value"  # Add/update
del my_dict["key"]  # Delete
"key" in my_dict  # Membership test
```

### Sets

```python
# Creation
my_set = {1, 2, 3}
my_set = set([1, 2, 3])

# Operations
my_set.add(4)  # Add element
my_set.remove(3)  # Remove element
set1 | set2  # Union
set1 & set2  # Intersection
set1 - set2  # Difference
```

## How Python Uniquely Solves This

Python dictionaries are hash tables with O(1) average lookup time. Sets use the same underlying structure for O(1) membership testing.

## Comparison with Other Languages

- **Go**: Maps are similar but require explicit type declarations
- **TypeScript**: Objects/Map similar, but dictionaries are more Pythonic
- **Rust**: HashMap/BTreeMap similar, but Python's syntax is cleaner

## Deliverables

Complete functions for:
1. Word frequency counting
2. Set operations
3. Dictionary manipulation
4. Finding unique elements

