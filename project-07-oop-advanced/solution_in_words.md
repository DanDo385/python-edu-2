# Solution in Words: Python OOP Advanced

## How to Think About This Problem

### Understanding Inheritance

Think of inheritance as a family tree. Children inherit traits from parents but can also have their own unique traits.

**Mental Model:**
```
Shape (parent)
  ├── Rectangle (child)
  └── Circle (child)

Both Rectangle and Circle ARE Shapes
They inherit common behavior but can override it
```

### Understanding Exceptions

Think of exceptions as unexpected events that need special handling. Like catching a ball that's thrown unexpectedly.

**Mental Model:**
```
Normal flow → Exception occurs → Catch it → Handle it → Continue
```

### Inheritance Thinking

1. **Identify commonality**: What do child classes share?
2. **Define parent**: Create base class with shared code
3. **Override methods**: Child classes customize behavior
4. **Use super()**: Call parent methods when needed

### Exception Thinking

1. **Identify risks**: What could go wrong?
2. **Wrap in try**: Put risky code in try block
3. **Catch exceptions**: Handle specific error types
4. **Provide fallback**: What to do if error occurs

