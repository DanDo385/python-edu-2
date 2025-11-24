# Solution in Words: Activation Functions

## How to Think About This Problem

### Understanding Why Activations Are Needed

Think of activation functions like adding curves to a straight line. Multiple straight lines still make a straight line, but curves allow complex shapes.

**Mental Model:**
```
Without activation: Line → Line → Line = Still a line
With activation: Line → Curve → Line → Curve = Complex shape!
```

### Understanding ReLU

**What we're doing:** Keeping positive values, zeroing negatives.

**How to think about it:**
- If input > 0: pass through unchanged
- If input < 0: output 0 (neuron "off")
- Simple but effective
- No saturation for positive inputs (good gradients!)

### Understanding Softmax

**What we're doing:** Converting raw scores to probabilities.

**How to think about it:**
- Input: [2, 1, 0.1] (logits)
- Softmax: [0.7, 0.2, 0.1] (probabilities, sum to 1)
- Largest input gets highest probability
- Used for multi-class classification

### Problem-Solving Strategy

1. **Understand purpose**: What does this activation do?
2. **Implement function**: Code the mathematical formula
3. **Consider properties**: Range, differentiability, gradients
4. **Think about use case**: When would you use this?


