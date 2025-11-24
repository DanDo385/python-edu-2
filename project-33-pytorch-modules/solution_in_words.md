# Solution Explanation: PyTorch Modules

## Overview

PyTorch's `nn.Module` is the foundation for all neural network models. It provides a clean, modular way to build networks compared to manually managing weights.

## Key Concepts Explained

### nn.Module Basics

**Why nn.Module?**
- Automatic parameter tracking
- Clean model definition
- Easy to use with optimizers
- Standard PyTorch pattern

**Basic Structure:**
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
    
    def forward(self, x):
        # Define forward pass
        return output
```

### Built-in Layers

**nn.Linear:**
- Fully connected layer
- Parameters: `input_size`, `output_size`
- Automatically handles weights and biases

**nn.ReLU / F.relu:**
- ReLU activation
- Can use `nn.ReLU()` or `F.relu()`

**nn.Dropout:**
- Dropout regularization
- Behaves differently in train vs eval mode

**nn.Sequential:**
- Container for stacking layers
- Simplifies model definition

### Forward Method

**Required:**
- Must define `forward` method
- Called automatically via `model(x)`
- Defines how data flows through model

**Example:**
```python
def forward(self, x):
    x = self.layer1(x)
    x = F.relu(x)
    x = self.layer2(x)
    return x
```

### Model Parameters

**Accessing Parameters:**
```python
# Iterate over parameters
for param in model.parameters():
    print(param.shape)

# Count parameters
total = sum(p.numel() for p in model.parameters())
```

**Key Points:**
- Parameters automatically tracked
- All parameters have `requires_grad=True` by default
- Optimizers use `model.parameters()`

## Implementation Details

### Simple Model

```python
class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x
```

**Key Points:**
- Call `super().__init__()` first
- Define layers in `__init__`
- Implement `forward` method
- Use `F.relu()` for activations

### Sequential Model

```python
class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.net(x)
```

**Key Points:**
- `nn.Sequential` stacks layers
- Simpler for linear stacks
- Still need `nn.Module` wrapper

### Model Usage

```python
# Instantiate
model = SimpleNN()

# Forward pass
output = model(x)  # Calls forward() automatically

# Parameters
for param in model.parameters():
    print(param.shape)

# Train/eval modes
model.train()   # Enable dropout, batch norm training
model.eval()    # Disable dropout, batch norm eval
```

## Common Pitfalls

1. **Forgetting super().__init__()**: Breaks inheritance
2. **Wrong forward signature**: Must match expected input/output
3. **Not calling super()**: Model won't work correctly
4. **Mixing train/eval**: Dropout/batch norm behave differently

## Real-World Application

**In Practice:**
- All PyTorch models use `nn.Module`
- Define architecture in `__init__`
- Implement computation in `forward`
- Use with optimizers: `optimizer = torch.optim.Adam(model.parameters())`

**Next Steps:**
- Project 34: Training models
- Project 35: CNNs
- All use `nn.Module` pattern!

## Why This Matters

`nn.Module` is **essential** for PyTorch:
- Standard way to build models
- Automatic parameter tracking
- Works seamlessly with optimizers
- Industry standard pattern

**Comparison to Project 25:**
- Project 25: Manual weight management (educational)
- Project 33: `nn.Module` (practical)
- Same concepts, but PyTorch handles details!

This project establishes the model-building pattern used in all PyTorch projects!
