# Project 33: Building Neural Network Modules in PyTorch

## Learning Objectives

- Learn how to create custom neural network models using `nn.Module`
- Understand the modular structure of PyTorch models
- Use built-in layers from `torch.nn` (Linear, ReLU, etc.)
- Define `forward` method for model computation
- Understand model parameters and how they're tracked
- Compare PyTorch modules to manual weight management

## Problem Description

PyTorch's `nn.Module` is the foundation for all neural network models. It provides a clean, modular way to build networks compared to manually managing weights (like in Project 25).

**Why nn.Module?**
- Automatic parameter tracking
- Clean model definition
- Easy to use with optimizers
- Standard PyTorch pattern

**What You'll Build:**
- Custom `nn.Module` subclass
- Multi-layer feedforward network
- Using `nn.Linear` and `nn.ReLU`
- Proper `forward` method

## Key Concepts

### nn.Module Basics

```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = self.layer2(x)
        return x
```

### Built-in Layers

**nn.Linear:**
- Fully connected layer
- Automatically handles weights and biases
- Parameters: `input_size`, `output_size`

**nn.ReLU:**
- ReLU activation function
- Can use `nn.ReLU()` or `F.relu()`

**Other layers:**
- `nn.Conv2d`: Convolutional layers
- `nn.Dropout`: Dropout regularization
- `nn.BatchNorm1d`: Batch normalization

### Forward Method

- **Required**: Must define `forward` method
- **Called via**: `model(x)` (calls `forward` automatically)
- **Input/Output**: Define input/output shapes

### Model Parameters

```python
# Access all parameters
for param in model.parameters():
    print(param.shape)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
```

## Solution Approach

### Building a Module

1. **Subclass nn.Module**: `class MyModel(nn.Module)`
2. **Call super().__init__()**: Initialize parent class
3. **Define layers**: In `__init__`
4. **Define forward**: How data flows through model
5. **Instantiate**: `model = MyModel()`

### Layer Usage

- **Linear layers**: `nn.Linear(in_features, out_features)`
- **Activations**: `nn.ReLU()` or `F.relu()`
- **Sequential**: `nn.Sequential()` for simple stacks

## How Python Uniquely Solves This

### 1. Clean Class Structure

```python
# Clean module definition
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.layer(x)
```

### 2. Automatic Parameter Tracking

```python
# Parameters automatically tracked
model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

## Deliverables

Complete implementation:
1. Custom `nn.Module` subclass
2. Multi-layer feedforward network
3. Proper `forward` method
4. Parameter counting
5. Model instantiation and usage

## Testing

Run: `pytest test.py -v`

Tests verify:
- Model architecture is correct
- Forward pass produces correct output shape
- Parameters are tracked correctly
- Model can be instantiated and used

## Next Steps

After completing this project, you'll understand:
- How to build PyTorch models using `nn.Module`
- How to use built-in layers
- How models are structured

This is essential - all PyTorch models use `nn.Module`!
