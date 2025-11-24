# Project 35: Convolutional Neural Networks for Image Classification (CIFAR-10)

## Learning Objectives

- Understand convolutional neural networks (CNNs) and how they work
- Learn to use convolutional layers (`nn.Conv2d`)
- Understand pooling layers (`nn.MaxPool2d`) for downsampling
- Build a CNN for CIFAR-10 image classification
- Understand how CNNs capture spatial patterns in images
- Train a CNN and achieve reasonable accuracy

## Problem Description

Convolutional Neural Networks (CNNs) are designed for image data. Unlike fully connected layers that treat images as flat vectors, CNNs preserve spatial structure and learn local patterns.

**CIFAR-10 Dataset:**
- 32x32 color images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images, 10,000 test images
- More challenging than MNIST (color, more classes, smaller images)

**Why CNNs?**
- Preserve spatial structure
- Learn local patterns (edges, textures)
- Parameter efficient (shared weights)
- Translation invariant

## Key Concepts

### Convolutional Layers

**nn.Conv2d:**
```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
```

**Parameters:**
- `in_channels`: Input channels (1 for grayscale, 3 for RGB)
- `out_channels`: Number of filters/feature maps
- `kernel_size`: Size of filter (e.g., 3x3)
- `stride`: Step size (default 1)
- `padding`: Zero padding (default 0)

### Pooling Layers

**nn.MaxPool2d:**
```python
nn.MaxPool2d(kernel_size, stride)
```

**Purpose:**
- Downsample feature maps
- Reduce spatial dimensions
- Increase receptive field

### CNN Architecture

```
Input (3, 32, 32)
  ↓
Conv2d(3, 32, 3) → ReLU → MaxPool2d(2)
  ↓
Conv2d(32, 64, 3) → ReLU → MaxPool2d(2)
  ↓
Flatten
  ↓
Linear(64*8*8, 128) → ReLU
  ↓
Linear(128, 10) → Output
```

## Solution Approach

### Building a CNN

1. **Convolutional layers**: Extract features
2. **Pooling layers**: Downsample
3. **Fully connected layers**: Classification
4. **Flatten**: Convert to 1D for FC layers

### Training

- Same training loop as Project 34
- Use CIFAR-10 dataset
- Achieve reasonable accuracy (≥50-60%)

## How Python Uniquely Solves This

### 1. Clean CNN Definition

```python
# Python - intuitive CNN definition
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32*16*16, 10)
```

### 2. Built-in Layers

- `nn.Conv2d`: Convolutional layers
- `nn.MaxPool2d`: Pooling layers
- `nn.BatchNorm2d`: Batch normalization

## Comparison with Other Languages

### Go
- **No CNN libraries**: Would need to implement manually
- **Much more complex**: Requires low-level implementation

### TypeScript
- **Limited ML support**: No mature CNN frameworks

### Rust
- **Less mature**: Fewer CNN implementations

## Deliverables

Complete implementation:
1. CNN model with convolutional and pooling layers
2. Proper architecture for CIFAR-10
3. Training loop (reuse from Project 34)
4. Evaluation on test set
5. Achieve reasonable accuracy

## Testing

Run: `pytest test.py -v`

Tests verify:
- Model architecture is correct
- Forward pass produces correct output shape
- Training runs without errors
- Model achieves reasonable accuracy (≥50%)

## Next Steps

After completing this project, you'll understand:
- How CNNs work for image classification
- How to use convolutional and pooling layers
- How to build CNNs in PyTorch

This knowledge is essential for computer vision tasks!
