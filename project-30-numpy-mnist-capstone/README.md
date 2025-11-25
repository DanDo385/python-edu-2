# Project 30: NumPy Neural Network Capstone – MNIST Digit Classifier

## Learning Objectives

- Build a complete neural network from scratch using only NumPy
- Handle a real-world dataset (MNIST handwritten digits)
- Implement multi-class classification with softmax and cross-entropy loss
- Train a network to recognize handwritten digits (0-9)
- Understand the limitations of pure NumPy (motivation for PyTorch)
- Achieve a major milestone: building a working digit classifier from first principles

## Problem Description

This is the capstone project of Phase II (NumPy-based ML). You'll build a complete neural network that solves a real-world task: recognizing handwritten digits from the MNIST dataset.

**MNIST Dataset:**
- 28x28 pixel grayscale images of handwritten digits (0-9)
- 60,000 training images, 10,000 test images
- Classic benchmark in computer vision
- Flattened: 784 input features (28 * 28)

**The Challenge:**
- Build a 2-3 layer neural network using only NumPy
- Train it to classify digits 0-9 (10 classes)
- Achieve reasonable accuracy (≥85% with simple network)
- Understand that while this works, PyTorch will make it much easier!

## Key Concepts

### Multi-Class Classification

**Difference from Binary Classification:**
- Binary: 2 classes (sigmoid output, binary cross-entropy)
- Multi-class: 10 classes (softmax output, cross-entropy)

**Softmax Activation:**
```
softmax(z_i) = exp(z_i) / Σ exp(z_j)

- Converts logits to probabilities
- Outputs sum to 1.0
- Each output is probability of that class
```

**Cross-Entropy Loss:**
```
Loss = -Σ y_true * log(y_pred)

- Measures difference between true and predicted distributions
- For one-hot encoded labels: -log(y_pred[true_class])
```

### Network Architecture

```
Input (784) → Hidden Layer (128, ReLU) → Output (10, Softmax) → Predictions
```

**Why This Architecture?**
- Input: 784 features (flattened 28x28 image)
- Hidden: 128 neurons (good balance of capacity and speed)
- Output: 10 neurons (one per digit class)

### Training Process

```
1. Load and preprocess MNIST data
2. Initialize network weights
3. For each epoch:
   a. For each batch:
      - Forward pass: Compute predictions
      - Compute loss: Cross-entropy
      - Backward pass: Compute gradients
      - Update weights: Gradient descent
4. Evaluate on test set
5. Visualize results (show predictions on sample images)
```

## Solution Approach

### Data Preprocessing

1. **Flatten images**: 28x28 → 784 features
2. **Normalize**: Scale pixel values to [0, 1] (divide by 255)
3. **One-hot encode labels**: 5 → [0,0,0,0,0,1,0,0,0,0]

### Forward Pass

1. **Input → Hidden**: Linear + ReLU
2. **Hidden → Output**: Linear + Softmax
3. **Output**: Probabilities for each class

### Loss Computation

- **Cross-entropy**: For multi-class classification
- **One-hot encoding**: True label is vector with 1 at correct class

### Backward Pass

- Compute gradients through softmax and cross-entropy
- Backpropagate through ReLU
- Update all weights and biases

## How Python Uniquely Solves This

### 1. NumPy for Efficient Computation

```python
# Vectorized operations
predictions = softmax(X @ W + b)  # Entire batch at once
loss = -np.mean(np.log(predictions[range(batch_size), y_true]))  # Vectorized loss
```

### 2. Easy Data Handling

```python
# Simple data loading and preprocessing
images = images / 255.0  # Normalize
images = images.reshape(-1, 784)  # Flatten
```

### 3. Visualization

```python
# Matplotlib for visualizing results
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f'Predicted: {predicted_class}')
```

## Deliverables

Complete implementation:
1. Load and preprocess MNIST data
2. Multi-layer neural network class
3. Softmax activation function
4. Cross-entropy loss function
5. Training loop with mini-batches
6. Evaluation on test set
7. Visualization of predictions

## Testing

Run: `pytest test.py -v`

Tests verify:
- Network architecture is correct (784 → 128 → 10)
- Softmax outputs sum to 1.0
- Cross-entropy loss computes correctly
- Training reduces loss
- Test accuracy ≥ 85% (with reasonable training)

## Next Steps

After completing this project, you'll have:
- Built a complete neural network from scratch
- Solved a real-world problem (digit recognition)
- Understood the full training pipeline
- **Appreciated why PyTorch exists** (this is a lot of code!)

This capstone demonstrates that neural networks work, but also shows why frameworks like PyTorch are essential for larger-scale work. Next, we'll use PyTorch to do this much more easily and efficiently!
