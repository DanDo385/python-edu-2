# Project 36: RNNs and LSTMs for Sequence Modeling

## Learning Objectives

- Understand Recurrent Neural Networks (RNNs) and their applications
- Learn Long Short-Term Memory (LSTM) networks
- Build RNN/LSTM models for sequence data
- Understand how RNNs handle sequential dependencies
- Train RNNs on sequence classification tasks
- Compare RNNs to feedforward networks for sequential data

## Problem Description

Recurrent Neural Networks (RNNs) are designed for sequential data where order matters. Unlike feedforward networks, RNNs have memory and can process sequences of variable length.

**Why RNNs?**
- Handle sequences of variable length
- Capture temporal dependencies
- Share parameters across time steps
- Natural for time series, text, speech

**Applications:**
- Text classification
- Time series prediction
- Language modeling
- Sentiment analysis

## Key Concepts

### RNN Basics

**Simple RNN:**
```python
nn.RNN(input_size, hidden_size, num_layers)
```

**How RNNs Work:**
- Process sequence one element at a time
- Maintain hidden state across time steps
- Hidden state carries information from previous steps

**Forward Pass:**
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
```

### LSTM Networks

**LSTM:**
```python
nn.LSTM(input_size, hidden_size, num_layers)
```

**Why LSTM?**
- Solves vanishing gradient problem
- Better long-term memory
- Gating mechanisms (forget, input, output gates)

**LSTM Components:**
- Forget gate: What to forget
- Input gate: What to remember
- Output gate: What to output

### Sequence Processing

**Input Format:**
- Shape: `(sequence_length, batch_size, input_size)`
- Or: `(batch_size, sequence_length, input_size)` with `batch_first=True`

**Output:**
- Hidden states at each time step
- Final hidden state (for classification)

## Solution Approach

### Building RNN Models

1. **Define RNN/LSTM layer**: Use `nn.RNN` or `nn.LSTM`
2. **Process sequences**: Handle variable-length sequences
3. **Extract final state**: Use last hidden state for classification
4. **Add classifier**: Fully connected layer for output

### Training

- Same training loop as Project 34
- Handle sequence data properly
- Use appropriate loss functions

## How Python Uniquely Solves This

### 1. Clean RNN Definition

```python
# Python - intuitive RNN definition
class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
```

### 2. Built-in RNN Layers

- `nn.RNN`: Simple RNN
- `nn.LSTM`: Long Short-Term Memory
- `nn.GRU`: Gated Recurrent Unit

## Comparison with Other Languages

### Go
- **No RNN libraries**: Would need to implement manually
- **Much more complex**: Requires low-level implementation

### TypeScript
- **Limited ML support**: No mature RNN frameworks

### Rust
- **Less mature**: Fewer RNN implementations

## Deliverables

Complete implementation:
1. Simple RNN model
2. LSTM model
3. Sequence processing functions
4. Training on sequence data
5. Comparison with feedforward networks

## Testing

Run: `pytest test.py -v`

Tests verify:
- RNN/LSTM models can be instantiated
- Forward pass produces correct output shape
- Models handle sequences correctly
- Training runs without errors

## Next Steps

After completing this project, you'll understand:
- How RNNs process sequential data
- When to use RNNs vs feedforward networks
- How LSTMs improve upon simple RNNs

This knowledge is essential for sequence modeling tasks!
