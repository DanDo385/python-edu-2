# Project 39: Sequence-to-Sequence Models

## Learning Objectives

- Understand sequence-to-sequence (seq2seq) architecture
- Build encoder-decoder models using RNNs/LSTMs
- Implement attention mechanism for seq2seq
- Understand how seq2seq models work for translation and generation
- Build a complete encoder-decoder system

## Problem Description

Sequence-to-sequence models map variable-length input sequences to variable-length output sequences. They're used for machine translation, text summarization, and dialogue systems.

**Why Seq2Seq?**
- Handle variable-length inputs and outputs
- Natural for translation tasks
- Foundation for many NLP applications
- Enables sequence generation

**Applications:**
- Machine translation
- Text summarization
- Dialogue systems
- Question answering

## Key Concepts

### Encoder-Decoder Architecture

**Encoder:**
- Processes input sequence
- Produces context vector
- Captures input information

**Decoder:**
- Generates output sequence
- Uses context from encoder
- Produces one token at a time

### Attention Mechanism

**Why Attention?**
- Context vector is bottleneck
- Attention allows decoder to focus on relevant parts
- Improves performance significantly

## Solution Approach

### Building Seq2Seq Models

1. **Encoder**: RNN/LSTM that processes input
2. **Decoder**: RNN/LSTM that generates output
3. **Attention**: Connect encoder and decoder
4. **Training**: Teacher forcing for training

## Deliverables

Complete implementation:
1. Encoder model
2. Decoder model
3. Seq2Seq model combining both
4. Attention mechanism
5. Training loop for seq2seq

## Testing

Run: `pytest test.py -v`

## Next Steps

After completing this project, you'll understand:
- How encoder-decoder models work
- How attention improves seq2seq
- Foundation for transformers
