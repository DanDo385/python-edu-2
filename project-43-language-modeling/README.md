# Project 43: Language Modeling with Transformers

## Learning Objectives

- Understand language modeling (predicting next token)
- Build GPT-like decoder-only transformer
- Implement causal (masked) self-attention
- Train language model on text data
- Generate text from trained model

## Problem Description

Language modeling predicts the next token in a sequence. GPT models use decoder-only transformers with causal masking.

**Computational Resources:**
- **GPU**: Language models are extremely GPU-intensive. Training GPT-scale models requires multiple GPUs or specialized hardware. Inference also benefits significantly from GPU acceleration.
- **Memory**: Large language models require substantial GPU memory. GPT-3 (175B parameters) requires hundreds of GB of VRAM. Even smaller models (1-7B parameters) need 16-40GB VRAM for training.
- **CPU**: Used for tokenization, data loading, and text processing, but model forward/backward passes are GPU-bound.

**ChatGPT Impact:**
- ChatGPT demonstrated that decoder-only transformers (GPT architecture) could excel at conversational AI through careful training and alignment techniques.
- It showed the importance of instruction tuning and reinforcement learning from human feedback (RLHF) for creating useful, safe language models.
- ChatGPT's success validated the scaling hypothesis - larger models trained on more data produce better results.

## Deliverables

Decoder-only transformer for language modeling with training and generation.

## Testing

Run: `pytest test.py -v`
