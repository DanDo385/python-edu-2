# Project 44: Fine-tuning Large Language Models

## Learning Objectives

- Understand fine-tuning strategies for LLMs
- Implement LoRA (Low-Rank Adaptation)
- Fine-tune pretrained models for specific tasks
- Understand parameter-efficient fine-tuning
- Adapt LLMs for downstream tasks

## Problem Description

Fine-tuning adapts pretrained LLMs for specific tasks. LoRA and other techniques make this parameter-efficient.

**Computational Resources:**
- **GPU**: Fine-tuning requires GPU acceleration, though LoRA reduces computational requirements compared to full fine-tuning. LoRA trains only a small subset of parameters, making it feasible on single GPUs (8-16GB VRAM) for models up to 7B parameters.
- **Memory**: LoRA significantly reduces memory requirements - instead of storing gradients for all parameters, only low-rank adaptation matrices need gradients. This makes fine-tuning accessible with less VRAM.
- **CPU**: Used for data processing and coordination, but training is GPU-bound.

**ChatGPT Impact:**
- ChatGPT's success led to widespread adoption of fine-tuning techniques for adapting large models to specific domains and tasks.
- The popularity of ChatGPT created demand for fine-tuning tools and techniques, making LoRA and other parameter-efficient methods mainstream.

## Deliverables

LoRA implementation and fine-tuning pipeline for LLMs.

## Testing

Run: `pytest test.py -v`
