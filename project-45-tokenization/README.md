# Project 45: Tokenization and Text Processing

## Learning Objectives

- Understand tokenization (BPE, WordPiece, SentencePiece)
- Implement basic tokenizer
- Handle special tokens
- Convert between text and tokens
- Prepare text for LLMs

## Problem Description

Tokenization converts text to numbers that models can process. Different strategies (BPE, WordPiece) handle vocabulary efficiently.

**Computational Resources:**
- **CPU**: Tokenization is primarily CPU-bound. Text processing, vocabulary lookups, and encoding/decoding operations run efficiently on CPU.
- **Memory (RAM)**: Tokenizers store vocabulary mappings in memory. Large vocabularies (50k+ tokens) require several GB of RAM, but this is manageable on modern systems.
- **GPU**: Not typically used for tokenization itself, though tokenized sequences are then transferred to GPU for model processing.

**ChatGPT Impact:**
- ChatGPT uses GPT tokenization (BPE-based), which became the de facto standard for many LLMs. Understanding tokenization is crucial for working with ChatGPT and similar models.

## Deliverables

Basic tokenizer implementation with encoding/decoding.

## Testing

Run: `pytest test.py -v`
