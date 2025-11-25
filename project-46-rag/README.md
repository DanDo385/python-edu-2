# Project 46: Retrieval-Augmented Generation (RAG)

## Learning Objectives

- Understand RAG architecture
- Implement document retrieval
- Combine retrieval with generation
- Build RAG system for question answering
- Understand how RAG improves LLM responses

## Problem Description

RAG combines retrieval (finding relevant documents) with generation (LLM producing answers) to improve accuracy and reduce hallucinations.

**Computational Resources:**
- **GPU**: The LLM generation component requires GPU acceleration. Embedding models for document retrieval also benefit from GPU, though smaller embedding models can run on CPU.
- **Memory**: RAG systems need memory for storing document embeddings (vector database) and for running the LLM. GPU memory is required for LLM inference.
- **CPU**: Document processing, retrieval search, and vector database operations can run efficiently on CPU, though GPU acceleration helps for large-scale systems.

**ChatGPT Impact:**
- ChatGPT's limitations (knowledge cutoff, potential hallucinations) drove interest in RAG as a way to ground responses in external knowledge.
- RAG became a popular pattern for building ChatGPT-like systems with domain-specific knowledge, combining retrieval with generation.

## Deliverables

Basic RAG implementation with retrieval and generation components.

## Testing

Run: `pytest test.py -v`
