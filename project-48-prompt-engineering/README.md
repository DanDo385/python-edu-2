# Project 48: Prompt Engineering

## Learning Objectives

- Understand prompt engineering techniques
- Implement few-shot prompting
- Chain-of-thought prompting
- Prompt templates
- Optimize prompts for better results

## Problem Description

Prompt engineering is crucial for getting good results from LLMs. Different techniques (few-shot, chain-of-thought) improve performance.

**Computational Resources:**
- **CPU**: Prompt engineering is primarily about text manipulation and template construction, which runs on CPU.
- **GPU**: The actual LLM inference that processes prompts requires GPU, but prompt engineering itself is CPU-bound text processing.
- **Memory**: Prompts consume context window space. Longer prompts (few-shot examples, chain-of-thought) require more GPU memory for processing.

**ChatGPT Impact:**
- ChatGPT's success made prompt engineering a critical skill. Users discovered that carefully crafted prompts dramatically improved ChatGPT's performance.
- Chain-of-thought prompting became popular after ChatGPT demonstrated its effectiveness, showing that models could reason better when prompted to show their work.
- ChatGPT's conversational interface made prompt engineering accessible to non-technical users, democratizing LLM interaction.

## Deliverables

Prompt engineering utilities and templates.

## Testing

Run: `pytest test.py -v`
