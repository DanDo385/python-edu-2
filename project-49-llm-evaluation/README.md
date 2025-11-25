# Project 49: LLM Evaluation Metrics

## Learning Objectives

- Understand LLM evaluation metrics
- Implement BLEU score
- Implement ROUGE score
- Perplexity calculation
- Human evaluation considerations

## Problem Description

Evaluating LLMs requires appropriate metrics. BLEU, ROUGE, and perplexity measure different aspects of model performance.

**Computational Resources:**
- **CPU**: Metric computation (BLEU, ROUGE) is CPU-bound text processing. These operations are relatively lightweight.
- **GPU**: Perplexity calculation requires running the model on evaluation data, which benefits from GPU acceleration.
- **Memory**: Evaluation datasets and model outputs need to be stored in memory, but requirements are modest compared to training.

**ChatGPT Impact:**
- ChatGPT's release highlighted the limitations of traditional metrics (BLEU, ROUGE) for evaluating conversational AI. Human evaluation became crucial, as ChatGPT showed that high-quality responses don't always correlate with high BLEU scores.
- The need to evaluate ChatGPT-like systems drove development of new evaluation approaches (human preference ranking, instruction following accuracy).

## Deliverables

Evaluation metrics implementation (BLEU, ROUGE, perplexity).

## Testing

Run: `pytest test.py -v`
