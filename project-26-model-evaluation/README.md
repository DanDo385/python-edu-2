# Project 26: Model Evaluation and Data Splitting

## Learning Objectives

- Understand why we split data into training, validation, and test sets
- Learn to avoid data leakage and measure performance correctly
- Implement data splitting functions with proper shuffling
- Compute evaluation metrics for classification and regression
- Understand overfitting vs generalization through proper evaluation
- Build a confusion matrix for classification tasks

## Problem Description

A fundamental mistake in machine learning is measuring performance on training data alone. A model that memorizes training examples will score perfectly on training data but fail on new data. Proper evaluation requires separating data and using appropriate metrics.

**Why Split Data?**
- **Training set**: Used to learn model parameters
- **Validation set**: Used to tune hyperparameters and detect overfitting
- **Test set**: Used only for final evaluation (never used during training!)

**The Danger of Data Leakage:**
- Using test data during training → overly optimistic results
- Not shuffling before splitting → temporal bias
- Measuring performance on training data → overfitting goes undetected

## Key Concepts

### Train/Validation/Test Split

```
Full Dataset
    │
    ├── Training Set (70-80%)
    │   └── Used to learn weights/parameters
    │
    ├── Validation Set (10-15%)
    │   └── Used to tune hyperparameters, detect overfitting
    │
    └── Test Set (10-15%)
        └── Used ONLY for final evaluation (never touched during training!)
```

### Why We Don't Measure on Training Data Alone

```
Training Accuracy: 95%  ← Model memorized training examples
Test Accuracy: 60%      ← Model fails on new data (OVERFITTING!)

The gap between training and test performance reveals overfitting.
```

### Evaluation Metrics

**For Classification:**
- **Accuracy**: (Correct predictions) / (Total predictions)
- **Precision**: (True Positives) / (True Positives + False Positives)
- **Recall**: (True Positives) / (True Positives + False Negatives)
- **Confusion Matrix**: Counts of TP, TN, FP, FN

**For Regression:**
- **Mean Squared Error (MSE)**: Average of squared differences
- **Mean Absolute Error (MAE)**: Average of absolute differences

### Data Splitting Best Practices

1. **Shuffle first**: Randomize order to avoid temporal bias
2. **Stratified split**: Maintain class distribution (for classification)
3. **No overlap**: Ensure sets are disjoint
4. **Fixed seed**: Reproducible splits for experiments

## Solution Approach

### Understanding the Evaluation Pipeline

```
1. Load dataset
2. Shuffle data (with random seed for reproducibility)
3. Split into train/val/test sets
4. Train model on training set
5. Evaluate on validation set (tune hyperparameters)
6. Final evaluation on test set (only once!)
```

### Metric Computation

**Classification Metrics:**
- Compare predicted labels vs true labels
- Count correct/incorrect predictions
- Build confusion matrix for detailed analysis

**Regression Metrics:**
- Compute differences between predictions and targets
- Square differences for MSE (penalizes large errors)
- Use absolute differences for MAE (robust to outliers)

## How Python Uniquely Solves This

### 1. NumPy for Efficient Splitting

```python
# Python - clean array operations
indices = np.arange(len(data))
np.random.shuffle(indices)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

# vs. Other languages - more verbose indexing
```

### 2. Vectorized Metric Computation

```python
# Python - vectorized operations
accuracy = np.mean(y_true == y_pred)  # Fast comparison
mse = np.mean((y_true - y_pred) ** 2)  # Vectorized squaring

# vs. Explicit loops in other languages
```

### 3. Rich Ecosystem

```python
# Python has sklearn for reference (though we implement from scratch)
from sklearn.model_selection import train_test_split  # Reference implementation

# But understanding the fundamentals is crucial!
```

## Comparison with Other Languages

### Go
- **No built-in ML libraries**: Would need to implement everything manually
- **Type safety**: More verbose but catches errors early
- **Performance**: Fast but requires more code

### TypeScript
- **Limited ML support**: Not designed for scientific computing
- **Web-focused**: Better for frontend applications

### Rust
- **Performance**: Can match NumPy performance
- **Complexity**: Steeper learning curve
- **Ecosystem**: Less mature for ML evaluation tools

## Deliverables

Complete functions for:
1. Data splitting with shuffling (train/val/test split)
2. Classification metrics (accuracy, confusion matrix)
3. Regression metrics (MSE, MAE)
4. Verification that splits are disjoint and properly sized
5. Comparison of training vs test performance

## Testing

Run: `pytest test.py -v`

Tests verify:
- Data splitting produces correct ratios
- No data points overlap between sets
- Metrics compute correctly on known inputs
- Training performance > test performance (generalization gap)

## Next Steps

After completing this project, you'll understand:
- Why proper evaluation is critical for ML
- How to detect overfitting through evaluation
- How to compute metrics correctly
- The importance of data splitting

This foundation is essential for all future ML projects - you'll always split data and evaluate properly, whether training simple models or complex deep networks.
