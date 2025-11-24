# Python-50x Minis: 50 Progressive Projects for Python, DSA & AI/ML Learning

Welcome to a comprehensive, hands-on curriculum designed to take you from Python basics to building and deploying Large Language Models (LLMs). This curriculum is structured in four progressive phases, each building on the previous one.

## 📚 Curriculum Overview

### Phase I — Python & DSA Fundamentals (Projects 01–15)
Master Python syntax, data structures, algorithms, and object-oriented programming. Build a solid foundation for everything that follows.

**Learning Path:**
- **Projects 01-03**: Python basics (syntax, control flow, functions)
- **Projects 04-05**: Data structures (lists, tuples, dictionaries, sets)
- **Projects 06-07**: Object-oriented programming (classes, inheritance, exceptions)
- **Projects 08-10**: Algorithms (recursion, searching, sorting)
- **Projects 11-15**: Advanced data structures and algorithms (stacks, queues, linked lists, trees, graphs, dynamic programming)

### Phase II — NumPy & ML Math Foundations (Projects 16–30)
Learn numerical computing with NumPy, implement machine learning algorithms from scratch, and understand the mathematical foundations of neural networks.

**Learning Path:**
- **Projects 16-17**: NumPy fundamentals and advanced operations
- **Projects 18-19**: Linear algebra and gradient descent
- **Projects 20-21**: Machine learning basics (linear regression, logistic regression)
- **Projects 22-23**: Neural network foundations (activation functions, backpropagation)
- **Projects 24-25**: Advanced neural network concepts (automatic differentiation, training MLPs)
- **Projects 26-27**: Model evaluation and regularization
- **Projects 28-29**: Hyperparameter tuning and optimization techniques
- **Project 30**: NumPy capstone — Build complete MNIST classifier from scratch

### Phase III — Deep Learning Systems with PyTorch (Projects 31–40)
Build deep learning models using PyTorch, including CNNs, RNNs, and sequence-to-sequence models.

**Learning Path:**
- **Projects 31-32**: PyTorch fundamentals (tensors, autograd)
- **Projects 33-34**: Building and training neural networks
- **Project 35**: Convolutional Neural Networks (CNNs) for image classification
- **Project 36**: RNNs and LSTMs for sequence modeling
- **Project 37**: Advanced CNNs and transfer learning
- **Project 38**: Data augmentation and regularization
- **Projects 39-40**: Sequence-to-sequence models and attention mechanisms

### Phase IV — Transformers, LLMs, and Modern AI Systems (Projects 41–50)
Implement transformers from scratch, train language models, and build production-ready LLM systems with RAG and deployment strategies.

**Learning Path:**
- **Projects 41-42**: Transformer architecture and complete implementation
- **Project 43**: Language modeling with transformers (GPT-style)
- **Projects 44-45**: Fine-tuning LLMs and tokenization
- **Project 46**: Retrieval-Augmented Generation (RAG)
- **Project 47**: LLM inference optimization
- **Project 48**: Prompt engineering techniques
- **Project 49**: LLM evaluation metrics
- **Project 50**: LLM deployment and production systems

## 🎯 How to Use This Curriculum

### Prerequisites
- Python 3.12 or later installed
- Basic command-line familiarity
- A text editor or IDE (VS Code, PyCharm, etc.)

### Project Structure

Each project follows this structure:

```
project-XX-project-name/
├── README.md              # Detailed problem description, learning objectives, and concepts
├── exercise.py            # YOUR TASK: Complete the functions marked with TODO
├── test.py                # Pytest tests to verify your solution
├── solution.py             # Fully commented reference solution
├── solution_in_words.md    # Step-by-step thinking process (no code)
└── requirements.txt       # Python package dependencies (for projects that need them)
```

### Workflow for Each Project

1. **Read the README.md**
   - Understand the problem and learning objectives
   - Review the solution approach and key concepts
   - Note how Python handles this differently from other languages

2. **Read solution_in_words.md**
   - Understand the thinking process
   - Learn how to approach the problem conceptually
   - No code here—just pure problem-solving logic

3. **Work on exercise.py**
   - Open `exercise.py` in the project directory
   - Find functions marked with `# TODO:` or incomplete implementations
   - Complete the functions according to the docstrings and comments

4. **Test Your Solution**
   ```bash
   # Navigate to the project directory
   cd project-XX-project-name
   
   # Install dependencies if needed (check requirements.txt)
   pip install -r requirements.txt
   
   # Run pytest (install if needed: pip install pytest)
   pytest test.py -v
   
   # Or run a specific test
   pytest test.py::test_function_name -v
   ```

5. **Compare with solution.py**
   - After attempting the exercise, review `solution.py`
   - Study the detailed comments explaining each step
   - Understand alternative approaches and edge cases

### Understanding pytest

**pytest** is a testing framework that makes writing and running tests simple.

#### Installation
```bash
pip install pytest
```

#### Basic Usage
- Run all tests: `pytest test.py`
- Run with verbose output: `pytest test.py -v`
- Run specific test: `pytest test.py::test_function_name`
- Run and show print statements: `pytest test.py -s`

#### How Tests Work
Tests in `test.py` import functions from `exercise.py` and verify they produce correct outputs:

```python
from exercise import my_function

def test_my_function():
    result = my_function(input_value)
    assert result == expected_output
```

If your `exercise.py` is correct, all tests pass (✓). If not, pytest shows what went wrong.

### Testing Strategy

1. **Write your solution** in `exercise.py`
2. **Run tests** to check correctness: `pytest test.py -v`
3. **Fix errors** based on test output
4. **Iterate** until all tests pass
5. **Review** `solution.py` for best practices and edge cases

### Progress Tracking

#### Phase I: Python & DSA Fundamentals (Projects 01–15)
- [ ] Project 01: Basic Python Syntax and Variables
- [ ] Project 02: Control Flow and Loops
- [ ] Project 03: Functions and Modular Programming
- [ ] Project 04: Data Structures I – Lists and Tuples
- [ ] Project 05: Data Structures II – Dictionaries and Sets
- [ ] Project 06: Python OOP Basics
- [ ] Project 07: Python OOP Advanced (Inheritance & Exceptions)
- [ ] Project 08: Recursion and Divide-and-Conquer
- [ ] Project 09: Searching Algorithms
- [ ] Project 10: Sorting Algorithms
- [ ] Project 11: Stack and Queue
- [ ] Project 12: Linked List
- [ ] Project 13: Trees (Binary Tree Basics)
- [ ] Project 14: Graphs and Graph Traversal Algorithms
- [ ] Project 15: Dynamic Programming Fundamentals

#### Phase II: NumPy & ML Math Foundations (Projects 16–30)
- [ ] Project 16: NumPy Basics
- [ ] Project 17: NumPy Advanced Operations
- [ ] Project 18: Linear Algebra with NumPy
- [ ] Project 19: Gradient Descent Implementation
- [ ] Project 20: Linear Regression from Scratch
- [ ] Project 21: Logistic Regression from Scratch
- [ ] Project 22: Activation Functions
- [ ] Project 23: Backpropagation Algorithm
- [ ] Project 24: Automatic Differentiation
- [ ] Project 25: Training a Shallow Neural Network (MLP)
- [ ] Project 26: Model Evaluation and Data Splitting
- [ ] Project 27: Overfitting and Regularization Techniques
- [ ] Project 28: Hyperparameter Tuning and Experimentation
- [ ] Project 29: Mini-Batch vs Stochastic Gradient Descent
- [ ] Project 30: NumPy Neural Network Capstone – MNIST Digit Classifier

#### Phase III: Deep Learning Systems with PyTorch (Projects 31–40)
- [ ] Project 31: PyTorch Tensors and GPU Fundamentals
- [ ] Project 32: PyTorch Autograd – Automatic Differentiation
- [ ] Project 33: Building Neural Network Modules in PyTorch
- [ ] Project 34: Training a PyTorch Model (MLP on MNIST)
- [ ] Project 35: Convolutional Neural Networks for CIFAR-10
- [ ] Project 36: RNNs and LSTMs for Sequence Modeling
- [ ] Project 37: Advanced CNNs and Transfer Learning
- [ ] Project 38: Data Augmentation and Advanced Regularization
- [ ] Project 39: Sequence-to-Sequence Models
- [ ] Project 40: Attention Mechanisms

#### Phase IV: Transformers, LLMs, and Modern AI Systems (Projects 41–50)
- [ ] Project 41: Transformer Architecture
- [ ] Project 42: Complete Transformer Implementation
- [ ] Project 43: Language Modeling with Transformers
- [ ] Project 44: Fine-tuning Large Language Models
- [ ] Project 45: Tokenization and Text Processing
- [ ] Project 46: Retrieval-Augmented Generation (RAG)
- [ ] Project 47: LLM Inference and Optimization
- [ ] Project 48: Prompt Engineering
- [ ] Project 49: LLM Evaluation Metrics
- [ ] Project 50: LLM Deployment and Production

## 🎓 Learning Progression Explained

### Phase I: Foundation (Projects 01-15)
**Goal**: Master Python and fundamental computer science concepts.

You'll start with basic Python syntax and progress through data structures and algorithms. This phase builds the programming foundation needed for everything else. By the end, you'll understand:
- How to write clean, efficient Python code
- How data structures work internally
- How to solve problems algorithmically
- Object-oriented programming principles

**Key Milestones:**
- Project 05: Comfortable with Python data structures
- Project 10: Can implement common algorithms
- Project 15: Understand advanced problem-solving techniques

### Phase II: Mathematical Foundations (Projects 16-30)
**Goal**: Understand the math behind machine learning and build neural networks from scratch.

This phase bridges programming and machine learning. You'll implement everything using NumPy, understanding every detail. This deep understanding makes PyTorch intuitive later.

**Key Milestones:**
- Project 20: Understand linear models
- Project 25: Can build and train neural networks from scratch
- Project 30: Complete ML system (MNIST classifier) using only NumPy

**Why This Matters**: Understanding the fundamentals makes advanced frameworks like PyTorch much easier to learn.

### Phase III: Deep Learning with PyTorch (Projects 31-40)
**Goal**: Master PyTorch and build sophisticated deep learning models.

Now you'll use PyTorch, which handles the low-level details you learned in Phase II. You'll build CNNs, RNNs, and attention-based models, applying the concepts from Phase II with powerful tools.

**Key Milestones:**
- Project 34: Complete PyTorch training pipeline
- Project 35: CNNs for computer vision
- Project 36: RNNs for sequence data
- Project 40: Attention mechanisms (foundation for transformers)

**Why This Matters**: PyTorch is the industry standard. These projects teach you to build production-ready models.

### Phase IV: Modern AI Systems (Projects 41-50)
**Goal**: Build and deploy Large Language Models.

This final phase brings everything together. You'll implement transformers (the architecture behind GPT, BERT, etc.), fine-tune LLMs, build RAG systems, and deploy models to production.

**Key Milestones:**
- Project 42: Complete transformer implementation
- Project 43: Language modeling (GPT-style)
- Project 46: RAG system (combines retrieval + generation)
- Project 50: Production deployment

**Why This Matters**: LLMs are transforming technology. These projects teach you to work with cutting-edge AI systems.

## 💡 Tips for Success

1. **Don't skip ahead**: Each project builds on previous concepts. Skipping will make later projects harder.

2. **Read thoroughly**: Understanding the "why" is as important as the "how". The README and solution_in_words.md explain concepts deeply.

3. **Experiment**: Try variations and edge cases beyond the exercises. This deepens understanding.

4. **Use the solution wisely**: Try solving first, then review the solution. It's a learning resource, not a cheat sheet.

5. **Take notes**: Document insights and patterns you discover. You'll reference these later.

6. **Install dependencies**: Some projects require additional packages. Check `requirements.txt` and install with `pip install -r requirements.txt`.

7. **Run tests frequently**: Don't wait until the end. Run tests after each function to catch errors early.

## 🔧 Getting Help

- **Conceptual questions**: Review `solution_in_words.md` for step-by-step thinking
- **Implementation questions**: Check `solution.py` for detailed code explanations
- **Test failures**: Read pytest error messages carefully—they point to the exact issue
- **Understanding differences**: Compare your approach with the solution to see alternative methods

## 🚀 Getting Started

1. **Navigate to Project 01:**
   ```bash
   cd project-01-basic-python-syntax
   ```

2. **Read the documentation:**
   - Start with `README.md` to understand the project
   - Review `solution_in_words.md` for conceptual guidance

3. **Start coding:**
   - Open `exercise.py`
   - Complete the TODO items
   - Run tests: `pytest test.py -v`

4. **Learn from solutions:**
   - After attempting, review `solution.py`
   - Understand the approach and best practices

5. **Move to the next project:**
   - Once tests pass, move to Project 02
   - Each project builds on the previous one

## 📈 Expected Timeline

- **Phase I** (Projects 01-15): 2-3 weeks for beginners, 1 week if experienced
- **Phase II** (Projects 16-30): 3-4 weeks (more mathematical concepts)
- **Phase III** (Projects 31-40): 2-3 weeks (PyTorch learning curve)
- **Phase IV** (Projects 41-50): 2-3 weeks (advanced concepts)

**Total**: Approximately 10-13 weeks for complete beginners, 6-8 weeks with some programming experience.

## 🎉 Completion

After completing all 50 projects, you'll have:
- Strong Python programming skills
- Deep understanding of data structures and algorithms
- Complete knowledge of neural networks (from scratch to PyTorch)
- Ability to build and deploy LLMs
- Production-ready AI/ML skills

**Congratulations on starting this journey!** Each project brings you closer to mastering Python, data science, and modern AI systems. 🐍🚀
