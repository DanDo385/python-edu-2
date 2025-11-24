"""
Project 33: Building Neural Network Modules in PyTorch

Complete the class definition below according to its docstring.
Run pytest test.py -v to test your solutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """
    Simple feedforward neural network.
    
    Architecture:
    Input → Linear(784, 128) → ReLU → Linear(128, 10) → Output
    
    This is a basic MLP for MNIST classification.
    """
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """
        Initialize the network.
        
        Args:
            input_size (int): Number of input features (default 784 for MNIST)
            hidden_size (int): Number of hidden neurons (default 128)
            output_size (int): Number of output classes (default 10)
        
        Note:
            - Call super().__init__() first
            - Define layers using nn.Linear()
            - Store layers as instance variables
        """
        # TODO: Initialize network
        # 1. Call super().__init__()
        # 2. Define self.layer1 = nn.Linear(input_size, hidden_size)
        # 3. Define self.layer2 = nn.Linear(hidden_size, output_size)
        pass
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, input_size)
        
        Returns:
            torch.Tensor: Output tensor, shape (batch_size, output_size)
        
        Note:
            - Apply layer1, then ReLU, then layer2
            - Can use F.relu() or nn.ReLU()
        """
        # TODO: Implement forward pass
        # 1. x = self.layer1(x)
        # 2. x = F.relu(x)  # or nn.ReLU()(x)
        # 3. x = self.layer2(x)
        # 4. Return x
        return None


class MLPWithDropout(nn.Module):
    """
    Multi-layer perceptron with dropout regularization.
    
    Architecture:
    Input → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → Output
    """
    
    def __init__(self, input_size=784, hidden1=128, hidden2=64, output_size=10, dropout_rate=0.5):
        """
        Initialize MLP with dropout.
        
        Args:
            input_size (int): Input features
            hidden1 (int): First hidden layer size
            hidden2 (int): Second hidden layer size
            output_size (int): Output classes
            dropout_rate (float): Dropout probability
        """
        # TODO: Initialize layers
        # Define: layer1, dropout1, layer2, dropout2, layer3
        pass
    
    def forward(self, x):
        """
        Forward pass with dropout.
        
        Note: Dropout behaves differently in train vs eval mode
        """
        # TODO: Implement forward pass with dropout
        return None


def count_parameters(model):
    """
    Count total number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        int: Total number of parameters
    
    Note:
        Use model.parameters() and p.numel() to count
    """
    # TODO: Count parameters
    # Hint: sum(p.numel() for p in model.parameters())
    return None


def get_model_output_shape(model, input_shape):
    """
    Get output shape of model for given input shape.
    
    Args:
        model (nn.Module): PyTorch model
        input_shape (tuple): Input shape (batch_size, input_size)
    
    Returns:
        tuple: Output shape
    
    Note:
        Create dummy input, run forward pass, check output shape
    """
    # TODO: Get output shape
    # 1. Create dummy input: torch.randn(input_shape)
    # 2. Run model forward: output = model(dummy_input)
    # 3. Return output.shape
    return None


class SequentialModel(nn.Module):
    """
    Model using nn.Sequential for simpler definition.
    
    Equivalent to SimpleNN but using Sequential container.
    """
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """
        Initialize using nn.Sequential.
        
        Note:
            Use nn.Sequential to stack layers
            Example: nn.Sequential(nn.Linear(...), nn.ReLU(), ...)
        """
        # TODO: Initialize using Sequential
        # self.net = nn.Sequential(...)
        pass
    
    def forward(self, x):
        """
        Forward pass using Sequential.
        
        Note: Just call self.net(x)
        """
        # TODO: Forward through Sequential
        return None
