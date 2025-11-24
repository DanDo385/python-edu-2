"""
Project 33: PyTorch Modules - SOLUTION

Complete solution demonstrating nn.Module usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """
    Simple feedforward neural network.
    
    Architecture: Input → Linear(784, 128) → ReLU → Linear(128, 10) → Output
    """
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """Initialize the network."""
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x


class MLPWithDropout(nn.Module):
    """
    Multi-layer perceptron with dropout regularization.
    """
    
    def __init__(self, input_size=784, hidden1=128, hidden2=64, output_size=10, dropout_rate=0.5):
        """Initialize MLP with dropout."""
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(hidden2, output_size)
    
    def forward(self, x):
        """Forward pass with dropout."""
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.layer3(x)
        return x


def count_parameters(model):
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def get_model_output_shape(model, input_shape):
    """Get output shape of model for given input shape."""
    model.eval()  # Set to eval mode
    with torch.no_grad():
        dummy_input = torch.randn(input_shape)
        output = model(dummy_input)
    return output.shape


class SequentialModel(nn.Module):
    """
    Model using nn.Sequential for simpler definition.
    """
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """Initialize using nn.Sequential."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        """Forward pass using Sequential."""
        return self.net(x)


# Example usage
if __name__ == "__main__":
    print("PyTorch Modules Demo")
    print("=" * 50)
    
    # Create model
    model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
    print(f"Model created: {model}")
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(32, 784)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Sequential model
    seq_model = SequentialModel()
    seq_output = seq_model(x)
    print(f"\nSequential model output shape: {seq_output.shape}")
    
    print("\nModels are ready for training! 🚀")
