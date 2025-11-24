"""
Test suite for Project 33: PyTorch Modules

Run with: pytest test.py -v
"""

import pytest
import torch
import torch.nn as nn
from exercise import (
    SimpleNN,
    MLPWithDropout,
    count_parameters,
    get_model_output_shape,
    SequentialModel,
)


class TestSimpleNN:
    """Test SimpleNN model."""
    
    def test_model_initialization(self):
        """Test model can be instantiated."""
        model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
        assert isinstance(model, nn.Module)
    
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
        x = torch.randn(32, 784)  # Batch of 32 samples
        
        output = model(x)
        
        assert output.shape == (32, 10)
    
    def test_forward_pass_values(self):
        """Test forward pass produces valid values."""
        model = SimpleNN(input_size=10, hidden_size=5, output_size=2)
        x = torch.randn(1, 10)
        
        output = model(x)
        
        # Should have valid float values
        assert torch.isfinite(output).all()
        assert output.dtype == torch.float32
    
    def test_parameters_tracked(self):
        """Test that parameters are tracked."""
        model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
        
        # Should have parameters
        params = list(model.parameters())
        assert len(params) >= 2  # At least 2 layers (weights + biases)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
        
        total_params = count_parameters(model)
        
        # Expected: 784*128 + 128 + 128*10 + 10 = 100480 + 138 = 100618
        expected = 784 * 128 + 128 + 128 * 10 + 10
        assert total_params == expected


class TestMLPWithDropout:
    """Test MLPWithDropout model."""
    
    def test_model_initialization(self):
        """Test model can be instantiated."""
        model = MLPWithDropout(input_size=784, hidden1=128, hidden2=64, output_size=10)
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass works."""
        model = MLPWithDropout(input_size=10, hidden1=5, hidden2=3, output_size=2)
        x = torch.randn(1, 10)
        
        output = model(x)
        
        assert output.shape == (1, 2)
    
    def test_dropout_in_model(self):
        """Test that dropout is in the model."""
        model = MLPWithDropout()
        
        # Check that model has dropout layers
        has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
        assert has_dropout


class TestSequentialModel:
    """Test SequentialModel."""
    
    def test_sequential_model(self):
        """Test Sequential model works."""
        model = SequentialModel(input_size=784, hidden_size=128, output_size=10)
        x = torch.randn(32, 784)
        
        output = model(x)
        
        assert output.shape == (32, 10)
    
    def test_sequential_equivalent(self):
        """Test Sequential model is equivalent to SimpleNN."""
        model_seq = SequentialModel(input_size=10, hidden_size=5, output_size=2)
        model_simple = SimpleNN(input_size=10, hidden_size=5, output_size=2)
        
        x = torch.randn(1, 10)
        
        # Both should produce same shape
        assert model_seq(x).shape == model_simple(x).shape


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_count_parameters(self):
        """Test parameter counting function."""
        model = SimpleNN(input_size=10, hidden_size=5, output_size=2)
        
        count = count_parameters(model)
        
        # Should be positive integer
        assert isinstance(count, int)
        assert count > 0
    
    def test_get_output_shape(self):
        """Test getting output shape."""
        model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
        
        output_shape = get_model_output_shape(model, (32, 784))
        
        assert output_shape == (32, 10)
    
    def test_output_shape_different_batch(self):
        """Test output shape with different batch size."""
        model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
        
        shape1 = get_model_output_shape(model, (16, 784))
        shape2 = get_model_output_shape(model, (64, 784))
        
        assert shape1[0] == 16
        assert shape2[0] == 64
        assert shape1[1] == shape2[1] == 10  # Same output size


class TestModelUsage:
    """Test model usage patterns."""
    
    def test_model_train_mode(self):
        """Test model train/eval modes."""
        model = MLPWithDropout()
        
        model.train()
        assert model.training == True
        
        model.eval()
        assert model.training == False
    
    def test_model_parameters_iteration(self):
        """Test iterating over model parameters."""
        model = SimpleNN()
        
        param_count = 0
        for param in model.parameters():
            param_count += 1
            assert param.requires_grad == True
        
        assert param_count > 0
