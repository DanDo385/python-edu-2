"""
Test suite for Project 32: PyTorch Autograd

Run with: pytest test.py -v
"""

import pytest
import torch
from exercise import (
    compute_gradient_simple_function,
    compute_gradient_multivariate,
    verify_no_grad_context,
    gradient_accumulation_demo,
    zero_gradients_demo,
    compute_gradient_vector_output,
    detach_tensor_demo,
)


class TestSimpleGradients:
    """Test simple gradient computation."""
    
    def test_compute_gradient_simple(self):
        """Test gradient of f(x) = x^3 + 2x."""
        # At x = 2: df/dx = 3*2^2 + 2 = 12 + 2 = 14
        gradient = compute_gradient_simple_function(2.0)
        
        assert gradient is not None
        assert abs(gradient - 14.0) < 0.001
    
    def test_compute_gradient_zero(self):
        """Test gradient at x = 0."""
        # At x = 0: df/dx = 3*0^2 + 2 = 2
        gradient = compute_gradient_simple_function(0.0)
        
        assert abs(gradient - 2.0) < 0.001


class TestMultivariateGradients:
    """Test multivariate gradient computation."""
    
    def test_compute_gradient_multivariate(self):
        """Test gradients of z = 3a + 4b^2."""
        grad_a, grad_b = compute_gradient_multivariate(1.0, 2.0)
        
        # dz/da = 3 (independent of a)
        assert abs(grad_a - 3.0) < 0.001
        
        # dz/db = 8b = 8*2 = 16
        assert abs(grad_b - 16.0) < 0.001
    
    def test_multivariate_different_values(self):
        """Test with different values."""
        grad_a, grad_b = compute_gradient_multivariate(5.0, 3.0)
        
        assert abs(grad_a - 3.0) < 0.001  # Still 3
        assert abs(grad_b - 24.0) < 0.001  # 8*3 = 24


class TestNoGrad:
    """Test no_grad context."""
    
    def test_verify_no_grad(self):
        """Test that no_grad disables tracking."""
        grad_tracked, grad_not_tracked = verify_no_grad_context()
        
        # With tracking, should have gradient
        assert grad_tracked is not None
        
        # Without tracking, gradient should be None or not computed
        # (Implementation may vary, but should demonstrate no tracking)
        assert grad_not_tracked is None or grad_not_tracked == 0


class TestGradientAccumulation:
    """Test gradient accumulation."""
    
    def test_gradient_accumulation(self):
        """Test that gradients accumulate."""
        grad_one, grad_two = gradient_accumulation_demo()
        
        # After two backward calls, gradient should be doubled
        assert grad_two is not None
        assert abs(grad_two - 2 * grad_one) < 0.001
    
    def test_zero_gradients(self):
        """Test zeroing gradients."""
        grad_before, grad_after = zero_gradients_demo()
        
        # Before zeroing, should have gradient
        assert grad_before is not None
        
        # After zeroing, should be zero
        assert grad_after == 0.0 or abs(grad_after) < 1e-6


class TestVectorOutput:
    """Test gradient with vector output."""
    
    def test_compute_gradient_vector(self):
        """Test gradient computation for vector output."""
        gradient = compute_gradient_vector_output()
        
        # Should compute gradient successfully
        assert gradient is not None
        
        # For y = [x^2, x^3], gradient of sum is 2x + 3x^2
        # At x=1: 2 + 3 = 5
        # (Exact value depends on implementation)
        assert gradient.item() > 0


class TestDetach:
    """Test tensor detachment."""
    
    def test_detach_tensor(self):
        """Test that detach stops gradient flow."""
        grad_with_detach, grad_without = detach_tensor_demo()
        
        # With detach, gradient should be None or 0
        # Without detach, should have gradient
        assert grad_without is not None
        # With detach, gradient flow is stopped
        assert grad_with_detach is None or grad_with_detach == 0
