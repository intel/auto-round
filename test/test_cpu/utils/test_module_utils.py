#!/usr/bin/env python3
"""
Test for get_module and set_module functions using PyTorch native APIs.
This test validates the replacement of custom implementations with torch.nn.Module.get_submodule
and torch.nn.Module.set_submodule.
"""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

import torch
import torch.nn as nn
from auto_round.utils.model import get_module, set_module, get_attr, set_attr


class TestModel(nn.Module):
    """Test model with various module structures."""
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30)
        )
        self.decoder = nn.Linear(30, 10)
        self.nested = nn.ModuleDict({
            'layer1': nn.Linear(5, 10),
            'layer2': nn.Linear(10, 5)
        })
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class TestGetModule:
    """Test cases for get_module function."""
    
    def test_get_top_level_module(self):
        """Test getting a top-level module."""
        model = TestModel()
        encoder = get_module(model, 'encoder')
        assert encoder is not None
        assert isinstance(encoder, nn.Sequential)
        assert len(encoder) == 3
    
    def test_get_nested_module(self):
        """Test getting a nested module."""
        model = TestModel()
        first_layer = get_module(model, 'encoder.0')
        assert first_layer is not None
        assert isinstance(first_layer, nn.Linear)
        assert first_layer.in_features == 10
        assert first_layer.out_features == 20
    
    def test_get_deeply_nested_module(self):
        """Test getting a deeply nested module."""
        model = TestModel()
        relu = get_module(model, 'encoder.1')
        assert relu is not None
        assert isinstance(relu, nn.ReLU)
    
    def test_get_nonexistent_module(self):
        """Test getting a non-existent module returns None."""
        model = TestModel()
        result = get_module(model, 'nonexistent')
        assert result is None
    
    def test_get_from_module_dict(self):
        """Test getting a module from ModuleDict."""
        model = TestModel()
        layer1 = get_module(model, 'nested.layer1')
        assert layer1 is not None
        assert isinstance(layer1, nn.Linear)
        assert layer1.in_features == 5
        assert layer1.out_features == 10
    
    def test_get_nonexistent_nested(self):
        """Test getting a non-existent nested module returns None."""
        model = TestModel()
        result = get_module(model, 'encoder.999')
        assert result is None


class TestSetModule:
    """Test cases for set_module function."""
    
    def test_set_top_level_module(self):
        """Test replacing a top-level module."""
        model = TestModel()
        new_decoder = nn.Linear(30, 5)
        set_module(model, 'decoder', new_decoder)
        
        assert model.decoder.out_features == 5
        retrieved = get_module(model, 'decoder')
        assert retrieved is new_decoder
    
    def test_set_nested_module(self):
        """Test replacing a nested module."""
        model = TestModel()
        new_activation = nn.GELU()
        set_module(model, 'encoder.1', new_activation)
        
        assert isinstance(model.encoder[1], nn.GELU)
        retrieved = get_module(model, 'encoder.1')
        assert isinstance(retrieved, nn.GELU)
    
    def test_set_entire_sequential(self):
        """Test replacing an entire Sequential module."""
        model = TestModel()
        new_encoder = nn.Sequential(
            nn.Linear(10, 15),
            nn.Tanh(),
            nn.Linear(15, 30)
        )
        set_module(model, 'encoder', new_encoder)
        
        assert len(model.encoder) == 3
        assert isinstance(model.encoder[1], nn.Tanh)
        retrieved = get_module(model, 'encoder')
        assert retrieved is new_encoder
    
    def test_set_in_module_dict(self):
        """Test replacing a module in ModuleDict."""
        model = TestModel()
        new_layer = nn.Linear(5, 15)
        set_module(model, 'nested.layer1', new_layer)
        
        assert model.nested['layer1'].out_features == 15
        retrieved = get_module(model, 'nested.layer1')
        assert retrieved is new_layer
    
    def test_set_and_get_consistency(self):
        """Test that set_module and get_module are consistent."""
        model = TestModel()
        new_layer = nn.Linear(20, 25)
        set_module(model, 'encoder.2', new_layer)
        
        retrieved = get_module(model, 'encoder.2')
        assert retrieved is new_layer
        assert retrieved.out_features == 25


class TestAliases:
    """Test that get_attr and set_attr aliases work correctly."""
    
    def test_get_attr_alias(self):
        """Test that get_attr is an alias for get_module."""
        model = TestModel()
        result1 = get_module(model, 'encoder')
        result2 = get_attr(model, 'encoder')
        assert result1 is result2
    
    def test_set_attr_alias(self):
        """Test that set_attr is an alias for set_module."""
        model = TestModel()
        new_decoder = nn.Linear(30, 8)
        set_attr(model, 'decoder', new_decoder)
        
        assert model.decoder.out_features == 8
        retrieved = get_attr(model, 'decoder')
        assert retrieved is new_decoder


if __name__ == '__main__':
    # Run tests manually if pytest is not available
    import sys
    
    print("Running get_module tests...")
    test_get = TestGetModule()
    test_get.test_get_top_level_module()
    test_get.test_get_nested_module()
    test_get.test_get_deeply_nested_module()
    test_get.test_get_nonexistent_module()
    test_get.test_get_from_module_dict()
    test_get.test_get_nonexistent_nested()
    print("✓ All get_module tests passed!")
    
    print("\nRunning set_module tests...")
    test_set = TestSetModule()
    test_set.test_set_top_level_module()
    test_set.test_set_nested_module()
    test_set.test_set_entire_sequential()
    test_set.test_set_in_module_dict()
    test_set.test_set_and_get_consistency()
    print("✓ All set_module tests passed!")
    
    print("\nRunning alias tests...")
    test_aliases = TestAliases()
    test_aliases.test_get_attr_alias()
    test_aliases.test_set_attr_alias()
    print("✓ All alias tests passed!")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
