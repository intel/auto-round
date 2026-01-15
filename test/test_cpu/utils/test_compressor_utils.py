"""
Unit tests for auto_round.compressors.utils module.
"""
import torch
import pytest

from auto_round.compressors.utils import get_fp_layer_names
from auto_round.utils import INNER_SUPPORTED_LAYER_TYPES


class TestGetFpLayerNames:
    """Test suite for get_fp_layer_names function."""

    def test_regular_linear_layers(self):
        """Test with regular Linear layers."""
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 10)
                self.layer2 = torch.nn.Linear(10, 10)
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(10, 10),
                    torch.nn.Linear(10, 10)
                )
        
        model = MockModel()
        
        # Test finding specific layer
        result = get_fp_layer_names(model, 'layer1')
        assert 'layer1' in result, "Should find layer1"
        
        # Test finding layers with pattern
        result = get_fp_layer_names(model, 'mlp')
        assert len(result) == 2, "Should find 2 layers in mlp"
        assert 'mlp.0' in result and 'mlp.1' in result
        
    def test_fp8linear_layers(self):
        """Test with FP8Linear layers (mocked by creating a proper class)."""
        # Create a proper mock FP8Linear class
        class FP8Linear(torch.nn.Linear):
            """Mock FP8Linear class for testing."""
            def __init__(self, in_features, out_features):
                super().__init__(in_features, out_features)
        
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 10)
                # Use proper FP8Linear mock
                self.layer2 = FP8Linear(10, 10)
                
                self.mlp = torch.nn.Sequential()
                linear1 = torch.nn.Linear(10, 10)
                self.mlp.add_module('0', linear1)
                linear2 = FP8Linear(10, 10)
                self.mlp.add_module('1', linear2)
        
        model = MockModel()
        
        # Test finding FP8Linear layer
        result = get_fp_layer_names(model, 'layer2')
        assert 'layer2' in result, "Should find FP8Linear layer (layer2)"
        
        # Test finding mixed Linear and FP8Linear in mlp
        result = get_fp_layer_names(model, 'mlp')
        assert len(result) == 2, "Should find 2 layers in mlp (both Linear and FP8Linear)"
        assert 'mlp.0' in result, "Should find regular Linear in mlp"
        assert 'mlp.1' in result, "Should find FP8Linear in mlp"
        
    def test_empty_ignore_layers(self):
        """Test with empty ignore_layers string."""
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 10)
        
        model = MockModel()
        result = get_fp_layer_names(model, '')
        assert len(result) == 0, "Empty ignore_layers should return empty list"
        
    def test_multiple_ignore_patterns(self):
        """Test with multiple ignore patterns."""
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 10)
                self.layer2 = torch.nn.Linear(10, 10)
                self.layer3 = torch.nn.Linear(10, 10)
        
        model = MockModel()
        result = get_fp_layer_names(model, 'layer1,layer3')
        assert 'layer1' in result, "Should find layer1"
        assert 'layer3' in result, "Should find layer3"
        assert 'layer2' not in result, "Should not find layer2"
        
    def test_pattern_with_digits(self):
        """Test pattern matching with digits (special case in the code)."""
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([
                    torch.nn.Linear(10, 10),
                    torch.nn.Linear(10, 10),
                    torch.nn.Linear(10, 10)
                ])
        
        model = MockModel()
        # Pattern ending with digit should get a dot appended for matching
        result = get_fp_layer_names(model, 'layers.0')
        # Should match 'layers.0'
        assert 'layers.0' in result, "Should match layers.0"
