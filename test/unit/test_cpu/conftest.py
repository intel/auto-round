"""CPU test-specific fixtures.

torch.compile behavior is controlled in test/conftest.py:
- default: disabled for all tests via a no-op patch
- opt in: use @pytest.mark.enable_torch_compile
"""
