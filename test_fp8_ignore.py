import torch
import torch.nn as nn

# -----------------------------
# Mock FP8 Linear layer
# -----------------------------
class FP8Linear(nn.Linear):
    """Simulates a FP8-native linear layer"""
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        # pretend this layer is FP8
        self.fp8 = True

# -----------------------------
# Mock model
# -----------------------------
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.attn1 = FP8Linear(10, 10)  # should be auto-detected
        self.mlp1 = FP8Linear(10, 10)   # should be ignored via ignore_layers
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.attn1(x)
        x = self.mlp1(x)
        x = self.fc2(x)
        return x

# -----------------------------
# Mock helper for FP8 detection
# -----------------------------
def is_fp8_linear(layer):
    return hasattr(layer, "fp8") and layer.fp8

# -----------------------------
# Your revised get_fp_layer_names function
# -----------------------------
def get_fp_layer_names(model: nn.Module, ignore_layers: str):
    not_to_quantized_layers = []

    # Auto-detect FP8 layers
    for n, m in model.named_modules():
        if is_fp8_linear(m):
            not_to_quantized_layers.append(n)
            print(f"Auto-detected FP8 layer to ignore: {n}")

    # this Processes user-specified ignore_layers
    if ignore_layers:
        ignore_list = ignore_layers.replace(" ", "").split(",")
        for fp_layer in ignore_list:
            if not fp_layer:
                continue
            # matching any layer whose name has the pattern
            for n, _ in model.named_modules():  # match by name only
                match_name = fp_layer
                if fp_layer[-1].isdigit():
                    match_name += "."
                if match_name in n:
                    if n not in not_to_quantized_layers: # avoiding duplicates
                        not_to_quantized_layers.append(n)
                        print(f"User-specified ignore layer matched: {n}")

    print(f"Final not_to_quantized_layers: {not_to_quantized_layers}")
    return not_to_quantized_layers

# -----------------------------
# Test the function
# -----------------------------
model = MockModel()
ignored_layers = get_fp_layer_names(model, ignore_layers="mlp1")

# Expected output:
# - attn1 (auto-detected FP8)
# - mlp1 (ignored by user)
