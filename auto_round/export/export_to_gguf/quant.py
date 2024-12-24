import torch

def q4_0_qdq(tensor):
    tensor = tensor.to(torch.float32)

    orig_shape = tensor.shape
    tensor = tensor.view(-1, 32)

    max_vals = torch.max(torch.abs(tensor), dim=1)[0]
    max_vals[max_vals == 0] = 1.0
    
    d = max_vals / -8.0
    ids = 1.0 / d
    
    scaled_tensors = tensor * ids[:, None]
    quantized_tensors = torch.clamp(scaled_tensors + 8.5, 0, 15).to(torch.uint8)

    # dequant
    dequant_tensors = (quantized_tensors.view(orig_shape)).to(torch.float16)
    return dequant_tensors

def q4_1_qdq(tensor):
    tensor = tensor.to(torch.float32)

    orig_shape = tensor.shape
    tensor = tensor.view(-1, 32)

    min_vals = torch.min(tensor, dim=1)[0]
    max_vals = torch.max(tensor, dim=1)[0]

    d = (max_vals - min_vals) / (2**4-1)
    d[d == 0] = 1.0
    ids = 1.0 / d
    
    quantized_tensors = (tensor - min_vals[:, None]) * ids[:,None]

    quantized_tensors = torch.clamp(quantized_tensors + 0.5, 0, 15).to(torch.uint8)

    dequant_tensors = (quantized_tensors.float() * d[:, None]) + min_vals[:,None]
    dequant_tensors = dequant_tensors.view(orig_shape).to(torch.float16)

    return dequant_tensors
    