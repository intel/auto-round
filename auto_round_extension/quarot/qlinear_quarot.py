import torch
import quarot
# class Quantizer(torch.nn.Module):
#     def __init__(self, input_clip_ratio=1.0):
#         super().__init__()
#         self.input_clip_ratio = input_clip_ratio
#
#     def forward(self, x):
#         scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1) / 7).to(torch.float16) * self.input_clip_ratio
#         quantized_x = quarot.sym_quant(x, scales_x)
#         packed_tensor = quarot.PackedQuantizedTensor(quantized_x, scales_x)
#         return packed_tensor


class LinearW4A4Sym(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        '''
        Symmetric 4-bit Linear Layer.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scales',
                             torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', (torch.randint(1, 7, (self.out_features, self.in_features//2),
                                                      # SubByte weight
                                                      dtype=torch.uint8, requires_grad=False)))
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None

    def forward(self, x):
        # if torch.cuda.current_device() != x.device:
        #    torch.cuda.set_device(x.device)


        # scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1) / 7).to(torch.float16) * self.input_clip_ratio ##input_clip_ratio

        scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1) / 7).to(torch.float16)
        quantized_x = quarot.sym_quant(x, scales_x)
        x = quarot.PackedQuantizedTensor(quantized_x, scales_x)

        # assert type(x) == quarot.PackedQuantizedTensor  # Quantized input is given
        x, scales_x = x.quantized_x, x.scales_x

        # shape_handler = ShapeHandler(quantized_x)
        # quantized_x = shape_handler.flatten(quantized_x)
        x = quarot.matmul(x, self.weight)
        # out = shape_handler.unflatten(
        #    quarot.sym_dequant(int_result, scales_x, self.weight_scales))
        if self.bias is not None:
            return quarot.sym_dequant(x, scales_x, self.weight_scales) + self.bias
        else:
            return quarot.sym_dequant(x, scales_x, self.weight_scales)

    @staticmethod
    def from_float(module: torch.nn.Linear, weight_scales, ):
        '''
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        if hasattr(module,'orig_layer'):
            module =  module.orig_layer
        weight_matrix = module.weight.data

        int_module = LinearW4A4Sym(module.in_features, module.out_features, bias=module.bias is not None,
                                dtype=weight_matrix.dtype).to(weight_matrix.dtype)
        if weight_scales is not None:
            assert weight_scales.shape == (module.out_features, 1), 'weight_scales should have shape (out_features, 1)'
            weight_matrix = weight_matrix.cuda()
            int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
            int_rounded_weight = (weight_matrix / weight_scales.cuda()).round()
            int_module.weight.copy_(quarot.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())

            if module.bias is not None:
                int_module.bias.copy_(module.bias)

        return int_module