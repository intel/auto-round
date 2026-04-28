# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
import torch
import sys


class ARK_DT:
    float64 = 64
    float32 = 32
    float16 = 16
    bfloat16 = 65552
    int2 = 258
    int3 = 259
    int4 = 260
    int5 = 261
    int6 = 262
    int7 = 263
    int8 = 264
    int32 = 288
    float8_e4m3 = 8
    float8_e5m2 = 65544
    float8_e8m0 = 196616
    undef = 0


def cvt_dtype(dtype):
    if dtype == torch.float32:
        return ARK_DT.float32
    if dtype == torch.float16:
        return ARK_DT.float16
    if dtype == torch.bfloat16:
        return ARK_DT.bfloat16
    if dtype == torch.float8_e4m3fn:
        return ARK_DT.float8_e4m3
    if dtype == torch.float8_e5m2:
        return ARK_DT.float8_e5m2
    if dtype == torch.int8:
        return ARK_DT.int8
    if dtype == torch.int32:
        return ARK_DT.int32
    return ARK_DT.undef


def cvtstr_dtype(dtype):
    if dtype == "fp32":
        return ARK_DT.float32
    if dtype == "fp16":
        return ARK_DT.float16
    if dtype == "bf16":
        return ARK_DT.bfloat16
    if dtype == "fp8_e4m3":
        return ARK_DT.float8_e4m3
    if dtype == "fp8_e5m2":
        return ARK_DT.float8_e5m2
    if dtype == "fp8_e8m0":
        return ARK_DT.float8_e8m0
    if dtype == "int8":
        return ARK_DT.int8
    if dtype == "int4":
        return ARK_DT.int4
    if dtype == "int2":
        return ARK_DT.int2
    if dtype == "int3":
        return ARK_DT.int3
    if dtype == "int5":
        return ARK_DT.int5
    if dtype == "int6":
        return ARK_DT.int6
    if dtype == "int7":
        return ARK_DT.int7
    if dtype == "int32":
        return ARK_DT.int32
    return ARK_DT.undef


def get_stream(A: torch.Tensor) -> int:
    if A.device.type == "cpu":
        return 0
    if A.device.type == "xpu":
        return torch.xpu.current_stream().sycl_queue


def singleton(cls):
    """
    一个简单的单例模式装饰器
    """
    instances = {}  # 存储类与实例的映射关系

    def get_instance(*args, **kwargs):
        # 如果类不在字典中，则创建一个并存入
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        # 如果已经在字典中，直接返回之前创建的那个
        return instances[cls]

    return get_instance


@singleton
class ARK:
    cpu_lib = None
    xpu_lib = None

    def __init__(self):
        try:
            from . import auto_round_kernel_cpu

            self.cpu_lib = auto_round_kernel_cpu
        except ImportError as e:
            print(f"ARK is unable to load CPU lib: {e}")
            self.cpu_lib = None

        if torch.xpu.is_available():
            try:
                from . import auto_round_kernel_xpu

                self.xpu_lib = auto_round_kernel_xpu
            except ImportError as e:
                print(f"ARK is unable to load XPU lib: {e}")
                self.xpu_lib = None

    def get_lib(self, A: torch.Tensor):
        lib = None
        if A.device.type == "xpu":
            lib = self.xpu_lib
        if A.device.type == "cpu":
            lib = self.cpu_lib
        if lib is None:
            raise NotImplementedError(f"Current device {A.device} is not supported")
        return lib

    # A: mxk,  B: nxk, bias: n
    def matmul(self, A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
        m = A.shape[0]
        n = B.shape[0]
        k = B.shape[1]
        lib = self.get_lib(A)
        ctype = A.dtype
        if A.device.type == "cpu":
            ctype = torch.float32
        C = torch.zeros(m, n, dtype=ctype, device=A.device)
        stream = get_stream(A)
        lib.matmul(
            stream,
            m,
            n,
            k,
            A.contiguous().data_ptr(),
            cvt_dtype(A.dtype),
            B.contiguous().data_ptr(),
            cvt_dtype(B.dtype),
            C.contiguous().data_ptr(),
            cvt_dtype(C.dtype),
            bias.to(C.dtype).contiguous().data_ptr(),
            True,
        )
        return C

    # A: mxk:s8,  B: nxk:s8, return: mxn:s32
    def igemm_s8s8s32(self, A: torch.Tensor, B: torch.Tensor):
        m = A.shape[0]
        n = B.shape[0]
        k = B.shape[1]
        lib = self.get_lib(A)
        if lib is None:
            raise NotImplementedError(f"Current device {A.device} is not supported")
        C = torch.zeros(m, n, dtype=torch.int32, device=A.device)
        stream = get_stream(A)
        lib.matmul(
            stream,
            m,
            n,
            k,
            A.contiguous().data_ptr(),
            cvt_dtype(A.dtype),
            B.contiguous().data_ptr(),
            cvt_dtype(B.dtype),
            C.contiguous().data_ptr(),
            cvt_dtype(C.dtype),
            0,
            True,
        )
        return C

    # A: mxk:DT,  B: nxk:s8, scaleB: n:DT
    # return: mxn:DT
    def woqgemm_s8(self, A: torch.Tensor, B: torch.Tensor, scaleB: torch.Tensor, bias: torch.Tensor):
        m = A.shape[0]
        n = B.shape[0]
        k = B.shape[1]
        lib = self.get_lib(A)

        C = torch.zeros(m, n, dtype=A.dtype, device=A.device)
        stream = get_stream(A)
        lib.woqgemm_s8(
            stream,
            m,
            n,
            k,
            A.contiguous().data_ptr(),
            cvt_dtype(A.dtype),
            B.contiguous().data_ptr(),
            C.contiguous().data_ptr(),
            bias.contiguous().data_ptr(),
            True,
            scaleB.contiguous().data_ptr(),
        )
        return C

    # A: mxk:DT,  B: BS:s8, bias: n:DT
    # return: C: mxn:DT
    def woqgemm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        bias: torch.Tensor,
        n,
        k,
        groupsize,
        compute_type,
        weight_type,
        scale_type,
        asym,
    ):
        m = A.shape[0]
        lib = self.get_lib(A)
        ct = cvtstr_dtype(compute_type)
        wt = cvtstr_dtype(weight_type)
        st = cvtstr_dtype(scale_type)
        C = torch.zeros(m, n, dtype=A.dtype, device=A.device)
        stream = get_stream(A)
        lib.woqgemm(
            stream,
            m,
            n,
            k,
            A.contiguous().data_ptr(),
            cvt_dtype(A.dtype),
            B.contiguous().data_ptr(),
            C.contiguous().data_ptr(),
            bias.contiguous().data_ptr(),
            groupsize,
            ct,
            wt,
            st,
            asym,
        )
        return C

    # QB: k*n:int8,  scaleB: k/blocksize*n:DT
    # return: blob:BS:int8
    def repack_quantized_weight(
        self,
        QB: torch.Tensor,
        scaleB: torch.Tensor,
        zp: torch.Tensor,
        groupsize,
        compute_type,
        weight_type,
        scale_type,
        asym,
    ):
        k = QB.shape[0]
        n = QB.shape[1]
        lib = self.get_lib(QB)
        stream = get_stream(QB)
        ct = cvtstr_dtype(compute_type)
        wt = cvtstr_dtype(weight_type)
        st = cvtstr_dtype(scale_type)
        BS = lib.packed_weight_size(stream, n, k, groupsize, ct, wt, st, asym)
        blob = torch.zeros(BS, dtype=torch.int8, device=QB.device)
        lib.repack_quantized_weight(
            stream,
            QB.contiguous().data_ptr(),
            zp.contiguous().data_ptr(),
            scaleB.contiguous().data_ptr(),
            blob.data_ptr(),
            n,
            k,
            groupsize,
            ct,
            wt,
            st,
            asym,
        )
        return blob

    # QB: blob:BS:int8
    # return: out:nxk:out_dtype
    def unpack_weight(
        self,
        blob: torch.Tensor,
        out_dtype: torch.dtype,
        n,
        k,
        groupsize,
        compute_type,
        weight_type,
        scale_type,
        asym,
    ):
        lib = self.get_lib(blob)
        stream = get_stream(blob)
        ct = cvtstr_dtype(compute_type)
        wt = cvtstr_dtype(weight_type)

        st = cvtstr_dtype(scale_type)
        oshape = (n, k) if blob.device.type == "xpu" else (k, n)
        out = torch.zeros(oshape, dtype=out_dtype, device=blob.device)
        lib.unpack_weight(
            stream,
            blob.data_ptr(),
            out.data_ptr(),
            cvt_dtype(out_dtype),
            n,
            k,
            groupsize,
            ct,
            wt,
            st,
            asym,
        )
        if blob.device.type == "cpu":
            return out.T
        return out

    def sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Scaled dot-product attention (SDPA) prefill+decode.

        Expects contiguous layouts:
        - query: [B, Hq, Sq, D]
        - key: [B, Hkv, Skv, D]
        - value: [B, Hkv, Skv, D]

        Args:
        - scale: Softmax scale. Uses 1 / sqrt(D) when None.

        Returns:
        - O: [B, Hq, Sq, D] (same dtype as value)
        """
        if query.device.type != "xpu":
            raise NotImplementedError("sdpa is only supported on XPU")

        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"Q must be float16 or bfloat16, got {query.dtype}")
        if key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError(
                f"K/V dtype must match Q dtype, got K={key.dtype}, V={value.dtype}, Q={query.dtype}"
            )

        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("Q/K/V must be 4D tensors")

        if not query.is_contiguous() or not key.is_contiguous() or not value.is_contiguous():
            raise ValueError("Q/K/V must be contiguous")

        B, Hq, Sq, D = query.shape
        Bk, Hkv, Skv, Dk = key.shape
        Bv, Hkv2, Skv2, Dv = value.shape

        if Bk != B or Bv != B:
            raise ValueError("Batch size mismatch between Q/K/V")
        if Hkv2 != Hkv or Skv2 != Skv or Dv != Dk:
            raise ValueError("K/V shape mismatch")
        if Dk != D:
            raise ValueError("Head dim mismatch between Q and K/V")
        if D not in (64, 128, 96, 192):
            raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128, 96, 192")

        if dropout_p != 0.0:
            raise NotImplementedError(f"dropout_p must be 0.0 (got {dropout_p}); dropout is not supported")

        if attn_mask is not None:
            if attn_mask.device.type != "xpu":
                raise ValueError("attn_mask must be on XPU")
            if not attn_mask.is_contiguous():
                raise ValueError("attn_mask must be contiguous")
            if attn_mask.dtype != torch.float32:
                raise ValueError(f"attn_mask must be float32 (additive bias), got {attn_mask.dtype}")
            expected_mask_shape = (B, 1, Sq, Skv)
            if attn_mask.shape != expected_mask_shape:
                raise ValueError(
                    f"attn_mask shape must be {expected_mask_shape}, got {tuple(attn_mask.shape)}"
                )

        lib = self.get_lib(query)
        stream = get_stream(query)
        O = torch.empty((B, Hq, Sq, D), device=query.device, dtype=value.dtype)
        lib.sdpa(
            stream,
            query.data_ptr(),
            key.data_ptr(),
            value.data_ptr(),
            O.data_ptr(),
            attn_mask.data_ptr() if attn_mask is not None else 0,
            cvt_dtype(query.dtype),
            cvt_dtype(key.dtype),
            cvt_dtype(O.dtype),
            B,
            Hq,
            Hkv,
            Sq,
            Skv,
            D,
            float(scale) if scale is not None else 1.0 / (D ** 0.5),
            bool(is_causal),
        )
        return O
    
    def sage(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        quant_block_size: int = 64,
        qscale: torch.Tensor = None,
        kscale: torch.Tensor = None,
    ) -> torch.Tensor:
        """SAGE attention prefill+decode.

        Expects contiguous layouts:
        - query: [B, Hq, Sq, D]
        - key: [B, Hkv, Skv, D]
        - value: [B, Hkv, Skv, D]

        Args:
        - scale: Attention scale. Uses 1.0 when None.
        - scale_block_size: Block size for qscale and kscale.

        Returns:
        - O: [B, Hq, Sq, D] (same dtype as value)
        """
        if query.device.type != "xpu":
            raise NotImplementedError("sdpa is only supported on XPU")

        # if query.dtype not in (torch.float16, torch.bfloat16):
        #     raise ValueError(f"Q must be float16 or bfloat16, got {query.dtype}")
        # if key.dtype != query.dtype or value.dtype != query.dtype:
        #     raise ValueError(f"K/V dtype must match Q dtype, got K={key.dtype}, V={value.dtype}, Q={query.dtype}")

        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("Q/K/V must be 4D tensors")

        if not query.is_contiguous() or not key.is_contiguous() or not value.is_contiguous():
            raise ValueError("Q/K/V must be contiguous")

        B, Hq, Sq, D = query.shape
        Bk, Hkv, Skv, Dk = key.shape
        Bv, Hkv2, Skv2, Dv = value.shape

        if Bk != B or Bv != B:
            raise ValueError("Batch size mismatch between Q/K/V")
        if Hkv2 != Hkv or Skv2 != Skv or Dv != Dk:
            raise ValueError("K/V shape mismatch")
        if Dk != D:
            raise ValueError("Head dim mismatch between Q and K/V")
        if D not in (64, 128):
            raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")

        lib = self.get_lib(query)
        stream = get_stream(query)
        O = torch.empty((B, Hq, Sq, D), device=query.device, dtype=value.dtype)
        lib.sage(
            stream,
            query.data_ptr(),
            key.data_ptr(),
            value.data_ptr(),
            O.data_ptr(),
            attn_mask.data_ptr() if attn_mask is not None else 0,
            quant_block_size,
            qscale.data_ptr() if qscale is not None else 0,
            kscale.data_ptr() if kscale is not None else 0,
            cvt_dtype(query.dtype),
            cvt_dtype(key.dtype),
            cvt_dtype(O.dtype),
            B,
            Hq,
            Hkv,
            Sq,
            Skv,
            D,
            float(scale) if scale is not None else 1.0,
            bool(is_causal),
        )
        return O
    
    def sagev1(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        quant_block_size: int = 64,
    ) -> torch.Tensor:
        """SAGE v1 attention prefill+decode.

        Expects contiguous layouts:
        - query: [B, Hq, Sq, D]
        - key: [B, Hkv, Skv, D]
        - value: [B, Hkv, Skv, D]

        Args:
        - scale: Attention scale. Uses 1.0 when None.
        - quant_block_size: Quantization block size used by the kernel.

        Returns:
        - O: [B, Hq, Sq, D] (same dtype as value)
        """
        if quant_block_size <= 0:
            return self.sdpa(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        if query.device.type != "xpu":
            raise NotImplementedError("sdpa is only supported on XPU")
        if query.dtype not in (torch.float16,):
            raise ValueError(f"Q must be float16, got {query.dtype}")
        if key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError(f"K/V dtype must match Q dtype, got K={key.dtype}, V={value.dtype}, Q={query.dtype}")

        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("Q/K/V must be 4D tensors")

        if not query.is_contiguous() or not key.is_contiguous() or not value.is_contiguous():
            raise ValueError("Q/K/V must be contiguous")

        B, Hq, Sq, D = query.shape
        Bk, Hkv, Skv, Dk = key.shape
        Bv, Hkv2, Skv2, Dv = value.shape

        if Bk != B or Bv != B:
            raise ValueError("Batch size mismatch between Q/K/V")
        if Hkv2 != Hkv or Skv2 != Skv or Dv != Dk:
            raise ValueError("K/V shape mismatch")
        if Dk != D:
            raise ValueError("Head dim mismatch between Q and K/V")
        if D not in (64, 128):
            raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")

        lib = self.get_lib(query)
        stream = get_stream(query)
        O = torch.empty((B, Hq, Sq, D), device=query.device, dtype=value.dtype)
        lib.sagev1(
            stream,
            query.data_ptr(),
            key.data_ptr(),
            value.data_ptr(),
            O.data_ptr(),
            attn_mask.data_ptr() if attn_mask is not None else 0,
            quant_block_size,
            cvt_dtype(query.dtype),
            cvt_dtype(key.dtype),
            cvt_dtype(value.dtype),
            cvt_dtype(O.dtype),
            B,
            Hq,
            Hkv,
            Sq,
            Skv,
            D,
            float(scale) if scale is not None else 1.0,
            bool(is_causal),
        )
        return O

    def sage_dynquant(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        quant_block_size: int = 64,
    ) -> torch.Tensor:
        """SAGE Attention with dynamic INT8 block-wise quantization of Q/K.

        Takes FP16 Q, K, V inputs. Quantizes Q and K to INT8 per-block
        using a fused SYCL kernel, then calls SAGE V1 with INT8 data.
        API is like SDPA but with an extra quant_block_size parameter.

        Args:
            query: [B, Hq, Sq, D] float16
            key:   [B, Hkv, Skv, D] float16
            value: [B, Hkv, Skv, D] float16
            quant_block_size: Number of tokens sharing one INT8 scale.
                E.g. 64 means 64 consecutive tokens share one absmax.
                0 means per-token (block_size=1).

        Returns:
            O: [B, Hq, Sq, D] float16
        """
        if query.device.type != "xpu":
            raise NotImplementedError("sage_dynquant is only supported on XPU")

        if query.dtype not in (torch.float16,):
            raise ValueError(f"Q must be float16, got {query.dtype}")

        B, Hq, Sq, D = query.shape
        _, Hkv, Skv, _ = key.shape

        # block_size=0 means per-token
        block_size = quant_block_size if quant_block_size > 0 else 1

        # SAGE V1 kernel uses K-tile size=32; scale_block_size must be 1 or >=32
        if block_size != 1 and block_size < 32:
            raise ValueError(
                f"quant_block_size={block_size} is not supported. "
                f"Must be 1 (per-token) or >= 32 (e.g. 32, 64, 128, 256)."
            )

        lib = self.get_lib(query)
        stream = get_stream(query)

        # Auto-pad Q and K/V seq lengths to be divisible by block_size
        # so sage_dynquant works as a drop-in replacement for SDPA
        def _ceil_div(a, b):
            return (a + b - 1) // b

        Sq_pad = _ceil_div(Sq, block_size) * block_size
        Skv_pad = _ceil_div(Skv, block_size) * block_size
        need_pad_q = Sq_pad != Sq
        need_pad_kv = Skv_pad != Skv

        if need_pad_q:
            pad_q = Sq_pad - Sq
            query = torch.nn.functional.pad(query, (0, 0, 0, pad_q))  # pad S dim with zeros
        if need_pad_kv:
            pad_kv = Skv_pad - Skv
            key = torch.nn.functional.pad(key, (0, 0, 0, pad_kv))
            value = torch.nn.functional.pad(value, (0, 0, 0, pad_kv))

        # Fused block-wise quantization via SYCL kernel
        # Tensor layout: [B, H, S, D] is contiguous → [B*H*S, D] flattened
        # block_size tokens share one scale → num_blocks = B*H*S / block_size
        # For Q: num_rows = B*Hq*Sq_pad, scale shape = [B, Hq, Sq_pad/block_size, 1]
        q_num_rows = B * Hq * Sq_pad
        q_num_blocks = q_num_rows // block_size
        q_i8 = torch.empty_like(query, dtype=torch.int8)
        q_scale = torch.empty(q_num_blocks, dtype=torch.float32, device=query.device)
        lib.sage_dynamic_quant(
            stream,
            query.data_ptr(),
            q_i8.data_ptr(),
            q_scale.data_ptr(),
            q_num_rows,
            D,
            block_size,
        )
        q_scale = q_scale.reshape(B, Hq, Sq_pad // block_size, 1)

        k_num_rows = B * Hkv * Skv_pad
        k_num_blocks = k_num_rows // block_size
        k_i8 = torch.empty_like(key, dtype=torch.int8)
        k_scale = torch.empty(k_num_blocks, dtype=torch.float32, device=key.device)
        lib.sage_dynamic_quant(
            stream,
            key.data_ptr(),
            k_i8.data_ptr(),
            k_scale.data_ptr(),
            k_num_rows,
            D,
            block_size,
        )
        k_scale = k_scale.reshape(B, Hkv, Skv_pad // block_size, 1)

        # Call SAGE v1 with matching scale_block_size
        out = self.sage(
            q_i8, k_i8, value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
            scale_block_size=block_size,
            qscale=q_scale,
            kscale=k_scale,
        )

        # Slice back to original seq length if padded
        if need_pad_q:
            out = out[:, :, :Sq, :]
        return out

    def moe_gemm(
        self,
        activations: torch.Tensor,
        weights: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        *,
        scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MOE GEMM (Mixture of Experts Grouped GEMM).

        Computes grouped GEMM for MOE layers where different experts process
        different numbers of tokens.

        Expects contiguous layouts:
        - activations: [total_tokens, K] (BF16/FP16)
        - weights: [num_experts, K, N] (BF16/FP16, Row major)
        - num_tokens_per_expert: [num_experts] (int32)
        - scales (optional): [num_experts, N] or None

        Returns:
        - outputs: [total_tokens, N] (same dtype as activations)
        """
        if activations.device.type != "xpu":
            raise NotImplementedError("moe_gemm is only supported on XPU")

        if activations.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"activations must be fp16/bf16, got {activations.dtype}")
        if weights.dtype != activations.dtype:
            raise ValueError(f"weights dtype must match activations dtype")

        if activations.ndim != 2 or weights.ndim != 3:
            raise ValueError("activations must be 2D [total_tokens, K], weights must be 3D [num_experts, K, N]")

        if not activations.is_contiguous() or not weights.is_contiguous():
            raise ValueError("activations and weights must be contiguous")

        if num_tokens_per_expert.dtype != torch.int32:
            num_tokens_per_expert = num_tokens_per_expert.to(torch.int32)

        if not num_tokens_per_expert.is_contiguous():
            num_tokens_per_expert = num_tokens_per_expert.contiguous()

        total_tokens, K = activations.shape
        num_experts, K_w, N = weights.shape  # weights are [num_experts, K, N]

        if K != K_w:
            raise ValueError(f"K dimension mismatch: activations K={K}, weights K={K_w}")

        if num_tokens_per_expert.shape[0] != num_experts:
            raise ValueError(f"num_tokens_per_expert length {num_tokens_per_expert.shape[0]} != num_experts {num_experts}")

        # Validate total tokens
        expected_total = int(num_tokens_per_expert.sum().item())
        if expected_total != total_tokens:
            raise ValueError(f"Sum of num_tokens_per_expert ({expected_total}) != total_tokens ({total_tokens})")

        lib = self.get_lib(activations)
        stream = get_stream(activations)
        outputs = torch.empty((total_tokens, N), device=activations.device, dtype=activations.dtype)

        scales_ptr = scales.data_ptr() if scales is not None else 0

        lib.moe_gemm(
            stream,
            activations.data_ptr(),
            weights.data_ptr(),
            scales_ptr,
            outputs.data_ptr(),
            cvt_dtype(activations.dtype),
            N,
            K,
            num_tokens_per_expert.data_ptr(),
            num_experts,
        )
        return outputs


if __name__ == "__main__":
    ark = ARK()
    print(ark.cpu_lib is None, ark.xpu_lib is None)

    def matmul():
        m = n = k = 128
        dt = torch.int8
        device = "cpu"
        has_bias = False
        if dt == torch.int8:
            A = torch.randint(-128, 127, (m, k), dtype=dt, device=device)
            B = torch.randint(-128, 127, (n, k), dtype=dt, device=device)
            C = ark.igemm_s8s8s32(A, B)
            print(C)
        else:
            A = torch.rand(m, k, dtype=dt, device=device) - 0.5
            B = torch.rand(k, n, dtype=dt, device=device) - 0.5
            bias = torch.rand(1, n, dtype=dt, device=device) if has_bias else torch.empty(0)
            C = ark.matmul(A, B, bias)
        ref = torch.matmul(A, B.T)
        if has_bias:
            ref = ref + bias
        dff = abs(C - ref)
        if dt != torch.int8:
            print(dff.max(), dff.mean())
            print(torch.allclose(ref, C, 0.01, 0.1))

    def woq():
        m = n = k = 128
        dt = torch.float32
        device = "cpu"
        A = torch.rand(m, k, dtype=dt, device=device) - 0.5
        bias = torch.rand(1, n, dtype=dt, device=device) + 1000
        B = torch.randint(-128, 127, (n, k), dtype=torch.int8, device=device)
        scaleB = torch.rand(n, 1, dtype=dt, device=device)
        C = ark.woqgemm_s8(A, B, scaleB, bias)
        print(C)
        DB = B.to(dt) * scaleB
        ref = torch.matmul(A, DB.T) + bias
        print(ref)
        dff = abs(C - ref)
        print(dff.max(), dff.mean())

    def pack_unpack():
        m = n = k = 128
        groupsize = 32
        dt = torch.float32
        device = "xpu"
        B = torch.randint(-8, 7, (k, n), dtype=torch.int8, device=device)
        zp = torch.randint(-8, 7, (k // groupsize, n), dtype=torch.int8, device=device)
        scaleB = torch.rand(k // groupsize, n, dtype=dt, device=device) / 100
        blob = ark.repack_quantized_weight(B, scaleB, zp, groupsize, "fp32", "int4", "fp32", False)
        dq = ark.unpack_weight(blob, dt, n, k, groupsize, "fp32", "int4", "fp32", False)
        print(blob, dq)
        scale_re = scaleB.repeat_interleave(repeats=groupsize, dim=0).to(dt)

        DB = B.to(dt) * scale_re
        dff = abs(DB.T - dq)
        print(dff.max(), dff.mean())

    pack_unpack()


__all__ = ["ARK"]


# -----------------------------------------------------------------------------
# Compatibility layer
#
# Some callers (e.g. auto_round_extension/ark/qlinear.py) historically imported
# this package as a module and expected certain functions to exist at the module
# level (e.g. ark.repack_quantized_weight, ark.woq_linear).
#
# The current implementation exposes these as methods on the singleton ARK()
# instance. The wrappers below keep backward compatibility without changing the
# compiled extension.
# -----------------------------------------------------------------------------


def _ark_instance():
    return ARK()


def check_isa_supported(_isa: str) -> bool:
    # Best-effort: some builds expose ISA checks via native libs; keep safe.
    # Returning False is conservative and avoids misconfiguration.
    return False


def repack_quantized_weight(*args, **kwargs):
    """Repack quantized weights into ARK/BestLA packed format.

    Supports two call styles:

    1) New style (recommended):
       repack_quantized_weight(QB, scaleB, zp, groupsize, compute_type, weight_type, scale_type, asym)

    2) Legacy style used by qlinear.py:
       repack_quantized_weight(QB, scaleB, zp, g_idx, compute_type, weight_type, scale_type, asym, groupsize)
       repack_quantized_weight(QB, scaleB, zp, g_idx, weight_type, compute_type, scale_type, asym, groupsize)
       (g_idx is ignored)
    """

    if kwargs:
        return _ark_instance().repack_quantized_weight(**kwargs)

    if len(args) == 8:
        QB, scaleB, zp, groupsize, compute_type, weight_type, scale_type, asym = args
    elif len(args) == 9:
        QB, scaleB, zp, _g_idx, a4, a5, scale_type, asym, groupsize = args
        # Legacy call sites sometimes swap compute_type/weight_type.
        compute_types = {"fp16", "bf16", "fp32", "fp8_e4m3", "fp8_e5m2", "fp8_e8m0"}
        if isinstance(a4, str) and a4 in compute_types:
            compute_type, weight_type = a4, a5
        else:
            weight_type, compute_type = a4, a5
    else:
        raise TypeError(
            "repack_quantized_weight() expects 8 or 9 positional arguments; "
            f"got {len(args)}"
        )

    # Some native paths may still expect a valid zp pointer even when asym=False.
    if (zp is None) or (isinstance(zp, torch.Tensor) and zp.numel() == 0):
        if not bool(asym):
            k = QB.shape[0]
            n = QB.shape[1]
            zp = torch.zeros((k // int(groupsize), n), dtype=torch.int8, device=QB.device)
        else:
            zp = torch.empty(0, dtype=torch.int8, device=QB.device)

    return _ark_instance().repack_quantized_weight(
        QB,
        scaleB,
        zp,
        groupsize,
        compute_type,
        weight_type,
        scale_type,
        asym,
    )


def unpack_weight(blob: torch.Tensor, out_dtype: torch.dtype, n, k, groupsize, compute_type, weight_type, scale_type, asym):
    return _ark_instance().unpack_weight(blob, out_dtype, n, k, groupsize, compute_type, weight_type, scale_type, asym)


def packed_weight_size(A: torch.Tensor, n, k, groupsize, compute_type, weight_type, scale_type, asym):
    # Keep signature convenient for Python callers; native library needs a stream.
    lib = _ark_instance().get_lib(A)
    stream = get_stream(A)
    ct = cvtstr_dtype(compute_type)
    wt = cvtstr_dtype(weight_type)
    st = cvtstr_dtype(scale_type)
    return lib.packed_weight_size(stream, n, k, groupsize, ct, wt, st, asym)


def woq_linear(
    A: torch.Tensor,
    packed_B: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    compute_type,
    weight_type,
    scale_type,
    asym,
    groupsize=None,
):
    """Linear helper that writes into a preallocated output tensor."""

    if groupsize is None:
        groupsize = A.shape[-1]

    result = _ark_instance().woqgemm(
        A,
        packed_B,
        bias,
        out.shape[-1],
        A.shape[-1],
        int(groupsize),
        compute_type,
        weight_type,
        scale_type,
        bool(asym),
    )
    out.copy_(result)
    return out

