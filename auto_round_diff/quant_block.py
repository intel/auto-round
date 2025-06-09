# import logging
# from types import MethodType
# import torch as th
# from torch import einsum
# import torch.nn as nn
# from einops import rearrange, repeat

# from qdiff.quant_layer import QuantModule, UniformAffineQuantizer, UniformAffineFP8Quantizer, StraightThrough
# from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, ResBlock, TimestepBlock, checkpoint
# from ldm.modules.diffusionmodules.openaimodel import QKMatMul, SMVMatMul
# from ldm.modules.attention import BasicTransformerBlock
# from ldm.modules.attention import exists, default

# from ddim.models.diffusion import ResnetBlock, AttnBlock, nonlinearity


# logger = logging.getLogger(__name__)


# class BaseQuantBlock(nn.Module):
#     """
#     Base implementation of block structures for all networks.
#     """
#     def __init__(self, act_quant_params: dict = {}):
#         super().__init__()
#         self.use_weight_quant = False
#         self.use_act_quant = False
#         # initialize quantizer

#         self.act_quantizer = UniformAffineFP8Quantizer(**act_quant_params)
#         self.activation_function = StraightThrough()

#         self.ignore_reconstruction = False

#     def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
#         # setting weight quantization here does not affect actual forward pass
#         self.use_weight_quant = weight_quant
#         self.use_act_quant = act_quant
#         for m in self.modules():
#             if isinstance(m, QuantModule):
#                 m.set_quant_state(weight_quant, act_quant)


# class QuantResBlock(BaseQuantBlock, TimestepBlock):
#     def __init__(
#         self, res: ResBlock, act_quant_params: dict = {}):
#         super().__init__(act_quant_params)
#         self.channels = res.channels
#         self.emb_channels = res.emb_channels
#         self.dropout = res.dropout
#         self.out_channels = res.out_channels
#         self.use_conv = res.use_conv
#         self.use_checkpoint = res.use_checkpoint
#         self.use_scale_shift_norm = res.use_scale_shift_norm

#         self.in_layers = res.in_layers

#         self.updown = res.updown

#         self.h_upd = res.h_upd
#         self.x_upd = res.x_upd

#         self.emb_layers = res.emb_layers
#         self.out_layers = res.out_layers

#         self.skip_connection = res.skip_connection

#     def forward(self, x, emb=None, split=0):
#         """
#         Apply the block to a Tensor, conditioned on a timestep embedding.
#         :param x: an [N x C x ...] Tensor of features.
#         :param emb: an [N x emb_channels] Tensor of timestep embeddings.
#         :return: an [N x C x ...] Tensor of outputs.
#         """
#         if split != 0 and self.skip_connection.split == 0:
#             return checkpoint(
#                 self._forward, (x, emb, split), self.parameters(), self.use_checkpoint
#             )
#         return checkpoint(
#                 self._forward, (x, emb), self.parameters(), self.use_checkpoint
#             )  

#     def _forward(self, x, emb, split=0):
#         # print(f"x shape {x.shape} emb shape {emb.shape}")
#         if emb is None:
#             assert(len(x) == 2)
#             x, emb = x
#         assert x.shape[2] == x.shape[3]

#         if self.updown:
#             in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
#             h = in_rest(x)
#             h = self.h_upd(h)
#             x = self.x_upd(x)
#             h = in_conv(h)
#         else:
#             h = self.in_layers(x)
#         emb_out = self.emb_layers(emb).type(h.dtype)
#         while len(emb_out.shape) < len(h.shape):
#             emb_out = emb_out[..., None]
#         if self.use_scale_shift_norm:
#             out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
#             scale, shift = th.chunk(emb_out, 2, dim=1)
#             h = out_norm(h) * (1 + scale) + shift
#             h = out_rest(h)
#         else:
#             h = h + emb_out
#             h = self.out_layers(h)
#         if split != 0:
#             return self.skip_connection(x, split=split) + h
#         return self.skip_connection(x) + h


# class QuantQKMatMul(BaseQuantBlock):
#     def __init__(
#         self, act_quant_params: dict = {}):
#         super().__init__(act_quant_params)
#         self.scale = None
#         self.use_act_quant = False
#         self.act_quantizer_q = UniformAffineFP8Quantizer(**act_quant_params)
#         self.act_quantizer_k = UniformAffineFP8Quantizer(**act_quant_params)
        
#     def forward(self, q, k):
#         if self.use_act_quant:
#             quant_q = self.act_quantizer_q(q * self.scale)
#             quant_k = self.act_quantizer_k(k * self.scale)
#             weight = th.einsum(
#                 "bct,bcs->bts", quant_q, quant_k
#             ) 
#         else:
#             weight = th.einsum(
#                 "bct,bcs->bts", q * self.scale, k * self.scale
#             )
#         return weight

#     def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
#         self.use_act_quant = act_quant


# class QuantSMVMatMul(BaseQuantBlock):
#     def __init__(
#         self, act_quant_params: dict = {}, sm_abit=8):
#         super().__init__(act_quant_params)
#         self.use_act_quant = False
#         self.act_quantizer_v = UniformAffineFP8Quantizer(**act_quant_params)
#         act_quant_params_w = act_quant_params.copy()
#         act_quant_params_w['n_bits'] = sm_abit
#         act_quant_params_w['symmetric'] = False
#         act_quant_params_w['always_zero'] = True
#         self.act_quantizer_w = UniformAffineFP8Quantizer(**act_quant_params_w)
        
#     def forward(self, weight, v):
#         if self.use_act_quant:
#             a = th.einsum("bts,bcs->bct", self.act_quantizer_w(weight), self.act_quantizer_v(v))
#         else:
#             a = th.einsum("bts,bcs->bct", weight, v)
#         return a

#     def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
#         self.use_act_quant = act_quant


# class QuantAttentionBlock(BaseQuantBlock):
#     def __init__(
#         self, attn: AttentionBlock, act_quant_params: dict = {}):
#         super().__init__(act_quant_params)
#         self.channels = attn.channels
#         self.num_heads = attn.num_heads
#         self.use_checkpoint = attn.use_checkpoint
#         self.norm = attn.norm
#         self.qkv = attn.qkv
        
#         self.attention = attn.attention

#         self.proj_out = attn.proj_out

#     def forward(self, x):
#         return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
#         #return pt_checkpoint(self._forward, x)  # pytorch

#     def _forward(self, x):
#         b, c, *spatial = x.shape
#         x = x.reshape(b, c, -1)
#         qkv = self.qkv(self.norm(x))
#         h = self.attention(qkv)
#         h = self.proj_out(h)
#         return (x + h).reshape(b, c, *spatial)


# def cross_attn_forward(self, x, context=None, mask=None):
#     h = self.heads

#     q = self.to_q(x)
#     context = default(context, x)
#     k = self.to_k(context)
#     v = self.to_v(context)

#     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

#     if self.use_act_quant:
#         quant_q = self.act_quantizer_q(q)
#         quant_k = self.act_quantizer_k(k)
#         sim = einsum('b i d, b j d -> b i j', quant_q, quant_k) * self.scale
#     else:
#         sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

#     if exists(mask):
#         mask = rearrange(mask, 'b ... -> b (...)')
#         max_neg_value = -th.finfo(sim.dtype).max
#         mask = repeat(mask, 'b j -> (b h) () j', h=h)
#         sim.masked_fill_(~mask, max_neg_value)

#     # attention, what we cannot get enough of
#     attn = sim.softmax(dim=-1)

#     if self.use_act_quant:
#         out = einsum('b i j, b j d -> b i d', self.act_quantizer_w(attn), self.act_quantizer_v(v))
#     else:
#         out = einsum('b i j, b j d -> b i d', attn, v)
#     out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
#     return self.to_out(out)


# class QuantBasicTransformerBlock(BaseQuantBlock):
#     def __init__(
#         self, tran: BasicTransformerBlock, act_quant_params: dict = {}, 
#         sm_abit: int = 8):
#         super().__init__(act_quant_params)
#         self.attn1 = tran.attn1
#         self.ff = tran.ff
#         self.attn2 = tran.attn2
        
#         self.norm1 = tran.norm1
#         self.norm2 = tran.norm2
#         self.norm3 = tran.norm3
#         self.checkpoint = tran.checkpoint
#         # self.checkpoint = False

#         # logger.info(f"quant attn matmul")
#         self.attn1.act_quantizer_q = UniformAffineFP8Quantizer(**act_quant_params)
#         self.attn1.act_quantizer_k = UniformAffineFP8Quantizer(**act_quant_params)
#         self.attn1.act_quantizer_v = UniformAffineFP8Quantizer(**act_quant_params)

#         self.attn2.act_quantizer_q = UniformAffineFP8Quantizer(**act_quant_params)
#         self.attn2.act_quantizer_k = UniformAffineFP8Quantizer(**act_quant_params)
#         self.attn2.act_quantizer_v = UniformAffineFP8Quantizer(**act_quant_params)
        
#         act_quant_params_w = act_quant_params.copy()
#         act_quant_params_w['n_bits'] = sm_abit
#         act_quant_params_w['always_zero'] = True
#         self.attn1.act_quantizer_w = UniformAffineFP8Quantizer(**act_quant_params_w)
#         self.attn2.act_quantizer_w = UniformAffineFP8Quantizer(**act_quant_params_w)

#         self.attn1.forward = MethodType(cross_attn_forward, self.attn1)
#         self.attn2.forward = MethodType(cross_attn_forward, self.attn2)
#         self.attn1.use_act_quant = False
#         self.attn2.use_act_quant = False

#     def forward(self, x, context=None):
#         # print(f"x shape {x.shape} context shape {context.shape}")
#         return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

#     def _forward(self, x, context=None):
#         if context is None:
#             assert(len(x) == 2)
#             x, context = x

#         x = self.attn1(self.norm1(x)) + x
#         x = self.attn2(self.norm2(x), context=context) + x
#         x = self.ff(self.norm3(x)) + x
#         return x
    
#     def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
#         self.attn1.use_act_quant = act_quant
#         self.attn2.use_act_quant = act_quant

#         # setting weight quantization here does not affect actual forward pass
#         self.use_weight_quant = weight_quant
#         self.use_act_quant = act_quant
#         for m in self.modules():
#             if isinstance(m, QuantModule):
#                 m.set_quant_state(weight_quant, act_quant)


# # the two classes below are for DDIM CIFAR
# class QuantResnetBlock(BaseQuantBlock):
#     def __init__(
#         self, res: ResnetBlock, act_quant_params: dict = {}):
#         super().__init__(act_quant_params)
#         self.in_channels = res.in_channels
#         self.out_channels = res.out_channels
#         self.use_conv_shortcut = res.use_conv_shortcut

#         self.norm1 = res.norm1
#         self.conv1 = res.conv1
#         self.temb_proj = res.temb_proj
#         self.norm2 = res.norm2
#         self.dropout = res.dropout
#         self.conv2 = res.conv2
#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 self.conv_shortcut = res.conv_shortcut
#             else:
#                 self.nin_shortcut = res.nin_shortcut


#     def forward(self, x, temb=None, split=0):
#         if temb is None:
#             assert(len(x) == 2)
#             x, temb = x

#         h = x
#         h = self.norm1(h)
#         h = nonlinearity(h)
#         h = self.conv1(h)

#         h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

#         h = self.norm2(h)
#         h = nonlinearity(h)
#         h = self.dropout(h)
#         h = self.conv2(h)

#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 x = self.conv_shortcut(x)
#             else:
#                 x = self.nin_shortcut(x, split=split)
#         out = x + h
#         return out


# class QuantAttnBlock(BaseQuantBlock):
#     def __init__(
#         self, attn: AttnBlock, act_quant_params: dict = {}, sm_abit=8):
#         super().__init__(act_quant_params)
#         self.in_channels = attn.in_channels

#         self.norm = attn.norm
#         self.q = attn.q
#         self.k = attn.k
#         self.v = attn.v
#         self.proj_out = attn.proj_out

#         self.act_quantizer_q = UniformAffineFP8Quantizer(**act_quant_params)
#         self.act_quantizer_k = UniformAffineFP8Quantizer(**act_quant_params)
#         self.act_quantizer_v = UniformAffineFP8Quantizer(**act_quant_params)
        
#         act_quant_params_w = act_quant_params.copy()
#         act_quant_params_w['n_bits'] = sm_abit
#         self.act_quantizer_w = UniformAffineFP8Quantizer(**act_quant_params_w)


#     def forward(self, x):
#         h_ = x
#         h_ = self.norm(h_)
#         q = self.q(h_)
#         k = self.k(h_)
#         v = self.v(h_)

#         # compute attention
#         b, c, h, w = q.shape
#         q = q.reshape(b, c, h*w)
#         q = q.permute(0, 2, 1)   # b,hw,c
#         k = k.reshape(b, c, h*w)  # b,c,hw
#         if self.use_act_quant:
#             q = self.act_quantizer_q(q)
#             k = self.act_quantizer_k(k)
#         w_ = th.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
#         w_ = w_ * (int(c)**(-0.5))
#         w_ = nn.functional.softmax(w_, dim=2)

#         # attend to values
#         v = v.reshape(b, c, h*w)
#         w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
#         # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
#         if self.use_act_quant:
#             v = self.act_quantizer_v(v)
#             w_ = self.act_quantizer_w(w_)
#         h_ = th.bmm(v, w_)
#         h_ = h_.reshape(b, c, h, w)

#         h_ = self.proj_out(h_)
        
#         out = x + h_
#         return out


# def get_specials(quant_act=False):
#     specials = {
#         ResBlock: QuantResBlock,
#         BasicTransformerBlock: QuantBasicTransformerBlock,
#         ResnetBlock: QuantResnetBlock,
#         AttnBlock: QuantAttnBlock,
#     }
#     if quant_act:
#         specials[QKMatMul] = QuantQKMatMul
#         specials[SMVMatMul] = QuantSMVMatMul
#     else:
#         specials[AttentionBlock] = QuantAttentionBlock
#     return specials

# import logging
# from types import MethodType
# import torch as th
# from torch import einsum
# import torch.nn as nn
# from einops import rearrange, repeat

# from qdiff.quant_layer import QuantModule, UniformAffineFP8Quantizer, StraightThrough
# from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, ResBlock, TimestepBlock, checkpoint
# from ldm.modules.diffusionmodules.openaimodel import QKMatMul, SMVMatMul
# from ldm.modules.attention import BasicTransformerBlock
# from ldm.modules.attention import exists, default

# from ddim.models.diffusion import ResnetBlock, AttnBlock, nonlinearity


# logger = logging.getLogger(__name__)


# class BaseQuantBlock(nn.Module):
#     """
#     Base implementation of block structures for all networks.
#     """
#     def __init__(self, act_quant_params: dict = {}):
#         super().__init__()
#         self.use_weight_quant = False
#         self.use_act_quant = False
#         # initialize quantizer

#         self.act_quantizer = UniformAffineFP8Quantizer(**act_quant_params)
#         self.activation_function = StraightThrough()

#         self.ignore_reconstruction = False

#     def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
#         # setting weight quantization here does not affect actual forward pass
#         self.use_weight_quant = weight_quant
#         self.use_act_quant = act_quant
#         for m in self.modules():
#             if isinstance(m, QuantModule):
#                 m.set_quant_state(weight_quant, act_quant)


# class QuantResBlock(BaseQuantBlock, TimestepBlock):
#     def __init__(
#         self, res: ResBlock, act_quant_params: dict = {}):
#         super().__init__(act_quant_params)
#         self.channels = res.channels
#         self.emb_channels = res.emb_channels
#         self.dropout = res.dropout
#         self.out_channels = res.out_channels
#         self.use_conv = res.use_conv
#         self.use_checkpoint = res.use_checkpoint
#         self.use_scale_shift_norm = res.use_scale_shift_norm

#         self.in_layers = res.in_layers

#         self.updown = res.updown

#         self.h_upd = res.h_upd
#         self.x_upd = res.x_upd

#         self.emb_layers = res.emb_layers
#         self.out_layers = res.out_layers

#         self.skip_connection = res.skip_connection

#     def forward(self, x, emb=None, split=0):
#         """
#         Apply the block to a Tensor, conditioned on a timestep embedding.
#         :param x: an [N x C x ...] Tensor of features.
#         :param emb: an [N x emb_channels] Tensor of timestep embeddings.
#         :return: an [N x C x ...] Tensor of outputs.
#         """
#         if split != 0 and self.skip_connection.split == 0:
#             return checkpoint(
#                 self._forward, (x, emb, split), self.parameters(), self.use_checkpoint
#             )
#         return checkpoint(
#                 self._forward, (x, emb), self.parameters(), self.use_checkpoint
#             )  

#     def _forward(self, x, emb, split=0):
#         # print(f"x shape {x.shape} emb shape {emb.shape}")
#         if emb is None:
#             assert(len(x) == 2)
#             x, emb = x
#         assert x.shape[2] == x.shape[3]

#         if self.updown:
#             in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
#             h = in_rest(x)
#             h = self.h_upd(h)
#             x = self.x_upd(x)
#             h = in_conv(h)
#         else:
#             h = self.in_layers(x)
#         emb_out = self.emb_layers(emb).type(h.dtype)
#         while len(emb_out.shape) < len(h.shape):
#             emb_out = emb_out[..., None]
#         if self.use_scale_shift_norm:
#             out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
#             scale, shift = th.chunk(emb_out, 2, dim=1)
#             h = out_norm(h) * (1 + scale) + shift
#             h = out_rest(h)
#         else:
#             h = h + emb_out
#             h = self.out_layers(h)
#         if split != 0:
#             return self.skip_connection(x, split=split) + h
#         return self.skip_connection(x) + h


# class QuantQKMatMul(BaseQuantBlock):
#     def __init__(
#         self, act_quant_params: dict = {}):
#         super().__init__(act_quant_params)
#         self.scale = None
#         self.use_act_quant = False
#         self.act_quantizer_q = UniformAffineFP8Quantizer(**act_quant_params)
#         self.act_quantizer_k = UniformAffineFP8Quantizer(**act_quant_params)
        
#     def forward(self, q, k):
#         if self.use_act_quant:
#             quant_q = self.act_quantizer_q(q * self.scale)
#             quant_k = self.act_quantizer_k(k * self.scale)
#             weight = th.einsum(
#                 "bct,bcs->bts", quant_q, quant_k
#             ) 
#         else:
#             weight = th.einsum(
#                 "bct,bcs->bts", q * self.scale, k * self.scale
#             )
#         return weight

#     def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
#         self.use_act_quant = act_quant


# class QuantSMVMatMul(BaseQuantBlock):
#     def __init__(
#         self, act_quant_params: dict = {}, sm_abit=8):
#         super().__init__(act_quant_params)
#         self.use_act_quant = False
#         self.act_quantizer_v = UniformAffineFP8Quantizer(**act_quant_params)
#         act_quant_params_w = act_quant_params.copy()
#         act_quant_params_w['n_bits'] = sm_abit
#         act_quant_params_w['symmetric'] = False
#         act_quant_params_w['always_zero'] = True
#         self.act_quantizer_w = UniformAffineFP8Quantizer(**act_quant_params_w)
        
#     def forward(self, weight, v):
#         if self.use_act_quant:
#             a = th.einsum("bts,bcs->bct", self.act_quantizer_w(weight), self.act_quantizer_v(v))
#         else:
#             a = th.einsum("bts,bcs->bct", weight, v)
#         return a

#     def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
#         self.use_act_quant = act_quant


# class QuantAttentionBlock(BaseQuantBlock):
#     def __init__(
#         self, attn: AttentionBlock, act_quant_params: dict = {}):
#         super().__init__(act_quant_params)
#         self.channels = attn.channels
#         self.num_heads = attn.num_heads
#         self.use_checkpoint = attn.use_checkpoint
#         self.norm = attn.norm
#         self.qkv = attn.qkv
        
#         self.attention = attn.attention

#         self.proj_out = attn.proj_out

#     def forward(self, x):
#         return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
#         #return pt_checkpoint(self._forward, x)  # pytorch

#     def _forward(self, x):
#         b, c, *spatial = x.shape
#         x = x.reshape(b, c, -1)
#         qkv = self.qkv(self.norm(x))
#         h = self.attention(qkv)
#         h = self.proj_out(h)
#         return (x + h).reshape(b, c, *spatial)


# def cross_attn_forward(self, x, context=None, mask=None):
#     h = self.heads

#     q = self.to_q(x)
#     context = default(context, x)
#     k = self.to_k(context)
#     v = self.to_v(context)

#     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

#     if self.use_act_quant:
#         quant_q = self.act_quantizer_q(q)
#         quant_k = self.act_quantizer_k(k)
#         sim = einsum('b i d, b j d -> b i j', quant_q, quant_k) * self.scale
#     else:
#         sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

#     if exists(mask):
#         mask = rearrange(mask, 'b ... -> b (...)')
#         max_neg_value = -th.finfo(sim.dtype).max
#         mask = repeat(mask, 'b j -> (b h) () j', h=h)
#         sim.masked_fill_(~mask, max_neg_value)

#     # attention, what we cannot get enough of
#     attn = sim.softmax(dim=-1)

#     if self.use_act_quant:
#         out = einsum('b i j, b j d -> b i d', self.act_quantizer_w(attn), self.act_quantizer_v(v))
#     else:
#         out = einsum('b i j, b j d -> b i d', attn, v)
#     out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
#     return self.to_out(out)


# class QuantBasicTransformerBlock(BaseQuantBlock):
#     def __init__(
#         self, tran: BasicTransformerBlock, act_quant_params: dict = {}, 
#         sm_abit: int = 8):
#         super().__init__(act_quant_params)
#         self.attn1 = tran.attn1
#         self.ff = tran.ff
#         self.attn2 = tran.attn2
        
#         self.norm1 = tran.norm1
#         self.norm2 = tran.norm2
#         self.norm3 = tran.norm3
#         self.checkpoint = tran.checkpoint
#         # self.checkpoint = False

#         # logger.info(f"quant attn matmul")
#         self.attn1.act_quantizer_q = UniformAffineFP8Quantizer(**act_quant_params)
#         self.attn1.act_quantizer_k = UniformAffineFP8Quantizer(**act_quant_params)
#         self.attn1.act_quantizer_v = UniformAffineFP8Quantizer(**act_quant_params)

#         self.attn2.act_quantizer_q = UniformAffineFP8Quantizer(**act_quant_params)
#         self.attn2.act_quantizer_k = UniformAffineFP8Quantizer(**act_quant_params)
#         self.attn2.act_quantizer_v = UniformAffineFP8Quantizer(**act_quant_params)
        
#         act_quant_params_w = act_quant_params.copy()
#         act_quant_params_w['n_bits'] = sm_abit
#         act_quant_params_w['always_zero'] = True
#         self.attn1.act_quantizer_w = UniformAffineFP8Quantizer(**act_quant_params_w)
#         self.attn2.act_quantizer_w = UniformAffineFP8Quantizer(**act_quant_params_w)

#         self.attn1.forward = MethodType(cross_attn_forward, self.attn1)
#         self.attn2.forward = MethodType(cross_attn_forward, self.attn2)
#         self.attn1.use_act_quant = False
#         self.attn2.use_act_quant = False

#     def forward(self, x, context=None):
#         # print(f"x shape {x.shape} context shape {context.shape}")
#         return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

#     def _forward(self, x, context=None):
#         if context is None:
#             assert(len(x) == 2)
#             x, context = x

#         x = self.attn1(self.norm1(x)) + x
#         x = self.attn2(self.norm2(x), context=context) + x
#         x = self.ff(self.norm3(x)) + x
#         return x
    
#     def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
#         self.attn1.use_act_quant = act_quant
#         self.attn2.use_act_quant = act_quant

#         # setting weight quantization here does not affect actual forward pass
#         self.use_weight_quant = weight_quant
#         self.use_act_quant = act_quant
#         for m in self.modules():
#             if isinstance(m, QuantModule):
#                 m.set_quant_state(weight_quant, act_quant)


# # the two classes below are for DDIM CIFAR
# class QuantResnetBlock(BaseQuantBlock):
#     def __init__(
#         self, res: ResnetBlock, act_quant_params: dict = {}):
#         super().__init__(act_quant_params)
#         self.in_channels = res.in_channels
#         self.out_channels = res.out_channels
#         self.use_conv_shortcut = res.use_conv_shortcut

#         self.norm1 = res.norm1
#         self.conv1 = res.conv1
#         self.temb_proj = res.temb_proj
#         self.norm2 = res.norm2
#         self.dropout = res.dropout
#         self.conv2 = res.conv2
#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 self.conv_shortcut = res.conv_shortcut
#             else:
#                 self.nin_shortcut = res.nin_shortcut


#     def forward(self, x, temb=None, split=0):
#         if temb is None:
#             assert(len(x) == 2)
#             x, temb = x

#         h = x
#         h = self.norm1(h)
#         h = nonlinearity(h)
#         h = self.conv1(h)

#         h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

#         h = self.norm2(h)
#         h = nonlinearity(h)
#         h = self.dropout(h)
#         h = self.conv2(h)

#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 x = self.conv_shortcut(x)
#             else:
#                 x = self.nin_shortcut(x, split=split)
#         out = x + h
#         return out


# class QuantAttnBlock(BaseQuantBlock):
#     def __init__(
#         self, attn: AttnBlock, act_quant_params: dict = {}, sm_abit=8):
#         super().__init__(act_quant_params)
#         self.in_channels = attn.in_channels

#         self.norm = attn.norm
#         self.q = attn.q
#         self.k = attn.k
#         self.v = attn.v
#         self.proj_out = attn.proj_out

#         self.act_quantizer_q = UniformAffineFP8Quantizer(**act_quant_params)
#         self.act_quantizer_k = UniformAffineFP8Quantizer(**act_quant_params)
#         self.act_quantizer_v = UniformAffineFP8Quantizer(**act_quant_params)
        
#         act_quant_params_w = act_quant_params.copy()
#         act_quant_params_w['n_bits'] = sm_abit
#         self.act_quantizer_w = UniformAffineFP8Quantizer(**act_quant_params_w)


#     def forward(self, x):
#         h_ = x
#         h_ = self.norm(h_)
#         q = self.q(h_)
#         k = self.k(h_)
#         v = self.v(h_)

#         # compute attention
#         b, c, h, w = q.shape
#         q = q.reshape(b, c, h*w)
#         q = q.permute(0, 2, 1)   # b,hw,c
#         k = k.reshape(b, c, h*w)  # b,c,hw
#         if self.use_act_quant:
#             q = self.act_quantizer_q(q)
#             k = self.act_quantizer_k(k)
#         w_ = th.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
#         w_ = w_ * (int(c)**(-0.5))
#         w_ = nn.functional.softmax(w_, dim=2)

#         # attend to values
#         v = v.reshape(b, c, h*w)
#         w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
#         # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
#         if self.use_act_quant:
#             v = self.act_quantizer_v(v)
#             w_ = self.act_quantizer_w(w_)
#         h_ = th.bmm(v, w_)
#         h_ = h_.reshape(b, c, h, w)

#         h_ = self.proj_out(h_)
        
#         out = x + h_
#         return out


# def get_specials(quant_act=False):
#     specials = {
#         ResBlock: QuantResBlock,
#         BasicTransformerBlock: QuantBasicTransformerBlock,
#         ResnetBlock: QuantResnetBlock,
#         AttnBlock: QuantAttnBlock,
#     }
#     if quant_act:
#         specials[QKMatMul] = QuantQKMatMul
#         specials[SMVMatMul] = QuantSMVMatMul
#     else:
#         specials[AttentionBlock] = QuantAttentionBlock
#     return specials

import logging
from types import MethodType
import torch as th
from torch import einsum
import torch.nn as nn
from einops import rearrange, repeat

from qdiff.quant_layer import QuantModule, UniformAffineQuantizer, StraightThrough
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, ResBlock, TimestepBlock, checkpoint
from ldm.modules.diffusionmodules.openaimodel import QKMatMul, SMVMatMul
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.attention import exists, default

from ddim.models.diffusion import ResnetBlock, AttnBlock, nonlinearity


logger = logging.getLogger(__name__)


class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    """
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantResBlock(BaseQuantBlock, TimestepBlock):
    def __init__(
        self, res: ResBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.channels = res.channels
        self.emb_channels = res.emb_channels
        self.dropout = res.dropout
        self.out_channels = res.out_channels
        self.use_conv = res.use_conv
        self.use_checkpoint = res.use_checkpoint
        self.use_scale_shift_norm = res.use_scale_shift_norm

        self.in_layers = res.in_layers

        self.updown = res.updown

        self.h_upd = res.h_upd
        self.x_upd = res.x_upd

        self.emb_layers = res.emb_layers
        self.out_layers = res.out_layers

        self.skip_connection = res.skip_connection

    def forward(self, x, emb=None, split=0):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if split != 0 and self.skip_connection.split == 0:
            return checkpoint(
                self._forward, (x, emb, split), self.parameters(), self.use_checkpoint
            )
        return checkpoint(
                self._forward, (x, emb), self.parameters(), self.use_checkpoint
            )  

    def _forward(self, x, emb, split=0):
        # print(f"x shape {x.shape} emb shape {emb.shape}")
        if emb is None:
            assert(len(x) == 2)
            x, emb = x
        assert x.shape[2] == x.shape[3]

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        if split != 0:
            return self.skip_connection(x, split=split) + h
        return self.skip_connection(x) + h


class QuantQKMatMul(BaseQuantBlock):
    def __init__(
        self, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.scale = None
        self.use_act_quant = False
        self.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        
    def forward(self, q, k):
        if self.use_act_quant:
            quant_q = self.act_quantizer_q(q * self.scale)
            quant_k = self.act_quantizer_k(k * self.scale)
            weight = th.einsum(
                "bct,bcs->bts", quant_q, quant_k
            ) 
        else:
            weight = th.einsum(
                "bct,bcs->bts", q * self.scale, k * self.scale
            )
        return weight

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_act_quant = act_quant


class QuantSMVMatMul(BaseQuantBlock):
    def __init__(
        self, act_quant_params: dict = {}, sm_abit=8):
        super().__init__(act_quant_params)
        self.use_act_quant = False
        self.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        act_quant_params_w['symmetric'] = False
        act_quant_params_w['always_zero'] = True
        self.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)
        
    def forward(self, weight, v):
        if self.use_act_quant:
            a = th.einsum("bts,bcs->bct", self.act_quantizer_w(weight), self.act_quantizer_v(v))
        else:
            a = th.einsum("bts,bcs->bct", weight, v)
        return a

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_act_quant = act_quant


class QuantAttentionBlock(BaseQuantBlock):
    def __init__(
        self, attn: AttentionBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.channels = attn.channels
        self.num_heads = attn.num_heads
        self.use_checkpoint = attn.use_checkpoint
        self.norm = attn.norm
        self.qkv = attn.qkv
        
        self.attention = attn.attention

        self.proj_out = attn.proj_out

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def cross_attn_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    if self.use_act_quant:
        quant_q = self.act_quantizer_q(q)
        quant_k = self.act_quantizer_k(k)
        sim = einsum('b i d, b j d -> b i j', quant_q, quant_k) * self.scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -th.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)

    if self.use_act_quant:
        out = einsum('b i j, b j d -> b i d', self.act_quantizer_w(attn), self.act_quantizer_v(v))
    else:
        out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


class QuantBasicTransformerBlock(BaseQuantBlock):
    def __init__(
        self, tran: BasicTransformerBlock, act_quant_params: dict = {}, 
        sm_abit: int = 8):
        super().__init__(act_quant_params)
        self.attn1 = tran.attn1
        self.ff = tran.ff
        self.attn2 = tran.attn2
        
        self.norm1 = tran.norm1
        self.norm2 = tran.norm2
        self.norm3 = tran.norm3
        self.checkpoint = tran.checkpoint
        # self.checkpoint = False

        # logger.info(f"quant attn matmul")
        self.attn1.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.attn1.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.attn1.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)

        self.attn2.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.attn2.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.attn2.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)
        
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        act_quant_params_w['always_zero'] = True
        self.attn1.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)
        self.attn2.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)

        self.attn1.forward = MethodType(cross_attn_forward, self.attn1)
        self.attn2.forward = MethodType(cross_attn_forward, self.attn2)
        self.attn1.use_act_quant = False
        self.attn2.use_act_quant = False

    def forward(self, x, context=None):
        # print(f"x shape {x.shape} context shape {context.shape}")
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        if context is None:
            assert(len(x) == 2)
            x, context = x

        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.attn1.use_act_quant = act_quant
        self.attn2.use_act_quant = act_quant

        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


# the two classes below are for DDIM CIFAR
class QuantResnetBlock(BaseQuantBlock):
    def __init__(
        self, res: ResnetBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.in_channels = res.in_channels
        self.out_channels = res.out_channels
        self.use_conv_shortcut = res.use_conv_shortcut

        self.norm1 = res.norm1
        self.conv1 = res.conv1
        self.temb_proj = res.temb_proj
        self.norm2 = res.norm2
        self.dropout = res.dropout
        self.conv2 = res.conv2
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = res.conv_shortcut
            else:
                self.nin_shortcut = res.nin_shortcut


    def forward(self, x, temb=None, split=0):
        if temb is None:
            assert(len(x) == 2)
            x, temb = x

        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x, split=split)
        out = x + h
        return out


class QuantAttnBlock(BaseQuantBlock):
    def __init__(
        self, attn: AttnBlock, act_quant_params: dict = {}, sm_abit=8):
        super().__init__(act_quant_params)
        self.in_channels = attn.in_channels

        self.norm = attn.norm
        self.q = attn.q
        self.k = attn.k
        self.v = attn.v
        self.proj_out = attn.proj_out

        self.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)
        
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        self.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        if self.use_act_quant:
            q = self.act_quantizer_q(q)
            k = self.act_quantizer_k(k)
        w_ = th.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        if self.use_act_quant:
            v = self.act_quantizer_v(v)
            w_ = self.act_quantizer_w(w_)
        h_ = th.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        
        out = x + h_
        return out


def get_specials(quant_act=False):
    specials = {
        ResBlock: QuantResBlock,
        BasicTransformerBlock: QuantBasicTransformerBlock,
        ResnetBlock: QuantResnetBlock,
        AttnBlock: QuantAttnBlock,
    }
    if quant_act:
        specials[QKMatMul] = QuantQKMatMul
        specials[SMVMatMul] = QuantSMVMatMul
    else:
        specials[AttentionBlock] = QuantAttentionBlock
    return specials