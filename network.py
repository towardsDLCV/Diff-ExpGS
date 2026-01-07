import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pygments.lexers.lean import LeanLexer
from torch.ao.nn.quantized.functional import upsample

from utils.loss_utils import *
import math
import ImageReward as RM

from einops import rearrange

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class LearnableTanh(nn.Module):

    def __init__(self, dim):
        super(LearnableTanh, self).__init__()
        self.slope = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)

    def forward(self, x):
        return self.scale * torch.tanh(self.slope * x)

##########################################################################
## Layer Norm
def to_2d(x):
    return rearrange(x, 'b c h w -> b (h w c)')

def to_3d(x):
#    return rearrange(x, 'b c h w -> b c (h w)')
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
#    return rearrange(x, 'b c (h w) -> b c h w',h=h,w=w)
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

import numbers

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) #* self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) #* self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.net = nn.Sequential(nn.Conv2d(dim, hidden_features, kernel_size=1, padding=0, bias=bias),
                                 nn.GELU(),
                                 nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, bias=bias,
                                           groups=hidden_features),
                                 nn.GELU(),
                                 nn.Conv2d(hidden_features, dim, kernel_size=1, padding=0, bias=bias),)

    def forward(self, x):
        return self.net(x)


class IntensityCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, ifBox=True):
        super(IntensityCrossAttention, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)  # * self.weight + self.bias

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x, I_cond):
        b, c, h, w = x.shape
        x_sort, idx_h = x[:, :c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:, :c // 2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # b,c,x,x

        v = v * I_cond

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:, :c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c // 2] = out_replace
        return out


class HistogramTransformer(nn.Module):

    def __init__(self, dim, num_heads, bias, LayerNorm_type='WithBias', ffn_expansion_factor=2.66, i_dim=48):
        super(HistogramTransformer, self).__init__()

        self.attn_g = IntensityCrossAttention(dim, num_heads, bias, True)
        self.norm_g = LayerNorm(dim, LayerNorm_type)
        self.norm_g1 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.norm_ff1 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x, I_cond):
        h, w = x.shape[2:]
        I_cond = F.interpolate(I_cond, size=(h, w), mode='bilinear', align_corners=False)

        x = x + self.attn_g(self.norm_g(x), self.norm_g1(I_cond))
        x_out = x + self.ffn(self.norm_ff1(x))
        return x_out



class HistAttn_Single_Stage(nn.Module):

    def __init__(self, scale=32, s_max=1.2, exp=0.6, hist_scale=0.25):
        super(HistAttn_Single_Stage, self).__init__()
        self.s_max = s_max
        self.scale = scale
        self.exp = exp
        self.hist_scale = hist_scale
        self.tanh1 = LearnableTanh(1)
        self.tanh2 = LearnableTanh(1)
        self.tanh3 = LearnableTanh(1)

        self.patch_embed = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, stride=1, bias=True),
                                         nn.ReLU(),
                                         nn.Conv2d(32, 64, 3, padding=1, stride=1, bias=True))
        self.I_embed = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, stride=1, bias=True),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, 3, padding=1, stride=1, bias=True))
        self.act = nn.ReLU()
        self.attn = HistogramTransformer(64, 2, False)

        self.heads = nn.Conv2d(64, 4, 3, padding=1, stride=1, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def upsampling(self, x, size, mode='bilinear', align_corners=False):
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)

    def dual_gamma(self, x, gamma1, gamma2, w):
        x = x.clamp(1e-6, 1)
        f1 = x.pow(gamma1)
        f2 = x.pow(gamma2)
        return w * f1 + (1 - w) * f2

    def hist_residual(self, x, I, target_exp=0.6, k=0.8, scale=1.0):
        mean_I = I.mean()
        gamma = torch.exp(k * (mean_I - target_exp))
        g = x.clamp(1e-6, 1).pow(gamma)
        return scale * (g - x)

    def forward(self, x):
        dtypes = x.dtype
        h, w = x.shape[2:]
        Y = self.avgpool(torch.mean(x, 1))

        i = x.max(1)[0].to(dtypes).unsqueeze(0)

        x_cond = self.act(self.patch_embed(x))
        i_cond = self.act(self.I_embed(i))

        attn = self.attn(x_cond, i_cond)
        raw = self.heads(attn)

        s1 = self.tanh1((raw[:, 0:1]))
        s2 = self.tanh2(raw[:, 1:2])
        weight = torch.sigmoid(raw[:, 2:3])
        r = self.tanh3(raw[:, 3:4])

        gamma1 = torch.exp(s1)
        gamma2 = torch.exp(s2)

        gamma1 = self.upsampling(gamma1, size=(h, w), mode='bilinear', align_corners=False)
        gamma2 = self.upsampling(gamma2, size=(h, w), mode='bilinear', align_corners=False)
        weight = self.upsampling(weight, size=(h, w), mode='bilinear', align_corners=False)
        r = self.upsampling(r, size=(h, w), mode='bilinear', align_corners=False)

        f_x = self.dual_gamma(x, gamma1, gamma2, weight)

        if Y >= self.exp:
            x = x + r * (x ** 2 - x)  # over exposure
        else:
            x = x + r * (f_x ** 2 - x)  # under exposure
        return x, r


class HEC(nn.Module):

    def __init__(self, stage=3, s_max=1.2, exp=0.6, hist_scale=0.25):
        super(HEC, self).__init__()
        print(f"stage: {stage}")
        # self.factor =
        self.exp = exp

        module_body = [HistAttn_Single_Stage(scale=s, s_max=s_max, exp=exp, hist_scale=hist_scale) for s in range(stage)]
        self.body = nn.ModuleList(module_body)

    def forward(self, x):
        r_list = []
        for m in self.body:
            x, r = m(x)
            r_list.append(r)
        rs = torch.cat(r_list, dim=1)
        return x, rs
