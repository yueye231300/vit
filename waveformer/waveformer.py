import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, trunc_normal_
import torch_dct

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1).contiguous()


def build_norm_layer(
    dim, norm_layer, in_format="channels_last", out_format="channels_last", eps=1e-6
):
    layers = []
    if norm_layer == "BN":
        if in_format == "channels_last":
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == "channels_last":
            layers.append(to_channels_last())
    elif norm_layer == "LN":
        if in_format == "channels_first":
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == "channels_first":
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f"build_norm_layer does not support {norm_layer}")
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == "ReLU":
        return nn.ReLU(inplace=True)
    elif act_layer == "SiLU":
        return nn.SiLU(inplace=True)
    elif act_layer == "GELU":
        return nn.GELU()

    raise NotImplementedError(f"build_act_layer does not support {act_layer}")


class StemLayer(nn.Module):
    r"""Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self, in_chans=3, out_chans=96, act_layer="GELU", norm_layer="BN"):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_chans, out_chans // 2, kernel_size=3, stride=2, padding=1
        )
        self.norm1 = build_norm_layer(
            out_chans // 2, norm_layer, "channels_first", "channels_first"
        )
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(
            out_chans // 2, out_chans, kernel_size=3, stride=2, padding=1
        )
        self.norm2 = build_norm_layer(
            out_chans, norm_layer, "channels_first", "channels_first"
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        channels_first=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = (
            partial(nn.Conv2d, kernel_size=1, padding=0)
            if channels_first
            else nn.Linear
        )
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Wave2D(nn.Module):
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dwconv = nn.Conv2d(
            dim, hidden_dim, kernel_size=3, padding=1, groups=min(dim, hidden_dim)
        )
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        # 新增: 用于生成不同的初始速度
        self.velocity_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        # Add wave speed parameter
        self.c = nn.Parameter(torch.ones(1) * 1)
        # Add damping parameter
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        # self.alpha = self.alpha.to('cuda')

    @staticmethod
    def get_decay_map(
        resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float
    ):
        # (1 - [(n\pi/a)^2 + (m\pi/b)^2]c2t2) * e^(-αt)
        # returns: (Res_h, Res_w)
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[
            :resh
        ].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[
            :resw
        ].view(1, -1)
        # Quadratic term for wave equation
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def infer_init_wave2d(self, freq):
        """预计算推理时需要的值"""
        H, W = self.res, self.res
        weight_exp = self.get_decay_map((H, W), device=freq.device, dtype=freq.dtype)
        # 预计算时间嵌入
        t = self.to_k(freq)  # (H, W, C)
        # 预计算cos和sin项
        self.register_buffer('cached_cos', torch.cos(self.c * t).permute(2, 0, 1).contiguous())  # (C, H, W)
        self.register_buffer('cached_sin', torch.sin(self.c * t).permute(2, 0, 1).contiguous() / (self.c + 1e-6))
        self.register_buffer('cached_decay', weight_exp)  # (H, W)
        # 删除不再需要的模块
        del self.to_k

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, H, W = x.shape
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())  # B,H,W,2C
        x, z = x.chunk(chunks=2, dim=-1)  # B,H,W,C
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        z = z.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        def dct2d(tensor):
            return torch_dct.dct_2d(tensor, norm="ortho")

        def idct2d(tensor):
            return torch_dct.idct_2d(tensor, norm="ortho")

        # 初始位移的DCT
        x_u0 = dct2d(x)
        # 修复: 使用velocity_linear生成不同的初始速度
        x_velocity = self.velocity_linear(x.permute(0, 2, 3, 1).contiguous())
        x_v0 = dct2d(x_velocity.permute(0, 3, 1, 2).contiguous())

        # 获取频率衰减图
        if (H, W) == getattr(self, '_cached_res', (0, 0)):
            decay_map = getattr(self, '_cached_decay_map')
        else:
            decay_map = self.get_decay_map((H, W), device=x.device, dtype=x.dtype)
            self._cached_res = (H, W)
            self._cached_decay_map = decay_map
        
        # 应用频率衰减 (高频衰减更多)
        decay_map = decay_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        x_u0 = x_u0 * decay_map
        x_v0 = x_v0 * decay_map

        # 计算波动方程的时间演化项
        if self.infer_mode and hasattr(self, 'cached_cos'):
            # 推理模式：使用预计算的值
            cos_term = self.cached_cos.unsqueeze(0).expand(B, -1, -1, -1)  # (B, C, H, W)
            sin_term = self.cached_sin.unsqueeze(0).expand(B, -1, -1, -1)
        else:
            # 训练模式
            if freq_embed is not None:
                # 检查 freq_embed 的空间尺寸是否与特征图匹配
                fe_h, fe_w, fe_c = freq_embed.shape
                if fe_h != H or fe_w != W:
                    # 插值调整 freq_embed 的空间尺寸
                    freq_embed_resized = freq_embed.permute(2, 0, 1).unsqueeze(0)  # (1, C, fe_h, fe_w)
                    freq_embed_resized = F.interpolate(freq_embed_resized, size=(H, W), mode='bilinear', align_corners=False)
                    freq_embed_resized = freq_embed_resized.squeeze(0).permute(1, 2, 0).contiguous()  # (H, W, C)
                else:
                    freq_embed_resized = freq_embed
                t = self.to_k(freq_embed_resized.unsqueeze(0).expand(B, -1, -1, -1).contiguous())
            else:
                t = torch.zeros((B, H, W, C), device=x.device, dtype=x.dtype)
            
            eps = 1e-6
            cos_term = torch.cos(self.c * t).permute(0, 3, 1, 2).contiguous()
            sin_term = torch.sin(self.c * t).permute(0, 3, 1, 2).contiguous() / (self.c + eps)

        # 波动方程解: u(t) = cos(ct)*u0 + sin(ct)/c * (v0 + alpha/2 * u0)
        wave_term = cos_term * x_u0
        velocity_term = sin_term * (x_v0 + (self.alpha / 2) * x_u0)
        final_term = wave_term + velocity_term

        x_final = idct2d(final_term)
        
        # 输出处理
        x = self.out_norm(x_final.permute(0, 2, 3, 1).contiguous())
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x * F.silu(z)  # 门控机制
        x = self.out_linear(x.permute(0, 2, 3, 1).contiguous())
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class WaveBlock(nn.Module):
    def __init__(
        self,
        res: int = 14,
        infer_mode=False,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        mlp_ratio: float = 4.0,
        post_norm=True,
        layer_scale=None,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.op = Wave2D(
            res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                channels_first=True,
            )
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None

        self.infer_mode = infer_mode

        if self.layer_scale:
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(hidden_dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(hidden_dim), requires_grad=True
            )

    def _forward(self, x: torch.Tensor, freq_embed):
        if not self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.op(x, freq_embed)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.op(self.norm1(x), freq_embed))
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
            return x
        if self.post_norm:
            x = x + self.drop_path(
                self.gamma1[:, None, None] * self.norm1(self.op(x, freq_embed))
            )
            if self.mlp_branch:
                x = x + self.drop_path(
                    self.gamma2[:, None, None] * self.norm2(self.mlp(x))
                )  # FFN
        else:
            x = x + self.drop_path(
                self.gamma1[:, None, None] * self.op(self.norm1(x), freq_embed)
            )
            if self.mlp_branch:
                x = x + self.drop_path(
                    self.gamma2[:, None, None] * self.mlp(self.norm2(x))
                )  # FFN
        return x

    def forward(self, input: torch.Tensor, freq_embed=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, freq_embed)
        else:
            return self._forward(input, freq_embed)


class AdditionalInputSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self[:-1]:
            if isinstance(module, nn.Module):
                x = module(x, *args, **kwargs)
            else:
                x = module(x)
        x = self[-1](x)
        return x


class WaveFormer(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.2,
        patch_norm=True,
        post_norm=True,
        layer_scale=None,
        use_checkpoint=False,
        mlp_ratio=4.0,
        img_size=224,
        act_layer="GELU",
        infer_mode=False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2**i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.depths = depths

        self.patch_embed = StemLayer(
            in_chans=in_chans,
            out_chans=self.embed_dim,
            act_layer="GELU",
            norm_layer="LN",
        )

        res0 = img_size / patch_size
        self.res = [int(res0), int(res0 // 2), int(res0 // 4), int(res0 // 8)]

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        self.infer_mode = infer_mode

        self.freq_embed = nn.ParameterList()
        for i in range(self.num_layers):
            self.freq_embed.append(
                nn.Parameter(
                    torch.zeros(self.res[i], self.res[i], self.dims[i]),
                    requires_grad=True,
                )
            )
            trunc_normal_(self.freq_embed[i], std=0.02)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(
                self.make_layer(
                    res=self.res[i_layer],
                    dim=self.dims[i_layer],
                    depth=depths[i_layer],
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    use_checkpoint=use_checkpoint,
                    norm_layer=LayerNorm2d,
                    post_norm=post_norm,
                    layer_scale=layer_scale,
                    downsample=(
                        self.make_downsample(
                            self.dims[i_layer],
                            self.dims[i_layer + 1],
                            norm_layer=LayerNorm2d,
                        )
                        if (i_layer < self.num_layers - 1)
                        else nn.Identity()
                    ),
                    mlp_ratio=mlp_ratio,
                    infer_mode=infer_mode,
                )
            )

        self.classifier = nn.Sequential(
            LayerNorm2d(self.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.num_features, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def make_downsample(dim=96, out_dim=192, norm_layer=LayerNorm2d):
        return nn.Sequential(
            # norm_layer(dim),
            # nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_dim),
        )

    @staticmethod
    def make_layer(
        res=14,
        dim=96,
        depth=2,
        drop_path=[0.1, 0.1],
        use_checkpoint=False,
        norm_layer=LayerNorm2d,
        post_norm=True,
        layer_scale=None,
        downsample=nn.Identity(),
        mlp_ratio=4.0,
        infer_mode=False,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(
                WaveBlock(
                    res=res,
                    hidden_dim=dim,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    mlp_ratio=mlp_ratio,
                    post_norm=post_norm,
                    layer_scale=layer_scale,
                    infer_mode=infer_mode,
                )
            )

        return AdditionalInputSequential(
            *blocks,
            downsample,
        )

    def _init_weights(self, m: nn.Module):
        """初始化模型权重"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # 使用Kaiming初始化Conv2d
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def infer_init(self):
        for i, layer in enumerate(self.layers):
            for block in layer[:-1]:
                block.op.infer_init_wave2d(self.freq_embed[i])
        del self.freq_embed

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.infer_mode:
            for layer in self.layers:
                x = layer(x)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x, self.freq_embed[i])  # (B, C, H, W)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from fvcore.nn import flop_count_table, flop_count_str, FlopCountAnalysis

    model = WaveFormer().cuda()
    input = torch.randn((1, 3, 224, 224), device=torch.device("cuda"))
    analyze = FlopCountAnalysis(model, (input,))
    print(flop_count_str(analyze))
