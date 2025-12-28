import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from einops import rearrange
from torchvision.ops import StochasticDepth


# overlap patch embedding
# stride < patch size to have overlapping patches
class LayerNorm2d(nn.LayerNorm):
    """
    LayerNorm2d 用于对channel维度进行归一化，保持height和width维度的信息不变。
    """

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x


class OverlapPatchMerging(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: int, stride: int
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=stride,
                padding=patch_size // 2,
                bias=False,
            ),
            LayerNorm2d(out_channels),
        )


class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio
            ),
            LayerNorm2d(channels),
        )
        self.att = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)
        # used to attention
        reduced_x = rearrange(reduced_x, "b c h w->b (h w) c")
        x = rearrange(x, "b c h w->b (h w) c")
        out = self.att(x, reduced_x, reduced_x)[0]
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out


channels = 8
# x = torch.randn(2, channels, 64, 64)
# print(x.shape)
# block = EfficientMultiHeadAttention(channels=channels, reduction_ratio=4)
# block(x).shape


# MixFFN
# 对于语意分割不需要使用位置编码，只需要考虑0填充的影响即可
class MixFFN(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            # dense layer,添加非线性,替换mlp
            nn.Conv2d(channels, channels, kernel_size=1),
            # depth wise conv
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.GELU(),
            # dense layer，混合通道信息
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )


# Encoder block
class ResidualAdd(nn.Module):
    """
    just an util layer
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kawargs):
        out = self.fn(x, **kawargs)
        return x + out


class SegFormerEncoderBlock(nn.Sequential):

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = 0.0,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(
                        channels=channels,
                        reduction_ratio=reduction_ratio,
                        num_heads=num_heads,
                    ),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixFFN(channels=channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch"),
                )
            ),
        )


x = torch.randn(1, channels, 64, 64)
block = SegFormerEncoderBlock(channels=channels, reduction_ratio=4)
block(x).shape

from typing import List


class SegFormerEncoderStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_probs: List[int],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            stride=overlap_size,
        )
        self.blocks = nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    channels=out_channels,
                    reduction_ratio=reduction_ratio,
                    num_heads=num_heads,
                    mlp_expansion=mlp_expansion,
                    drop_path_prob=drop_probs[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)


from typing import Sequence


def chunks(data: Sequence, sizes: List[int]):
    """
    Givens an sequence ,returns slices useing sizes as indices

    :param data: 支持切片的序列类型（如list、tuple）
    :type data: Sequence
    :param sizes: 每个片段的长度列表
    :type sizes: List[int]
    """
    curr = 0
    for size in sizes:
        chunk = data[curr : curr + size]
        curr += size
        yield chunk


class SegFormerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansion: List[int],
        drop_prob: float = 0.0,
    ):
        super().__init__()
        # 随机丢弃一些结果，防止过拟合
        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                    [in_channels, *widths],
                    widths,
                    patch_sizes,
                    overlap_sizes,
                    chunks(drop_probs, sizes=depths),
                    depths,
                    reduction_ratios,
                    all_num_heads,
                    mlp_expansion,
                )
            ]
        )

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


# 测试的block
class SegFormerDecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scaler_factor: int = 2):
        super().__init__(
            # 确定上采样方式，但是未来这个方式可能会被移除，这个需要注意
            nn.UpsamplingBilinear2d(scale_factor=scaler_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )


class SegFormerDecoder(nn.Module):
    def __init__(self, out_channels: int, widths: List[int], scale_factors: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factors)
                for in_channels, scale_factors in zip(widths, scale_factors)
            ]
        )

    def forward(self, features):
        new_features = []
        for feature, stage in zip(features, self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features


# 使用decoder head 完成对于特征的融合，使用卷积层
class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1),
            nn.ReLU(),  # why relu who knows
            nn.BatchNorm2d(channels),  # why batchnorm who knows
        )
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        x = torch.cat(features, dim=1)  # concat on channel dim
        x = self.fuse(x)
        x = self.classifier(x)
        return x


# 最后一步，超级拼装


class SegFormer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        decoder_channels: int,
        scale_factors: List[int],
        num_classes: int,
        drop_prob: float = 0.0,
    ):

        super().__init__()
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
        self.decoder = SegFormerDecoder(decoder_channels, widths[::-1], scale_factors)
        self.head = SegFormerSegmentationHead(
            decoder_channels, num_classes, num_features=len(widths)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        segmentation = self.head(features)
        return segmentation


segformer = SegFormer(
    in_channels=3,
    widths=[64, 128, 256, 512],
    depths=[3, 4, 6, 3],
    all_num_heads=[1, 2, 4, 8],
    patch_sizes=[7, 3, 3, 3],
    overlap_sizes=[4, 2, 2, 2],
    reduction_ratios=[8, 4, 2, 1],
    mlp_expansions=[4, 4, 4, 4],
    decoder_channels=256,
    scale_factors=[8, 4, 2, 1],
    num_classes=50,
)

segmentation = segformer(torch.randn((1, 3, 224, 224)))
print(segmentation.shape)
