from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError("Expected a pair for image size.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class PatchEmbedding(nn.Module):
    patch_size: int
    hidden_dim: int

    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        batch_size, height, width, channels = images.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch_size.")

        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        patches = images.reshape(
            batch_size,
            patch_height,
            self.patch_size,
            patch_width,
            self.patch_size,
            channels,
        )
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(
            batch_size,
            patch_height * patch_width,
            self.patch_size * self.patch_size * channels,
        )
        return nn.Dense(self.hidden_dim, name="projection")(patches)


class MLPBlock(nn.Module):
    hidden_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.mlp_dim)(inputs)
        x = nn.gelu(x)
        return nn.Dense(self.hidden_dim)(x)


class EncoderBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.LayerNorm()(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
        )(x)
        x = x + inputs
        y = nn.LayerNorm()(x)
        y = MLPBlock(hidden_dim=self.hidden_dim, mlp_dim=self.mlp_dim)(y)
        return x + y


class VisionTransformer(nn.Module):
    image_size: int | Sequence[int]
    patch_size: int
    hidden_dim: int
    depth: int
    num_heads: int
    mlp_dim: int
    num_classes: int

    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        expected_height, expected_width = _pair(self.image_size)
        if images.shape[1:3] != (expected_height, expected_width):
            raise ValueError("Input image shape does not match image_size.")

        tokens = PatchEmbedding(
            patch_size=self.patch_size,
            hidden_dim=self.hidden_dim,
        )(images)
        batch_size, num_patches, _ = tokens.shape

        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.hidden_dim),
        )
        cls_tokens = jnp.tile(cls_token, (batch_size, 1, 1))
        x = jnp.concatenate([cls_tokens, tokens], axis=1)

        position_embeddings = self.param(
            "position_embeddings",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches + 1, self.hidden_dim),
        )
        x = x + position_embeddings

        for layer_index in range(self.depth):
            x = EncoderBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                name=f"encoder_block_{layer_index}",
            )(x)

        x = nn.LayerNorm(name="encoder_norm")(x)
        cls_output = x[:, 0]
        return nn.Dense(self.num_classes, name="classifier")(cls_output)
