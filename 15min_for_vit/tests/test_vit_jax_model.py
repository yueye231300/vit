import jax
import jax.numpy as jnp


def test_package_exports_exist():
    from vit_jax import VisionTransformer

    assert VisionTransformer is not None


def test_patch_embedding_shapes():
    from vit_jax.model import PatchEmbedding

    batch = jnp.ones((2, 28, 28, 1), dtype=jnp.float32)
    layer = PatchEmbedding(patch_size=4, hidden_dim=32)
    variables = layer.init(jax.random.PRNGKey(0), batch)
    tokens = layer.apply(variables, batch)

    assert tokens.shape == (2, 49, 32)


def test_vit_forward_returns_batch_logits():
    from vit_jax import VisionTransformer

    model = VisionTransformer(
        image_size=28,
        patch_size=4,
        hidden_dim=32,
        depth=2,
        num_heads=4,
        mlp_dim=64,
        num_classes=10,
    )
    batch = jnp.ones((3, 28, 28, 1), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), batch)
    logits = model.apply(variables, batch)

    assert logits.shape == (3, 10)
