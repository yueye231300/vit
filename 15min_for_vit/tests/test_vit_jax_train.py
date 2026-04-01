import jax
import jax.numpy as jnp
import numpy as np

from vit_jax import VisionTransformer


def _build_model() -> VisionTransformer:
    return VisionTransformer(
        image_size=28,
        patch_size=4,
        hidden_dim=32,
        depth=2,
        num_heads=4,
        mlp_dim=64,
        num_classes=10,
    )


def test_create_train_state_returns_expected_objects():
    from vit_jax.train import create_train_state

    state = create_train_state(
        rng=jax.random.PRNGKey(0),
        model=_build_model(),
        learning_rate=1e-3,
        input_shape=(2, 28, 28, 1),
    )

    assert state.apply_fn is not None
    assert state.params is not None
    assert int(state.step) == 0


def test_train_step_updates_state_and_returns_metrics():
    from vit_jax.train import create_train_state, train_step

    state = create_train_state(
        rng=jax.random.PRNGKey(0),
        model=_build_model(),
        learning_rate=1e-3,
        input_shape=(2, 28, 28, 1),
    )
    batch = {
        "images": jnp.ones((2, 28, 28, 1), dtype=jnp.float32),
        "labels": jnp.array([1, 3], dtype=jnp.int32),
    }

    new_state, metrics = train_step(state, batch)

    assert int(new_state.step) == int(state.step) + 1
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["loss"].shape == ()
    assert metrics["accuracy"].shape == ()
    assert not jnp.array_equal(
        state.params["classifier"]["kernel"],
        new_state.params["classifier"]["kernel"],
    )


def test_prepare_batch_returns_expected_dtypes_and_shapes():
    from vit_jax.input_pipeline import prepare_batch

    images = np.full((2, 28, 28), 255, dtype=np.uint8)
    labels = np.array([1, 3], dtype=np.uint8)

    batch = prepare_batch(images, labels)

    assert batch["images"].shape == (2, 28, 28, 1)
    assert batch["images"].dtype == jnp.float32
    assert batch["labels"].dtype == jnp.int32
    assert jnp.all(batch["images"] == 1.0)
