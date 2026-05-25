from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


Batch = Mapping[str, jnp.ndarray]


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return losses.mean()


def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> dict[str, jnp.ndarray]:
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    loss = cross_entropy_loss(logits, labels)
    return {
        "loss": loss,
        "accuracy": accuracy,
    }


def create_train_state(
    rng: jax.Array,
    model: Any,
    learning_rate: float,
    input_shape: tuple[int, ...],
) -> train_state.TrainState:
    sample_inputs = jnp.ones(input_shape, dtype=jnp.float32)
    variables = model.init(rng, sample_inputs)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )


@jax.jit
def train_step(
    state: train_state.TrainState,
    batch: Batch,
) -> tuple[train_state.TrainState, dict[str, jnp.ndarray]]:
    def loss_fn(params: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        logits = state.apply_fn({"params": params}, batch["images"])
        loss = cross_entropy_loss(logits, batch["labels"])
        return loss, logits

    (_, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch["labels"])
    return new_state, metrics


@jax.jit
def eval_step(
    state: train_state.TrainState,
    batch: Batch,
) -> dict[str, jnp.ndarray]:
    logits = state.apply_fn({"params": state.params}, batch["images"])
    return compute_metrics(logits, batch["labels"])
