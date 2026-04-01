from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from vit_jax.input_pipeline import batch_iterator, load_mnist
from vit_jax.model import VisionTransformer
from vit_jax.train import create_train_state, eval_step, train_step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small Vision Transformer with JAX.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    return parser


def _mean_metrics(metrics_list: list[dict[str, jax.Array]]) -> dict[str, float]:
    return {
        key: float(np.mean([float(jax.device_get(metrics[key])) for metrics in metrics_list]))
        for key in metrics_list[0]
    }


def run_training(args: argparse.Namespace) -> None:
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    dataset = load_mnist(args.data_dir)
    model = VisionTransformer(
        image_size=28,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        num_classes=args.num_classes,
    )
    state = create_train_state(
        rng=jax.random.PRNGKey(args.seed),
        model=model,
        learning_rate=args.learning_rate,
        input_shape=(args.batch_size, 28, 28, 1),
    )

    for epoch in range(args.epochs):
        train_metrics = []
        for batch in batch_iterator(
            dataset["train_images"],
            dataset["train_labels"],
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        ):
            state, metrics = train_step(state, batch)
            train_metrics.append(metrics)

        eval_metrics = []
        for batch in batch_iterator(
            dataset["test_images"],
            dataset["test_labels"],
            args.batch_size,
            shuffle=False,
        ):
            eval_metrics.append(eval_step(state, batch))

        train_summary = _mean_metrics(train_metrics)
        eval_summary = _mean_metrics(eval_metrics)
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_summary['loss']:.4f} "
            f"train_acc={train_summary['accuracy']:.4f} "
            f"eval_loss={eval_summary['loss']:.4f} "
            f"eval_acc={eval_summary['accuracy']:.4f}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
