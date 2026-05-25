from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Iterator

import jax.numpy as jnp
import numpy as np


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file_obj:
        magic, count, rows, cols = struct.unpack(">IIII", file_obj.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image file magic number: {magic}")
        buffer = file_obj.read()
    return np.frombuffer(buffer, dtype=np.uint8).reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file_obj:
        magic, count = struct.unpack(">II", file_obj.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label file magic number: {magic}")
        buffer = file_obj.read()
    return np.frombuffer(buffer, dtype=np.uint8).reshape(count)


def download_mnist(data_dir: str | Path) -> None:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for filename in MNIST_FILES.values():
        destination = data_path / filename
        if destination.exists():
            continue
        urllib.request.urlretrieve(MNIST_BASE_URL + filename, destination)


def load_mnist(data_dir: str | Path = "data/mnist") -> dict[str, np.ndarray]:
    data_path = Path(data_dir)
    download_mnist(data_path)
    return {
        "train_images": _read_idx_images(data_path / MNIST_FILES["train_images"]),
        "train_labels": _read_idx_labels(data_path / MNIST_FILES["train_labels"]),
        "test_images": _read_idx_images(data_path / MNIST_FILES["test_images"]),
        "test_labels": _read_idx_labels(data_path / MNIST_FILES["test_labels"]),
    }


def prepare_batch(images: np.ndarray, labels: np.ndarray) -> dict[str, jnp.ndarray]:
    batch_images = np.asarray(images, dtype=np.float32)
    if batch_images.ndim == 3:
        batch_images = batch_images[..., None]
    if batch_images.ndim != 4:
        raise ValueError(
            "Expected images with shape (batch, height, width[, channels])."
        )
    if batch_images.max(initial=0.0) > 1.0:
        batch_images = batch_images / 255.0

    batch_labels = np.asarray(labels, dtype=np.int32)
    return {
        "images": jnp.asarray(batch_images, dtype=jnp.float32),
        "labels": jnp.asarray(batch_labels, dtype=jnp.int32),
    }


def batch_iterator(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int | None = None,
) -> Iterator[dict[str, jnp.ndarray]]:
    indices = np.arange(images.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield prepare_batch(images[batch_indices], labels[batch_indices])
