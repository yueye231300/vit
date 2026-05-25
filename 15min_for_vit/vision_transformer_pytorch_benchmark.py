from __future__ import annotations

import argparse
import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attn_inputs = self.norm1(inputs)
        attn_outputs, _ = self.attention(attn_inputs, attn_inputs, attn_inputs)
        outputs = inputs + attn_outputs
        mlp_inputs = self.norm2(outputs)
        return outputs + self.mlp(mlp_inputs)


class BenchmarkVisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        hidden_dim: int = 8,
        depth: int = 4,
        num_heads: int = 2,
        mlp_dim: int = 32,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes

        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, hidden_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [EncoderBlock(hidden_dim, num_heads, mlp_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.position_embeddings

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]
        return self.classifier(cls_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a benchmark-oriented Vision Transformer in PyTorch."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--mlp-dim", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    return parser


def prepare_images(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.ndim != 4:
        raise ValueError(
            "Expected images with shape (batch, height, width) or (batch, channels, height, width)."
        )

    images = images.to(torch.float32)
    if images.max().item() > 1.0:
        images = images / 255.0
    return images


def _download_mnist(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in MNIST_FILES.values():
        destination = data_dir / filename
        if destination.exists():
            continue
        urllib.request.urlretrieve(MNIST_BASE_URL + filename, destination)


def _read_idx_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as file_obj:
        magic, count, rows, cols = struct.unpack(">IIII", file_obj.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image file magic number: {magic}")
        buffer = file_obj.read()
    data = torch.frombuffer(bytearray(buffer), dtype=torch.uint8).clone()
    return data.view(count, rows, cols)


def _read_idx_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as file_obj:
        magic, count = struct.unpack(">II", file_obj.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label file magic number: {magic}")
        buffer = file_obj.read()
    data = torch.frombuffer(bytearray(buffer), dtype=torch.uint8).clone()
    return data.view(count).to(torch.long)


def create_data_loaders(
    data_dir: Path, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    _download_mnist(data_dir)
    train_images = prepare_images(
        _read_idx_images(data_dir / MNIST_FILES["train_images"])
    )
    train_labels = _read_idx_labels(data_dir / MNIST_FILES["train_labels"])
    test_images = prepare_images(
        _read_idx_images(data_dir / MNIST_FILES["test_images"])
    )
    test_labels = _read_idx_labels(data_dir / MNIST_FILES["test_labels"])

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def _accumulate_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss: torch.Tensor,
    total_loss: float,
    total_correct: int,
    total_items: int,
) -> tuple[float, int, int]:
    total_loss += loss.item() * labels.size(0)
    total_correct += (logits.argmax(dim=1) == labels).sum().item()
    total_items += labels.size(0)
    return total_loss, total_correct, total_items


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss, total_correct, total_items = _accumulate_metrics(
            logits, labels, loss, total_loss, total_correct, total_items
        )

    return {
        "loss": total_loss / total_items,
        "accuracy": total_correct / total_items,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss, total_correct, total_items = _accumulate_metrics(
            logits, labels, loss, total_loss, total_correct, total_items
        )

    return {
        "loss": total_loss / total_items,
        "accuracy": total_correct / total_items,
    }


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    model = BenchmarkVisionTransformer(
        image_size=28,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        num_classes=args.num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        eval_metrics = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"eval_loss={eval_metrics['loss']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
