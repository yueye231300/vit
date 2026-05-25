import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset


class PyTorchBenchmarkViTTests(unittest.TestCase):
    def test_parser_defaults_match_jax_benchmark(self) -> None:
        from vision_transformer_pytorch_benchmark import build_parser

        args = build_parser().parse_args([])

        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.learning_rate, 0.005)
        self.assertEqual(args.patch_size, 4)
        self.assertEqual(args.hidden_dim, 8)
        self.assertEqual(args.depth, 4)
        self.assertEqual(args.num_heads, 2)
        self.assertEqual(args.mlp_dim, 32)
        self.assertEqual(args.num_classes, 10)

    def test_model_forward_returns_batch_logits(self) -> None:
        from vision_transformer_pytorch_benchmark import BenchmarkVisionTransformer

        model = BenchmarkVisionTransformer()
        batch = torch.randn(3, 1, 28, 28)
        logits = model(batch)

        self.assertEqual(tuple(logits.shape), (3, 10))

    def test_prepare_images_normalizes_uint8_images(self) -> None:
        from vision_transformer_pytorch_benchmark import prepare_images

        images = torch.full((2, 28, 28), 255, dtype=torch.uint8)
        prepared = prepare_images(images)

        self.assertEqual(prepared.dtype, torch.float32)
        self.assertEqual(tuple(prepared.shape), (2, 1, 28, 28))
        self.assertTrue(torch.allclose(prepared, torch.ones_like(prepared)))

    def test_train_one_epoch_returns_scalar_metrics(self) -> None:
        from vision_transformer_pytorch_benchmark import (
            BenchmarkVisionTransformer,
            train_one_epoch,
        )

        model = BenchmarkVisionTransformer()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = torch.nn.CrossEntropyLoss()
        dataset = TensorDataset(
            torch.randn(8, 1, 28, 28),
            torch.randint(0, 10, (8,), dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        metrics = train_one_epoch(
            model, loader, optimizer, criterion, torch.device("cpu")
        )

        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIsInstance(metrics["loss"], float)
        self.assertIsInstance(metrics["accuracy"], float)


if __name__ == "__main__":
    unittest.main()
