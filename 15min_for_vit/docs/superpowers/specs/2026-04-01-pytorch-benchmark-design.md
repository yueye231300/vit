# PyTorch Benchmark ViT Design

**Goal**

Add a new PyTorch-only Vision Transformer benchmark script that is more suitable for fair speed comparisons against the JAX implementation than the current educational from-scratch script.

**Project Context**

The current PyTorch file, `vision_transformer_from_scratch.py`, is intentionally educational but performance-hostile. It uses Python loops in patch extraction and attention, depends on `torchvision` for MNIST loading, and uses `tqdm` for display. Those choices are reasonable for explanation, but they make the runtime comparison against the JAX implementation misleading.

**Recommended Approach**

Create a new standalone file named `vision_transformer_pytorch_benchmark.py` instead of rewriting the existing script.

This keeps the repository with two clearly different PyTorch examples:
- the current educational script
- a benchmark-oriented script with fewer dependencies and a more standard, optimized implementation

**Architecture**

The new benchmark script should:
- use only `torch` plus Python standard library
- download and parse raw MNIST IDX files directly, reusing the same dataset source concept as the JAX version
- use `nn.Conv2d(kernel_size=4, stride=4)` for patch embedding
- use `nn.MultiheadAttention(batch_first=True)` inside encoder blocks
- use learnable class token and learnable positional embeddings
- use a compact training loop without `tqdm`

This keeps the script close to the JAX benchmark structure:
- patch embedding
- class token
- positional embedding
- transformer blocks
- classification head

**Default Training Configuration**

The default training arguments should match the current JAX benchmark defaults and the PyTorch script's current `main()` configuration:
- `epochs=50`
- `batch_size=32`
- `learning_rate=0.005`
- `patch_size=4`
- `hidden_dim=8`
- `depth=4`
- `num_heads=2`
- `mlp_dim=32`
- `num_classes=10`

**Data Flow**

The new script should:
1. Download MNIST raw gzip files if missing
2. Parse IDX image and label files
3. Normalize images to `[0, 1]`
4. Build PyTorch tensors and data loaders
5. Train and evaluate the model
6. Print concise epoch metrics

**Dependencies**

The new benchmark file should avoid nonessential dependencies:
- keep `torch`
- remove the need for `torchvision`
- remove the need for `tqdm`

This improves portability and makes environment setup lighter.

**Testing Strategy**

Use TDD for the new file.

Initial tests should cover:
- raw batch to model forward shape
- parser defaults
- model defaults matching the JAX benchmark defaults

The benchmark script itself should also be smoke-tested with a short run.

**Acceptance Criteria**

The work is complete when:
- `vision_transformer_pytorch_benchmark.py` exists
- it depends only on `torch` plus standard library
- its defaults align with the JAX benchmark defaults
- it runs successfully in the provided PyTorch environment
- it is clearly more suitable for speed comparison than the original teaching-oriented script
