# PyTorch Benchmark ViT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new PyTorch-only benchmark Vision Transformer script with fewer dependencies and more realistic performance characteristics than the current educational script.

**Architecture:** The implementation will live in a single new runnable file so it is easy to compare against both the original PyTorch script and the JAX benchmark. The model and training path will be vectorized and use standard PyTorch modules such as `Conv2d` patch embedding and `MultiheadAttention`, while data loading will parse MNIST raw files directly without `torchvision`.

**Tech Stack:** Python standard library, PyTorch, pytest

---

## File Structure

- Create: `15min_for_vit/vision_transformer_pytorch_benchmark.py`
- Create: `15min_for_vit/tests/test_pytorch_benchmark_vit.py`
- Modify: `15min_for_vit/requirements.txt` only if required for clarity

### Task 1: Lock default benchmark configuration with tests

**Files:**
- Create: `15min_for_vit/tests/test_pytorch_benchmark_vit.py`
- Test: `15min_for_vit/tests/test_pytorch_benchmark_vit.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_parser_defaults_match_jax_benchmark():
    ...

def test_model_forward_returns_batch_logits():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pytorch_benchmark_vit.py -v`
Expected: FAIL because the new benchmark module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create the new script with:
- argument parser defaults
- benchmark model skeleton
- forward path returning logits shaped `(batch_size, 10)`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pytorch_benchmark_vit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add 15min_for_vit/tests/test_pytorch_benchmark_vit.py 15min_for_vit/vision_transformer_pytorch_benchmark.py
git commit -m "feat: add pytorch benchmark vit skeleton"
```

### Task 2: Add raw MNIST loading without torchvision

**Files:**
- Modify: `15min_for_vit/vision_transformer_pytorch_benchmark.py`
- Modify: `15min_for_vit/tests/test_pytorch_benchmark_vit.py`

- [ ] **Step 1: Write the failing test**

```python
def test_prepare_batch_normalizes_uint8_images():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pytorch_benchmark_vit.py::test_prepare_batch_normalizes_uint8_images -v`
Expected: FAIL because the helper does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement:
- gzip download helper
- IDX parsing helpers
- batch normalization helper
- dataset wrapper or tensor dataset construction

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pytorch_benchmark_vit.py::test_prepare_batch_normalizes_uint8_images -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add 15min_for_vit/vision_transformer_pytorch_benchmark.py 15min_for_vit/tests/test_pytorch_benchmark_vit.py
git commit -m "feat: add raw mnist loading for pytorch benchmark"
```

### Task 3: Add benchmark training and evaluation loop

**Files:**
- Modify: `15min_for_vit/vision_transformer_pytorch_benchmark.py`
- Modify: `15min_for_vit/tests/test_pytorch_benchmark_vit.py`

- [ ] **Step 1: Write the failing test**

```python
def test_train_one_epoch_returns_scalar_metrics():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pytorch_benchmark_vit.py::test_train_one_epoch_returns_scalar_metrics -v`
Expected: FAIL because the training helper is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

Implement:
- optimizer creation
- one-epoch training helper
- evaluation helper
- runnable `main()`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pytorch_benchmark_vit.py::test_train_one_epoch_returns_scalar_metrics -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add 15min_for_vit/vision_transformer_pytorch_benchmark.py 15min_for_vit/tests/test_pytorch_benchmark_vit.py
git commit -m "feat: add pytorch benchmark training loop"
```

### Task 4: End-to-end verification

**Files:**
- Verify: `15min_for_vit/vision_transformer_pytorch_benchmark.py`
- Verify: `15min_for_vit/tests/test_pytorch_benchmark_vit.py`

- [ ] **Step 1: Run the benchmark tests**

Run: `python -m pytest tests/test_pytorch_benchmark_vit.py -v`
Expected: PASS

- [ ] **Step 2: Run a short smoke test**

Run: `python vision_transformer_pytorch_benchmark.py --epochs 1 --batch-size 32`
Expected: script prints device plus train and eval metrics without `torchvision` or `tqdm`.

- [ ] **Step 3: Review output**

Check that:
- device prints correctly
- epoch metrics print correctly
- the script uses the intended defaults

- [ ] **Step 4: Commit**

```bash
git add 15min_for_vit/vision_transformer_pytorch_benchmark.py 15min_for_vit/tests/test_pytorch_benchmark_vit.py 15min_for_vit/docs/superpowers/specs/2026-04-01-pytorch-benchmark-design.md 15min_for_vit/docs/superpowers/plans/2026-04-01-pytorch-benchmark.md
git commit -m "feat: add pytorch benchmark vit implementation"
```
