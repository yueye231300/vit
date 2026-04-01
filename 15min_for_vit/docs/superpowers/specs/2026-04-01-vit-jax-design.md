# vit_jax Design

**Goal**

Build a new Vision Transformer implementation named `vit_jax` using `Flax + Optax + JIT + TrainState`, keep the scope close to the existing MNIST-based PyTorch example, and make the result easy to compare, run, and extend.

**Project Context**

The current directory contains a single reference implementation, `vision_transformer_from_scratch.py`, which trains a small ViT on MNIST in PyTorch. The new JAX implementation should preserve the same educational value while using a more standard JAX training style instead of a line-by-line translation.

**Recommended Approach**

Use a small package layout under `vit_jax/` rather than a single monolithic script.

This keeps the code readable while still showing the standard JAX separation of concerns:
- `model.py` defines the Flax modules
- `train.py` defines train state creation and jitted train and eval steps
- `input_pipeline.py` prepares MNIST batches as NumPy/JAX arrays
- `main.py` is the runnable entrypoint
- `README.md` explains setup, running, and PyTorch vs JAX differences

This is more standard than a direct translation because parameters, RNG state, and optimizer state are handled explicitly and the training step is expressed as a pure function.

**Architecture**

The new implementation will mirror the conceptual structure of the PyTorch model:
- patch extraction
- patch projection to hidden tokens
- class token
- positional embeddings
- transformer encoder blocks
- classification head

But it will express these pieces in Flax modules and a functional training loop:
- `VisionTransformer` will be a `flax.linen.Module`
- initialization will use `model.init(rng, sample_batch)`
- forward passes will use `model.apply({"params": params}, images, train=...)`
- optimizer setup will use `optax.adam`
- training state will use `flax.training.train_state.TrainState`
- `train_step` and `eval_step` will be `jax.jit`-compiled

**Data Flow**

Training data will come from MNIST, matching the current project scope.

The flow will be:
1. Load MNIST and convert images to NumPy arrays with shape `(batch, height, width, channels)` or normalize consistently for JAX
2. Pass images into the ViT model
3. Compute logits and cross-entropy loss
4. Use `jax.value_and_grad` to get gradients
5. Apply gradients through `TrainState.apply_gradients`
6. Run a separate jitted eval step for validation metrics

**Testing Strategy**

Use TDD for the new code instead of relying on the existing repository setup.

The initial tests should cover:
- patch extraction and token shape expectations
- model forward output shape
- train state creation
- one train step returning updated state and scalar metrics

These tests are enough to establish a working correctness baseline before running a full MNIST training script.

**Environment Setup**

The current environment does not yet have JAX configured. The implementation should therefore include:
- installation of `jax`, `jaxlib`, `flax`, `optax`, and any minimal runtime dependencies needed for MNIST loading
- a short setup note in the new `vit_jax` documentation
- commands the user can rerun locally to test the implementation

Given the current repository shape, environment setup will be kept lightweight and local to this worktree where possible.

**Error Handling and Constraints**

This is an educational implementation, so the design intentionally keeps scope narrow:
- CPU execution is acceptable
- MNIST remains the default dataset
- model size stays small enough to run locally
- no attempt will be made to build a general training framework

Code should still fail clearly for invalid patch sizes or incompatible image shapes, and the training script should print enough information for the user to confirm the run is working.

**Comparison Material**

The finished work should explicitly explain the writing-style differences between PyTorch and JAX:
- object state vs explicit state
- implicit autograd vs explicit `grad/value_and_grad`
- `optimizer.step()` vs `apply_gradients`
- eager forward methods vs `init/apply`
- device movement patterns
- randomness handling with PRNG keys

This explanation is a first-class deliverable, not an afterthought.

**Acceptance Criteria**

The work is complete when:
- a new runnable `vit_jax` implementation exists in this project
- the code uses `Flax + Optax + JIT + TrainState`
- the environment can install the required JAX stack
- the user can run the code locally to verify behavior
- the repository contains a clear PyTorch vs JAX comparison for this implementation
