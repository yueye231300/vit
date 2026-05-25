# vit_jax

`vit_jax` is a small Vision Transformer implementation for MNIST written with `JAX + Flax + Optax`.

It is meant to be compared directly with the existing PyTorch script in this directory:
- `vision_transformer_from_scratch.py`

The default model and training arguments are aligned with the PyTorch script's current `main()` configuration so you can make a closer speed comparison without manually re-entering hyperparameters.

## Setup

This worktree uses a project-local conda environment with the JAX GPU build.

From the worktree root:

```bash
conda activate /home/yuyue/projects/vit/.worktrees/vit-jax/.conda
cd /home/yuyue/projects/vit/.worktrees/vit-jax/15min_for_vit
```

If you want to recreate the environment:

```bash
conda env create --prefix /home/yuyue/projects/vit/.worktrees/vit-jax/.conda --file environment.yml --yes


To verify that JAX sees the GPU:

```bash
python -c "import jax; print(jax.__version__); print(jax.default_backend()); print(jax.devices())"
```

Expected output on this machine includes:
- `0.9.2`
- `gpu`
- `CudaDevice(id=0)`

## Run

Train the model:

```bash
python run_vit_jax.py --epochs 1 --batch-size 32
```

The script will:
- print the JAX backend and visible devices
- download MNIST into `data/mnist` if it is missing
- train a small ViT
- print train and eval loss and accuracy per epoch

## Project Layout

- `vit_jax/model.py`: patch embedding, transformer blocks, and classifier
- `vit_jax/train.py`: loss function, `TrainState`, `train_step`, `eval_step`
- `vit_jax/input_pipeline.py`: MNIST download, IDX parsing, batching, normalization
- `vit_jax/main.py`: CLI entrypoint and training loop
- `run_vit_jax.py`: thin launcher
- `tests/`: minimal regression tests for model shape, train state, and batch preparation

## What These JAX Pieces Mean

`Flax`
- JAX ecosystem neural network library
- roughly plays the role of `torch.nn`
- model structure is defined separately from parameter values

`Optax`
- optimizer library for JAX
- roughly plays the role of `torch.optim`
- Adam and other optimizers are expressed as explicit transformations

`JIT`
- `jax.jit` compiles a pure function for faster execution
- here it is used on `train_step` and `eval_step`

`TrainState`
- a Flax helper object that stores model parameters, optimizer state, and the apply function
- it gives you a clean way to carry training state through each step

## PyTorch vs JAX

This repository now shows the same ViT idea in two different styles.

`PyTorch` in `vision_transformer_from_scratch.py`
- parameters live inside the module object
- forward pass is usually `model(images)`
- gradients come from `loss.backward()`
- optimization happens with `optimizer.step()`
- device placement is usually explicit with `.to(device)`
- randomness is often more implicit

`JAX` in `vit_jax`
- parameters are initialized with `model.init(...)`
- forward pass is usually `model.apply({"params": params}, images)`
- gradients come from `jax.value_and_grad(...)`
- optimization happens by returning a new `TrainState`
- JAX chooses the backend device from the active runtime
- randomness is explicit through `jax.random.PRNGKey`

The main writing-style difference is that PyTorch is more object-state oriented, while JAX is more function-state oriented.

## Why This Version Is More Standard JAX

This implementation does not try to copy the PyTorch script line by line.

Instead it uses the usual JAX workflow:
- `flax.linen.Module` for model definition
- `TrainState` for training state
- `optax.adam` for optimization
- `jax.jit` for compiled train and eval steps
- explicit batch preparation into `jnp.ndarray`

That is the main reason this version is easier to extend into a larger JAX project.

## Notes for WSL2 GPU

On this machine, JAX successfully selected the GPU backend under WSL2.

During verification, JAX also printed a warning about parsing the kernel mode driver version. The important point is that backend detection still succeeded and `jax.devices()` returned `CudaDevice(id=0)`.
