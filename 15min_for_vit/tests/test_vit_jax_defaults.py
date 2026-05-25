from vit_jax.main impor buld_parser
from vit_jax.model import VisionTransformer


def test_model_defaults_match_pytorch_training_script():
    model = VisionTransformer()

    assert model.image_size == 28
    assert model.patch_size == 4
    assert model.hidden_dim == 8
    assert model.depth == 4
    assert model.num_heads == 2
    assert model.mlp_dim == 32
    assert model.num_classes == 10


def test_cli_defaults_match_pytorch_training_script():
    args = build_parser().parse_args([])

    assert args.epochs == 10
    assert args.batch_size == 32
    assert args.learning_rate == 0.005
    assert args.patch_size == 4
    assert args.hidden_dim == 8
    assert args.depth == 4
    assert args.num_heads == 2
    assert args.mlp_dim == 32
    assert args.num_classes == 10
