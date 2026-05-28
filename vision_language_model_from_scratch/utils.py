import glob
import json
import os
import token
from typing import Tuple

from fsspec.spec import tokenize
from modeling_gemma import PaliGemmaConfig, PaliGemmaForConditionalGeneration
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.paligemma.processing_paligemma import PaliGemmaImagesKwargs


def load_hf_model(
    model_path: str, device: str
) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # find all the *.safetensors files
    safetensors_file = glob.glob(os.path.join(model_path, "*.safetensors"))

    # load the safetensor one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_file:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # tie weights
    model.tie_weight()

    return (model, tokenizer)
