import fire
from timm.layers import attention
import torch
from torch.cuda import temperature
from modeling_gemma import KVCache
from PIL import Image
from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache,PaliGemmaForConditionalGeneration
from utils import load_hf_model

def move_inputs_to_device(model_inputs:dict,device:str):
    model_inputs = {k:v.to(device) for k,v in model_inputs.items()}
    return model_inputs

def _sample_top_p(probs:torch.Tensor,p:float):
    # （B，vocab_size)
    probs_sort,probs_idx = torch.sort(probs,dim=-1,descending=True)
    # (B,vocab_size)
    probs_sum = torch.cumsum(probs_sort,dim=-1) # data sum by the index
    # (substracting "prob_sort" shifts the cumulative sum by one 1 position to the right before masking)
    mask = probs_sum- probs_sort > p
    # zero out all the probabilities of the tokens that are not selected by the top p
    probs_sort[mask] =0.0
    # redistributed the probabilities so that they can sum up to 1
    probs_sort.div_(probs_sort.sum(dim=-1,keepdim=True))
    # sample a token (its token) from the top p distribution
    next_token = torch.multinomial(probs_sort,num_samples=1)
    # get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1 ,next_token)




def get_model_inputs(processor:PaliGemmaProcessor,prompt:str, image_file_path:str,device:str):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text = prompts , images = images)
    model_inputs = move_inputs_to_device(model_inputs,device)
    return model_inputs


def test_inference(
    model : PaliGemmaForConditionalGeneration,
    processor:PaliGemmaProcessor,
    device : str,
    prompt : str,
    image_file_path:str,
    max_tokens_to_generate:int,
    temperature: float,
    top_p:float,
    do_sample:bool,
):
    model_inputs = get_model_inputs(processor,prompt,image_file_path,device)
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    pixel_values = model_inputs['pixel_values']

    kv_cache = KVCache()

    # generate tokens until you see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        # get the model outputs  ,the first padding
        outputs = model(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask =attention_mask,
            kv_cache = kv_cache,
        )
        kv_cache = outputs['kv_cache']
        # we take the last seq_len
        next_token_logits = outputs['logits'][:,-1,:]
        # sample the next token
        if do_sample:
            # apply temperature
            next_token_logits = torch.softmax(next_token_logits/temperature,dim=-1)
            next_token = _sample_top_p(next_token_logits,top_p)
        else :
            next_token = torch.argmax(next_token_logits,dim=-1,keepdim=True)

        assert next_token.size() == (1,1)
        next_token = next_token.squeeze(0) # remove the batch dim
        generated_tokens.append(next_token)
        # stop if the stop token has been genetated
        if next_token.item() == stop_token:
            break
        # append the next token to the input
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask,torch.ones((1,1),device=input_ids.device)],dim=-1
        )
    generated_tokens = torch.cat(generated_tokens,dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens= True)

    print(decoded.strip())






def main(
  model_path:str=None,
  prompt : str =None,
  image_file_path : str= None,
  max_tokens_to_generate :int=100,
  temperature:float=0.8,
  top_p:float=0.9,
  do_sample:bool=False,
  only_cpu:bool=False
):
    device = "cpu"
    if not only_cpu :
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
    print("device in use :", device)

    print("loading model")
    model,tokenizer = load_hf_model(model_path,device)
    model = model.to(device=device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer,num_image_tokens,image_size)
    print("Running Inference")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
    fire.Fire(main)
