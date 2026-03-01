from typing import Optional,Tuple
import torch 
import torch.nn as nn 

class SiglipVisionConfig:
    """
    This is the configuration class to store the configuration of a `SiglipVisionModel`.
    It is used to instantiate a SiglipVisionModel model according to the specified arguments, defining the model architecture.
    """
    def __init__(
        self, 
        hidden_size=718,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens:int=None,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
        self.num_image_tokens = num_image_tokens

class SigLipVisionEmbeddings(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim, # 存在768个[3 * 16 * 16]的卷积核
            kernel_size=self.patch_size,
            stride = self.patch_size,
            padding=0, # This indicates no padding is added 
        )

        self.num_patches = (self.image_size//self.patch_size)**2
        self.num_position = self.num_patches 
        # 一个查找表，将原来的的[1,num_patches] -> [1,num_patches,embed_dim] 
        # position_embedding 先得到ids，然后得到对应的embed_dim 的位置参数，用于学习位置参数
        self.position_embedding = nn.Embedding(self.num_position,self.embed_dim)
        
        self.register_buffer(
            # 不参与梯度更新
            "position_ids",
            torch.arange(self.num_position).expand((1,-1)),
            persistent=False
        )

        
        
    def forward(self,pixel_values:torch.FloatTensor)->torch.Tensor:
        _,_,height,width = pixel_values # [batch_size, channels,height,width]
        # convolve the "patch_size" kernal over the image , with no overlapping patches 
        # The output of the convolution the shape [Batch_Size,Embed_Dim,Num_patches_H,Num_patches_W]
        patch_embeds = self.patch_embedding(pixel_values)
        # use the flatten to make 2d dim to the flatten list 
        # from [Batch_Size,Embed_Dim,Num_patches_H,Num_patches_W] to the [Batch_Size,Embed_dim,Num_Patches]
        embeddings = patch_embeds.flatten(2)
        # transport the channels to the last 
        embeddings = embeddings.transpose(1,2)
        # add position embedding to each patch，给出图片位置信息。 
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        return embeddings   

class SiglipVisionTransformer(nn.Module):
    def __init__ (self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size 

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)

    def forward(self,pixel_values:torch.Tensor) -> torch.Tensor:
        # pixel_values[Batch_Size,Channels,Height,Width] -> [Batch_Size,Num_Patches,Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)        


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.vision_model = SiglipVisionTransformer(config)

        def forward(self,pixel_values)-> Tuple:
            # [Batch_Size,Channels,Height,Width] -> [Batch_Size,Num_Patches,Embed_Dim]
            return self.vision_model(pixel_values= pixel_values)
        
