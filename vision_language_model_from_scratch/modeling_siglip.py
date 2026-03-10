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
    
class SigLipAttention(nn.Module):
    "multi-headed attention from 'Attention is all you need' paper"

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size 
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads 
        self.scale = self.embed_dim**-0.5 # 1/sqrt(d_k) , d_k is the dimension of the key vector
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim,self.embed_dim) 

    def forward(self,hidden_states :torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states : [Batch_size,num_patches,embed_dim]
        batch_size,seq_len,_ = hidden_states.size()
        query_states = self.q_proj(hidden_states) # [Batch_size,num_patches,embed_dim]
        key_states = self.k_proj(hidden_states) # [BatchSize,num_patches,embed_dim]
        value_states = self.v_proj(hidden_states) # [BatchSize,num_patches,
        # query_states: [BatchSize,num_heads,num_patches,head_dim]
        query_states = query_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        

class SiglipMLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config 
        self.fc1 = nn.Linear(config.hidden_size ,config.intermediate_size) # four times the hidden size 
        self.fc2 = nn.Linear(config.intermediate_size,config.hidden_size)
    
    def forward(self,hidden_states:torch.Tensor) -> torch.Tensor:
        # [Batch_size ,num_patches,Embed_dim] -> [Batch_size,num_patches,intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # add the nonlinearity to map the complex function, and relu would cause the negative value to be zero
        # so the gelu would be better than the relu, which can keep the negative value and make the model more powerful
        # the nonliearity function can make the gradient flow better, without force the model to be positive 
        hidden_states = nn.functional.gelu(hidden_states,approximate="tanh")
        # [Batch_size,num_patches,intermediate_size] -> [Batch_size,num_patches,embed_dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
    
class SiglipEncoderLayer(nn.Moudle):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
    
    def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
        # residual :[batch_size,num_patches,embed_dim]
        residual = hidden_states # use the same input for the residual connection 
        # [batch_size,num_patches,embed_dim]-> [batch_size,num_patches,embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [batch_size,num_patches,embed_dim]-> [batch_size,num_patches,embed_dim]
        hidden_states, _ = self.self_attn(hidden_states)
        # the residual connection 
        hidden_states = residual+ hidden_states
        # residual [batch_size,num_patches,embed_dim] 
        residual = hidden_states
        # [batch_size,num_patches,embed_dim]-> [batch_size,num_patches,embed_dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [batch_size,num_patches,embed_dim]-> [batch_size,num_patches,embed_dim]
        hidden_states = self.mlp(hidden_states) # add the parameters and nonlinearity to map the complex function
        # [batch_size,num_patches,embed_dim]
        hidden_states = residual + hidden_states

        return hidden_states 


class SiglipVisionTransformer(nn.Module):
    def __init__ (self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size 

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)

    def forward(self,hidden_states:torch.Tensor) -> torch.Tensor:
        # residual:[batch_size, num_patches,embed_dim]
        residual = hidden_states
        
       


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__() 
        self.config = config 
        self.vision_model = SiglipVisionTransformer(config)

        def forward(self,pixel_values)-> Tuple:
            # [Batch_Size,Channels,Height,Width] -> [Batch_Size,Num_Patches,Embed_Dim]
            return self.vision_model(pixel_values= pixel_values)
        
