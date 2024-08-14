import torch
import torch.nn as nn
from pydantic import BaseModel
class config(BaseModel):
    embedd_dim: int = 768
    n_head: int = 6
    n_layer: int = 6
    vocab_size: int=30522
    block_size: int =512
    hidden_dimension:int = 3072


class MultiHeadedAttention(nn.Module):
    def __init__(self, config, mask=None,cross_attention=False):
        super().__init__()
        self.cross_attention=cross_attention
        # create key, query and vector 
        assert config.embedd_dim % config.n_head == 0, f"Embedding dimension and N-head must be divisible"
        self.config = config
        self.mask = mask 
        if cross_attention == False:
            self.key = nn.Linear(config.embedd_dim, config.embedd_dim)
            self.query = nn.Linear(config.embedd_dim, config.embedd_dim)
            self.value = nn.Linear(config.embedd_dim, config.embedd_dim)
        elif cross_attention:
            self.query = nn.Linear(config.embedd_dim, config.embedd_dim)

        
    
    def forward(self,x, key_from_decoder = None, value_from_decoder=None):
        """
            For every given word in the sentence we are trying 
            to find the attention score, So the model can understand
            which word is important to other word with in the sentence
        """
        B,T,C = x.size()
        if self.cross_attention == False:
            k = self.key(x) # B, T, C  where C= 768''' T is the max length,and C is the channel'''
            Q = self.query(x) # B, T, C where C = 768
            V = self.value(x) # B, T, C where C = 768
        else:
            Q = self.query(x)
            k = key_from_decoder
            V = value_from_decoder


        """
        we now need to divide the head into multihead so that 
        we can perform the multi head attention parallely which give 
        more boost to power of transformer.
        """        
        k = k.view(B, T, self.config.n_head, C//self.config.n_head).transpose(1,2) # B, T, 6, 128  - convert to - B, 6, T, 128 because we want to perform the operation on multiple head
        Q = Q.view(B,T, self.config.n_head, C//self.config.n_head).transpose(1,2)
        V = V.view(B, T, self.config.n_head,C//self.config.n_head).transpose(1,2)

        scores = torch.matmul(Q, k.transpose(-2,-1))/torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32)) # B, 6, T, 128

        if self.mask is not None:
            mask = torch.tril(torch.ones(B, self.config.n_head, T, T), diagonal=-1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        output = torch.matmul(torch.softmax(scores, -1), V) # B, N_head, T, 128
        # once  we have the output we need to again move it back to B,T,C
        output = output.transpose(1,2) # B, T, N_head, 128
        output = output.contiguous().view(B, T, C)
        return output




class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.linear1 = nn.Linear(config.embedd_dim, config.hidden_dimension)
        self.linear2 = nn.Linear(config.hidden_dimension, config.embedd_dim)
        self.relu = nn.GELU() # not using the approximation one because it's almost same as
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.relu(self.dropout(self.linear1(x)))
        x = self.dropout(self.linear2(x))
        return x


class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.atten = MultiHeadedAttention(config)
        self.layer_norm = nn.LayerNorm(config.embedd_dim)
        self.mlp = MLP(config)
        self.layer_norm1  = nn.LayerNorm(config.embedd_dim)

    def forward(self, x):
        attn = self.atten(x)
        x = x + self.layer_norm(attn)
        mlp_output = self.mlp(x)
        x = x + self.layer_norm1(mlp_output)
        return x 

class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoder_transformer = nn.ModuleDict({
            'word_embeddings': nn.Embedding(config.vocab_size, config.embedd_dim),
            'position_embeddings': nn.Embedding(config.block_size, config.embedd_dim),
            'layer': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
    
        self.pool_linear = nn.Linear(in_features=config.embedd_dim, out_features=config.embedd_dim)
        self.activation  = nn.Tanh()

    def forward(self, x):
        B, T = x.size()
        word_embedding = self.encoder_transformer["word_embeddings"](x)
        position_range = torch.arange(0, T, dtype=torch.long, device=x.device)
        position_embedding = self.encoder_transformer["position_embeddings"](position_range)
        input_to_block = word_embedding + position_embedding
        for block in self.encoder_transformer["layer"]:
            input_to_block = block(input_to_block)
        #"pool" the model by simply taking the hidden state corresponding
        

        pooled_output = input_to_block[:, 0] 

        pooled_output = self.pool_linear(pooled_output)
        pooled_activation = self.activation(pooled_output)
        
        return pooled_activation, input_to_block
