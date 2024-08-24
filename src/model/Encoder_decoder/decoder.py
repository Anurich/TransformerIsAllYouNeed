import torch.nn as nn
from model.Encoder_decoder.encoder import  MultiHeadedAttention, MLP
import torch

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.masked_attention = MultiHeadedAttention(config, mask="consider")
        self.layer_norm = nn.LayerNorm(config.embedd_dim)
        self.attention =  MultiHeadedAttention(config, mask=None, cross_attention=True)
        self.layer_norm1 = nn.LayerNorm(config.embedd_dim)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(config.embedd_dim)
    
    def forward(self,x, key, value):
        masked_output = self.masked_attention(x)
        masked_output = self.layer_norm(masked_output)
        x = x + masked_output
        attention_output = self.attention(x, key, value)
        attention_output = self.layer_norm1(attention_output)
        x = x + attention_output
        linear_pass = self.mlp(x)
        output = x+ self.layer_norm2(linear_pass)
        return output


class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.decoder_transformer = nn.ModuleDict({
            "output_embeddings": nn.Embedding(config.vocab_size, config.embedd_dim),
            "positional_embeddings": nn.Embedding(config.block_size, config.embedd_dim),
            "DecoderBlock": nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        })

        self.linear_output = nn.Linear(in_features=config.embedd_dim, out_features=config.embedd_dim)
        self.softmax  = nn.Softmax(dim=-1)


    def forward(self, x, key, value):
        B, T = x.size()
        output_encoding = self.decoder_transformer["output_embeddings"](x)
        positional_range = torch.arange(0, T, dtype=torch.long, device=x.device)
        position_encoding = self.decoder_transformer["positional_embeddings"](positional_range)

        decoder_input = output_encoding + position_encoding

        for block in self.decoder_transformer["DecoderBlock"]:
            decoder_input = block(decoder_input, key, value)
        
        
        output = self.softmax(self.linear_output(decoder_input))

        return output


