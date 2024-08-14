from model.Encoder_decoder import Encoder, config
from model.Encoder_decoder import Decoder
import torch.nn as nn
from transformers import AutoTokenizer
class TransformerIsAllYouNeed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = config()
        self.transformer_encoder_block = Encoder(self.config)
        self.transformer_decoder_block = Decoder(self.config)

    
    def forward(self, x):
        _, encoder_block_output = self.transformer_encoder_block(x)
        output = self.transformer_decoder_block(x,encoder_block_output, encoder_block_output)
        return output


input_sentence="""
The key and value used in the cross-attention mechanism of the decoder are indeed taken 
from the output of the encoder block, specifically after the feed-forward network. This ensures that these vectors carry the full contextual and transformed information from the encoder, making them highly informative for the decoder's
attention mechanism.
"""
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded_output = tok(input_sentence, return_tensors="pt")
model= TransformerIsAllYouNeed()
output = model(encoded_output["input_ids"])
print(output.shape)


