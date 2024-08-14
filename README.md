# Transformer Is All You Need.

## Overview

This repository contains code for two key projects:

1. **Encoder-Decoder Architecture**: Implements the Transformer architecture described in "Attention Is All You Need."
2. **Custom BERT Replica**: A customized version of the BERT-base-uncased model that loads pre-trained weights from Hugging Face.

## Folder Structure

### Encoder-Decoder Architecture

- **Path**: `model/Encoder_decoder/`
- **Description**: Contains implementations of the Transformer model's encoder-decoder architecture. This is based on the seminal paper *"Attention Is All You Need"*. 

  **Key Components:**
  - `encoder.py`: Contains the implementation of the encoder module.
  - `decoder.py`: Contains the implementation of the decoder module.

### Custom BERT Replica

- **Path**: `replicate_bert/`
- **Description**: Contains a custom replica of the BERT-base-uncased model. This custom model is built on top of the original BERT architecture, with modifications as per specific requirements.

  **Key Components:**
  - `bert.py`: Implements the custom BERT model architecture. Script to load pre-trained weights from Hugging Face into the custom BERT model.

### transformer.py 
It contain the code to call the Encoder-decoder architecture
## Installation

### Code
```python
    from model.Encoder_decoder import Encoder, config
    from model.Encoder_decoder import Decoder
    import torch.nn as nn
    from transformers import AutoTokenizer
    from model.replicate_bert import bert
    from transformers import BertTokenizer
    import torch

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


    ## For custom Bert Implementation 

    custom_model = bert.load_pretrained_model_weight_to_custom_model()
    print("Weight loaded sucessfully !")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    logits = custom_model(encoded_input["input_ids"], encoded_input["token_type_ids"])
    print("Custom Model Weight Loaded from HF: ", torch.softmax(logits, -1))
```
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Anurich/TransformerIsAllYouNeed.git