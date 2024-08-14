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

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Anurich/TransformerIsAllYouNeed.git
