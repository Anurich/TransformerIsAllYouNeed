# Transformer Is All You Need.

## Overview

This repository contains code for two key projects:

1. **Encoder-Decoder Architecture**: Implements the Transformer architecture described in "Attention Is All You Need."
2. **Custom BERT Replica**: A customized version of the BERT-base-uncased model that loads pre-trained weights from Hugging Face.

## Folder Structure

### Encoder-Decoder Architecture

- **Path**: `encoder_decoder/`
- **Description**: Contains implementations of the Transformer model's encoder-decoder architecture. This is based on the seminal paper *"Attention Is All You Need"*. 

  **Key Components:**
  - `encoder.py`: Contains the implementation of the encoder module.
  - `decoder.py`: Contains the implementation of the decoder module.
  - `attention.py`: Implements the attention mechanisms used in the model.
  - `train.py`: A script for training the model.

### Custom BERT Replica

- **Path**: `custom_bert/`
- **Description**: Contains a custom replica of the BERT-base-uncased model. This custom model is built on top of the original BERT architecture, with modifications as per specific requirements.

  **Key Components:**
  - `custom_bert.py`: Implements the custom BERT model architecture.
  - `load_pretrained.py`: Script to load pre-trained weights from Hugging Face into the custom BERT model.
  - `train.py`: A script for training the custom BERT model on specific datasets.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/repository.git
