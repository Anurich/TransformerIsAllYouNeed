import torch.nn as nn
from dataclasses import dataclass
from transformers import BertForSequenceClassification as HF_BertForSequenceClassification

import torch
@dataclass
class config:
    vocab_size = 30522
    block_size = 512
    embedding_dim = 768
    n_layer = 12
    n_heads =  12 
    token_type_embedding = 2
    num_of_expert_per_token= 2
    total_number_of_epxerts= 6


class BertAttention(nn.Module):
    def __init__(self):
        super().__init__()

        assert config.embedding_dim%config.n_heads == 0, f"Please check Embedding dimension or n_heads division should be equal to 0"
        self.attention = nn.ModuleDict({
            "query":nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim, bias=True),
            "key": nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim, bias=True),
            "value": nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim, bias=True),
            "dropout": nn.Dropout(p=0.1, inplace=False),
            "output": nn.ModuleDict({
                "dense": nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim, bias=True),
                "LayerNorm":nn.LayerNorm(config.embedding_dim,  eps=1e-12, elementwise_affine=True),
                "dropout": nn.Dropout(p=0.1, inplace=False)
            })
        })


        self.intermediate = nn.ModuleDict({
            "dense": nn.Linear(in_features=config.embedding_dim, out_features=3072, bias=True),
            "intermediate_act_fn": nn.GELU()
        })

        self.output = nn.ModuleDict({
            "dense": nn.Linear(in_features=3072, out_features=config.embedding_dim, bias=True),
            "LayerNorm": nn.LayerNorm((config.embedding_dim,), eps=1e-12, elementwise_affine=True),
            "dropout": nn.Dropout(p=0.1, inplace=False)
        })

    def forward(self, x):
        # we first need to get the size of X
        B, T, C = x.size() # batch, maxlength, embedding dimension
        
        Q = self.attention["query"](x)
        K = self.attention["key"](x)
        V = self.attention["value"](x)

        # we need to split this into multi head 
        Q = Q.view(B, T, config.n_heads, C//config.n_heads).transpose(1,2) # B, Head, T, C
        K = K.view(B, T, config.n_heads, C//config.n_heads).transpose(1,2) # B, Head, T, C
        V = V.view(B, T, config.n_heads, C//config.n_heads).transpose(1,2) # B, Head, T, C

        # performing the self attetion 
        QK =  torch.matmul(Q, K.transpose(-1,-2))/torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float32))
        # now we  need to compute the score 
        score =  torch.matmul(torch.softmax(QK, dim=-1),V) # B, HEAD, T, C
        # now we need to put the score back to original form 
        output = self.attention["dropout"](score.transpose(1,2).contiguous().view(B, T, C)) # B, HEAD, C

        output = self.attention.output["dense"](output)
        x =  x+ self.attention.output["LayerNorm"](output) # residual conntection
        output = self.attention.output["dropout"](x)

        output = self.intermediate["dense"](output)
        output = self.intermediate["intermediate_act_fn"](output)

        output = self.output["dense"](output)
        x = x + self.output["LayerNorm"](output) # residual connection
        output = self.output["dropout"](x)


        return output




class BertForSequenceClassification(nn.Module):
    def __init__(self, total_class = 2) -> None:
        super().__init__()

        self.embeddings = nn.ModuleDict({
            "word_embeddings": nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim, padding_idx=0),
            "position_embeddings": nn.Embedding(num_embeddings=config.block_size, embedding_dim=config.embedding_dim),
            "token_type_embeddings": nn.Embedding(num_embeddings=config.token_type_embedding, embedding_dim=config.embedding_dim),
            "LayerNorm": nn.LayerNorm(normalized_shape=config.embedding_dim,eps=1e-12, elementwise_affine=True),
            "dropout": nn.Dropout(p=0.1,inplace=False)
        })

        self.encoder = nn.ModuleDict({
            "layer": nn.ModuleList([BertAttention() for _ in range(config.n_layer)])
        })


        self.BertPooler = nn.ModuleDict({
            "dense": nn.Linear(in_features=config.embedding_dim, out_features=config.embedding_dim, bias=True),
            "activation": nn.Tanh()
        })

        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.classifier = nn.Linear(in_features=config.embedding_dim, out_features=total_class) 

    def forward(self, x, token_type_ids, last_hidden_state=True):
        B, T = x.size()
        
        token_embeddings = self.embeddings["word_embeddings"](x)
        positional_range = torch.arange(0, T, dtype=torch.long, device=x.device)
        positional_embeddings = self.embeddings["position_embeddings"](positional_range)
        token_type_embeddings = self.embeddings["token_type_embeddings"](token_type_ids)

        # we sum it up 
        input_to_self_attention = token_embeddings + positional_embeddings + token_type_embeddings
        input_to_self_attention = self.embeddings["dropout"](self.embeddings["LayerNorm"](input_to_self_attention))
        # we pass through attention block 12 times 
        hidden_states = []
        for layer in self.encoder["layer"]:
            input_to_self_attention = layer(input_to_self_attention)
            if last_hidden_state:
                hidden_states.append(input_to_self_attention)

        pooled_output = self.BertPooler["dense"](input_to_self_attention[:, 0])
        pooled_output = self.BertPooler["activation"](pooled_output)

        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def load_pretrained_model_weight_to_custom_model():
    model_hf = HF_BertForSequenceClassification.from_pretrained("bert-base-uncased")
    custom_model = BertForSequenceClassification()
    pretrained_dict = model_hf.state_dict()
    custom_dict    = custom_model.state_dict()
    pretrained_keys = list(pretrained_dict.keys())
    custom_keys = list(custom_dict.keys())
    assert len(pretrained_keys) == len(custom_keys)

    weight_transfer_dict = dict()
    for key, val in pretrained_dict.items():
        if key.replace("bert.", "").replace("self.","") in custom_keys:
            weight_transfer_dict[key.replace("bert.", "").replace("self.","")] = val

    custom_dict.update(weight_transfer_dict)
    custom_model.load_state_dict(custom_dict)

    return custom_model


