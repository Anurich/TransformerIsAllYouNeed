import torch.nn  as nn
from model.replicate_bert.bert import config
import torch

class ExpertArchitecture(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, 4*config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim*4, config.embedding_dim),
            nn.Dropout()
        )
    def forward(self,x):
        return self.net(x)



class MOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_of_expert_per_token = config.num_of_expert_per_token
        self.gating_network = nn.Linear(in_features=config.embedding_dim, out_features=config.total_number_of_epxerts, bias=False)
        self.experts = nn.ModuleList([ExpertArchitecture(config) for _ in range(config.total_number_of_epxerts)])
    
    def forward(self, x):
        B, T, C = x.size() # Batch, seq_length, Channel 
        x = x.view(-1, C)
        
        # we pass this  to gating network but we will add the noise to our x 
        random_noise = torch.rand_like(x)
        noisy_x = random_noise + x
        
        # pass this through the gating network 
        gated_output = self.gating_network(noisy_x) # return B*T, number of expert to pick from 
        gated_probability = torch.softmax(gated_output, dim=1, dtype=torch.float)
        # we need to select the top k expert to which we wanted to send the input 
        gated_probability, selected_index = torch.topk(gated_probability, k=config.num_of_expert_per_token,dim=-1)
        # normalised the gated prob
        gated_probability /= gated_probability.sum(dim=-1, keepdim=True)
        final_output = torch.zeros((B*T, C), dtype=x.dtype)
        # now we create a one hot encode
        # given that we have selected index 
        # where each row represent a token and also suggest the expert we need to choose for that token
        # so we creat the mask where each row is one hot encoded based on the selected index
        mask = torch.nn.functional.one_hot(selected_index, num_classes=config.total_number_of_epxerts)

        for expert_idx in range(config.total_number_of_epxerts):
            expert = self.experts[expert_idx]
            idx, top_x =torch.where(mask[expert_idx]) # select the idx where the value is non-zero

            current_state = x[None, top_x].view(-1, C)
            current_states = expert(current_state) * gated_probability[top_x, idx, None]
            final_output.index_add_(0, top_x, current_states.to(x.dtype))
        
        outputs = final_output.reshape(B,T,C)
        return outputs

