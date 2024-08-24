from src.custom_dataset_for_MLM.custom_dataset import tokenization
from transformers import  DataCollatorForLanguageModeling
from src.model.replicate_bert.bert_with_MOE import BertForSequenceClassificationMOE
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
data_path = "model/data/romeo_juliet.txt"
custom_dataset = tokenization(data_path)


data_collator = DataCollatorForLanguageModeling(tokenizer=custom_dataset.tokenizer, mlm=True, mlm_probability=0.15)
bert_moe = BertForSequenceClassificationMOE(type="masked")

dataset = custom_dataset.toknized_data.select_columns(['input_ids', 'token_type_ids', "attention_mask"])
dataset = dataset.train_test_split(test_size=0.2)

train_loader = DataLoader(dataset["train"], batch_size=12, collate_fn=data_collator)
dev_loader = DataLoader(dataset["test"], batch_size=12, collate_fn=data_collator)


epoch = 10
optimizer = torch.optim.Adam(bert_moe.parameters())
loss_fn = torch.nn.CrossEntropyLoss()
for i in tqdm(range(epoch)):
    total_loss = 0
    for train_data in train_loader:
        optimizer.zero_grad()
        input_ids = train_data["input_ids"]
        token_type_ids = train_data["token_type_ids"]
        labels = train_data["labels"]

        output = bert_moe(x=input_ids, token_type_ids=token_type_ids, labels=labels)
        loss_fn  = output["loss"]
        
        total_loss += loss_fn.item()
        loss_fn.backward()
        optimizer.step()
    

    if i % 5 ==0:
        print("Train Loss: ",total_loss/len(train_loader))

# for testing
with torch.no_grad:
    bert_moe.eval()
    total_dev_loss = 0.0
    for dev_data in dev_loader:
        input_ids = dev_data["input_ids"]
        token_type_ids = dev_data["token_type_ids"]
        labels = dev_data["labels"]
        output = bert_moe(x=input_ids, token_type_ids=token_type_ids, labels=labels)
        loss_fn  = output["loss"]
        total_dev_loss += loss_fn.item()
    
    print("Dev Loss: ", total_dev_loss/len(dev_loader))