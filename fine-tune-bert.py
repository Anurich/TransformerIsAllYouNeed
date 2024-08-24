from model.custom_dataset_for_MLM.custom_dataset import tokenization
from transformers import  DataCollatorForLanguageModeling
# from model.replicate_bert.bert_with_MOE import BertForSequenceClassificationMOE
from model.replicate_bert.bert_with_MOE import BertForSequenceClassificationMOE
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
data_path = "model/data/romeo_juliet.txt"
custom_dataset = tokenization(data_path)



data_collator = DataCollatorForLanguageModeling(tokenizer=custom_dataset.tokenizer, mlm=True, mlm_probability=0.15)
bert_masked = BertForSequenceClassificationMOE(type="masked")

dataset = custom_dataset.toknized_data.select_columns(['input_ids', 'token_type_ids', "attention_mask"])
dataset = dataset.train_test_split(test_size=0.2)

train_loader = DataLoader(dataset["train"], batch_size=12, collate_fn=data_collator)
dev_loader = DataLoader(dataset["test"], batch_size=12, collate_fn=data_collator)


epoch = 10
optimizer = torch.optim.Adam(bert_masked.parameters())
loss_fn = torch.nn.CrossEntropyLoss()
for i in tqdm(range(epoch)):
    total_loss = 0
    for train_data in train_loader:
        optimizer.zero_grad()
        input_ids = train_data["input_ids"]
        token_type_ids = train_data["token_type_ids"]
        labels = train_data["labels"]

        output = bert_masked(x=input_ids, token_type_ids=token_type_ids, labels=labels)
        loss_fn  = output["loss"]
        
        total_loss += loss_fn.item()
        loss_fn.backward()
        optimizer.step()
    

    if i % 5 ==0:
        print(total_loss/len(train_loader))