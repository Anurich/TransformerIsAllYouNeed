from src.custom_dataset_for_MLM.custom_dataset import tokenization
from transformers import  DataCollatorForLanguageModeling
from src.model.bert_with_moe.bert_with_MOE import BertForSequenceClassificationMOE
from transformers import PreTrainedTokenizerFast

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from src.model.train_tokenizer.tokenizer import TrainTokenizerBert

data_path = "src/data/romeo_juliet.txt"
custom_tokenization = TrainTokenizerBert(train_corpus=data_path)
custom_tokenization.fit()
tokenizer = PreTrainedTokenizerFast.from_pretrained("src/data/save_tokenizer_bert/bert_bpe.json", 
                                                    unk_token="[UNK]",
                                                    pad_token="[PAD]",
                                                    cls_token="[CLS]",
                                                    sep_token="[SEP]",
                                                    mask_token="[MASK]"
                                                    )
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
bert_moe = BertForSequenceClassificationMOE(type="masked")

custom_dataset = tokenization(dataset_path=data_path, tokenizer=tokenizer)
dataset = custom_dataset.toknized_data.select_columns(['input_ids', 'token_type_ids', "attention_mask"])
dataset = dataset.train_test_split(test_size=0.2)

train_loader = DataLoader(dataset["train"], batch_size=12, collate_fn=data_collator)
dev_loader = DataLoader(dataset["test"], batch_size=12, collate_fn=data_collator)



epoch = 10
optimizer = torch.optim.Adam(bert_moe.parameters())
loss_fn = torch.nn.CrossEntropyLoss()
for i in tqdm(range(epoch)):
    total_loss = 0
    for train_data in tqdm(train_loader):
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