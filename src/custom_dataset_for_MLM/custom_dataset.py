from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
from datasets import Dataset
class customDataset:
    def __init__(self, data_path) -> None:
        data = open(data_path, "r").read()
        sentences = sent_tokenize(data)
        chunked_data = [{"text": sentence}for sentence in sentences]
        self.train_data   = Dataset.from_list(chunked_data)



class tokenization(customDataset):
    def __init__(self, dataset_path, tokenizer) -> None:
        super().__init__(dataset_path)
        self.tokenizer = tokenizer
        self.toknized_data = self.train_data.map(self.tokenization_step, batched=True, batch_size=12)

    def tokenization_step(self, examples):
        return self.tokenizer(examples["text"],max_length=128, padding="max_length", \
                              truncation=True,return_tensors="pt")
    