from tokenizers import ByteLevelBPETokenizer, Tokenizer, models,trainers, normalizers,Regex, pre_tokenizers,processors,decoders
from pathlib import Path
import os 
from tokenizers.processors import BertProcessing

class TrainTokenizerRoberta:
    def __init__(self, train_corpus, path_to_save="src/data/save_tokenizer_roberta/"):
        self.path_to_save = path_to_save
        print(Path(".").glob(train_corpus))
        self.paths = [str(x) for x in Path(".").glob(train_corpus)]
        # Initialize a tokenizer
        self.tokenizer =ByteLevelBPETokenizer(lowercase=True)

    def fit(self):
        # Customize training
        self.tokenizer.train(files=self.paths, vocab_size=20000, min_frequency=2,
                        show_progress=True,
                        special_tokens=[
                                        "<s>",
                                        "<pad>",
                                        "</s>",
                                        "<unk>",
                                        "<mask>",
                    ])
        # Save the trained tokenizer
        self.tokenizer.save_model(self.path_to_save)

    @staticmethod
    def load_tokenizer(path_to_save="src/data/save_tokenizer_roberta/"):
        # Load the tokenizer from the saved file
        tokenizer = ByteLevelBPETokenizer(
            os.path.abspath(os.path.join(path_to_save,'vocab.json')),
            os.path.abspath(os.path.join(path_to_save,'merges.txt'))
        )
        # Prepare the tokenizer
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)
        return tokenizer


class TrainTokenizerBert:
    def __init__(self, train_corpus, path_to_save="src/data/save_tokenizer_bert/bert_bpe.json") -> None:
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        self.tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Replace(Regex(r"[\p{Other}&&[^\n\t\r]]"), ""),
                normalizers.Replace(Regex(r"[\s]"), " "),
                normalizers.Lowercase(),
                normalizers.NFD(), normalizers.StripAccents()]
        )
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        self.trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
        self.train_corpus = train_corpus
        self.path_to_save = path_to_save

    def fit(self):
        self.tokenizer.train([self.train_corpus], trainer=self.trainer)
        cls_token_id = self.tokenizer.token_to_id("[CLS]")
        sep_token_id = self.tokenizer.token_to_id("[SEP]")
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"[CLS]:0 $A:0 [SEP]:0",
            pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
        )
        self.tokenizer.decoder = decoders.WordPiece(prefix="##")

        self.tokenizer.save(self.path_to_save)