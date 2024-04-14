import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchtext.data.functional import to_map_style_dataset
import torchtext.datasets as datasets
import spacy
from torchtext.vocab import build_vocab_from_iterator
from os.path import exists
from OriginalTransformer.modules import Batch
from OriginalTransformer.loss import LabelSmoothing

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        self.load_data()
        self.load_tokenizers()
        self.determine_vocab_from_datasets()

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return (Batch(b[0], b[1], self.english_vocab["<blank>"]) for b in DataLoader(self.train_iter, batch_size=self.batch_size, shuffle=True)),

    def val_dataloader(self):
        return (Batch(b[0], b[1], self.english_vocab["<blank>"]) for b in DataLoader(self.valid_iter, batch_size=self.batch_size, shuffle=True)),

    def load_data(self):
        self.train_iter, self.valid_iter, self.test_iter = datasets.Multi30k(
            language_pair=("de", "en")
        )

    def load_tokenizers(self):
        try:
            self.german_tokenizer = spacy.load("de_core_news_sm")
        except IOError:
            os.system("python -m spacy download de_core_news_sm")
            self.german_tokenizer = spacy.load("de_core_news_sm")
        try:
            self.english_tokenizer = spacy.load("en_core_web_sm")
        except IOError:
            os.system("python -m spacy download en_core_web_sm")
            self.english_tokenizer = spacy.load("en_core_web_sm")

    def determine_vocab_from_datasets(self):
        if not exists("vocab.pt"):
            self.german_vocab = build_vocab_from_iterator(
                yield_tokens(self.train_iter + self.valid_iter + self.test_iter, self.german_tokenizer, index=0),
                min_freq=2,
                specials=["<s>", "</s>", "<blank>", "<unk>"],
            )
            self.english_vocab = build_vocab_from_iterator(
                yield_tokens(self.train_iter + self.valid_iter + self.test_iter, self.english_tokenizer, index=1),
                min_freq=2,
                specials=["<s>", "</s>", "<blank>", "<unk>"],
            )
            torch.save((self.german_vocab, self.english_vocab), "vocab.pt")
        else:
            self.german_vocab, self.english_vocab = torch.load("vocab.pt")

    def return_label_smoothing_criterion(self):
        return LabelSmoothing(size=len(self.english_vocab), padding_idx = self.english_vocab["<blank>"], smoothing=0.1)

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]

def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenize(from_to_tuple[index], tokenizer)

if __name__ == "__main__":
    dm = DataModule()
    dm.prepare_data()
