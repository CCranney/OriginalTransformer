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
from torch.nn.functional import pad

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
        return DataLoader(self.train_iter, batch_size=self.batch_size, shuffle=True, collate_fn=self.convert_batch_sentences_to_tokens)

    def val_dataloader(self):
        return DataLoader(self.valid_iter, batch_size=self.batch_size, shuffle=False, collate_fn=self.convert_batch_sentences_to_tokens)

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
            self.german_vocab.set_default_index(self.german_vocab["<unk>"])
            self.english_vocab.set_default_index(self.english_vocab["<unk>"])
            torch.save((self.german_vocab, self.english_vocab), "vocab.pt")
        else:
            self.german_vocab, self.english_vocab = torch.load("vocab.pt")

    def convert_batch_sentences_to_tokens(
            self,
            batch,
            max_padding=72,
            pad_id=2,
    ):
        bs_id = torch.tensor([0], device=torch.device('mps'))  # <s> token id
        eos_id = torch.tensor([1], device=torch.device('mps'))  # </s> token id
        src_list, tgt_list = [], []
        for (_src, _tgt) in batch:
            processed_src = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        self.german_vocab(tokenize(_src, self.german_tokenizer)),
                        dtype=torch.int64,
                        device=torch.device('mps'),
                    ),
                    eos_id,
                ],
                0,
            )
            processed_tgt = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        self.english_vocab(tokenize(_tgt, self.english_tokenizer)),
                        dtype=torch.int64,
                        device=torch.device('mps'),
                    ),
                    eos_id,
                ],
                0,
            )
            src_list.append(
                # warning - overwrites values for negative values of padding - len
                pad(
                    processed_src,
                    (
                        0,
                        max_padding - len(processed_src),
                    ),
                    value=pad_id,
                )
            )
            tgt_list.append(
                pad(
                    processed_tgt,
                    (0, max_padding - len(processed_tgt)),
                    value=pad_id,
                )
            )

        src = torch.stack(src_list)
        tgt = torch.stack(tgt_list)
        return Batch(src, tgt, self.english_vocab["<blank>"])


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
