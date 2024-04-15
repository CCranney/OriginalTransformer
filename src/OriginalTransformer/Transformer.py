from torch import nn
from OriginalTransformer.loss import SimpleLossCompute
from OriginalTransformer.modules import MultiHeadAttention, FeedForwardNetwork, PositionalEncoding, Encoder, EncoderLayer, Decoder, DecoderLayer, Embeddings, Generator
from OriginalTransformer.functions.utils import subsequent_mask
from copy import deepcopy as c
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR

class Transformer(pl.LightningModule):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, labelSmoothingCriterion, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, batch_size=32):
        super(Transformer, self).__init__()
        attn = MultiHeadAttention(h, d_model)
        ff = FeedForwardNetwork(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.batch_size = batch_size
        self.accum_iter = 10
        self.labelSmoothingCriterion = labelSmoothingCriterion
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab_size), c(position))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position))
        self.generator = Generator(d_model, tgt_vocab_size)


    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def training_step(self, batch, batch_idx):
        out = self.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        lossComputation = SimpleLossCompute(self.generator, self.labelSmoothingCriterion)
        loss, loss_node = lossComputation(out, batch.tgt_y, batch.ntokens)
        if batch_idx % 10 == 0:
            optimizer = self.optimizers()
            print(f'---->{batch_idx}, {optimizer.param_groups[0]["lr"]}, {loss}')

        return loss_node

    def validation_step(self, batch, batch_idx):
        out = self.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        lossComputation = SimpleLossCompute(self.generator, c(self.labelSmoothingCriterion))
        loss, loss_node = lossComputation(out, batch.tgt_y, batch.ntokens)
        self.log("val_loss", loss, batch_size=self.batch_size, prog_bar=True)
        print(f'val_loss: {loss}')
        return loss_node

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(
                step, 512, factor=1, warmup=3000
            ),
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": lr_scheduler,
                "interval": "step",
                }
            }

    def on_training_epoch_end(self, outputs):
        epoch_num = self.current_epoch
        print(f"Epoch {epoch_num} completed.")


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    #print(f'{step},{model_size},{factor},{warmup}')
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
