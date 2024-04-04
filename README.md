# OriginalTransformer
My implementation of the original transformer in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

I'm drawing heavily on [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) blogpost.

## Walking through my understanding

This is if you want to follow my logic of development, not just be overwhelmed with the end result. I left detailed notes in each file, recording what I understood as I understood it.

1. Layer Normalization (LayerNorm.py)
2. Sublayers (SublayerUnit.py)
3. Attention (attention.py)
