# OriginalTransformer
My implementation of the original transformer in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Overview of project intent

I'm drawing heavily on [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) blogpost. I'm also reading [Aleksa Gordić's implementation](https://github.com/gordicaleksa/pytorch-original-transformer/tree/main).

For the first few commits, there will be shameless copy and pasting. The intention is to perform the following in this project:

1. Understand the original transformer, piece by piece. I'll be copying and pasting, but diving deeply into the intent of each function/class.
2. Get the transformer working, most likely through a translation task of some kind.
3. Practice coding the entire thing from scratch from memory, using the final translation task as a "unit test" of sorts. 
   4. This will likely be done by coding individual components from scratch first.
4. Once I can rewrite the entire program from memory three times over three days, I will consider my comprehension mastered.

## Walking through my understanding

This is if you want to follow my logic of development, not just be overwhelmed with the end result. I left detailed notes in each file, recording what I understood as I understood it.

1. Layer Normalization (LayerNorm.py)
2. Sublayers (SublayerUnit.py)
3. Attention (attention.py)
4. MultiHeadAttention (MultiHeadAttention.py)


