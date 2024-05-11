import torch
import math

def attention(query, key, value, mask=None, dropout=None):
    """
    This attention equation is at the heart of transformers (hence the title "Attention Is All You Need"). I am going
        to write out my understanding for the purpose of query (q), key (k), and value (v) and why they matter. This
        might be wrong, but it is how I understood it when taking the Deep Learning Coursera course. I will describe
        them in the context of machine translation (say, from French to English).

    I see heads as a version of filters for CNNs. The idea is that, provided
        an input, a head will extract certain features from that input. Rather than visual features (hard lines, red
        to green transition etc.), these pick up on concrete, grammatical meaning of input sentences. What is happening?
        Where? Who is doing it? And so on. In that way, think of a head as a **question** interrogating the meaning of
        the sentence. Multiple heads (covered later) mean the sentence is interrogated with several questions to
        extract it's full purpose and meaning.

    From my course notes:
        In effect, each word is asked the following "questionnaires" by the head:

            (q): What do you want to know about other words in the sentence in relation to the question?
            (k): What do you have to offer other words in the sentence in relation to the question?
            (v): How important are you to the sentence as a whole in relation to the question?

        *In the equation, each word then asks other words the following questions:

            1. How is what you are (the other word's k) compare to what I want to know (my q)?
            2. How does that compare to every other word's answer (softmax "attention")?
            3. How important are you to the sentence? (other word's v)

        These form the basis of the "attention" for any given word in the context of the question asked by the head.

    There are usually several layers involved. I like to think that deeper and deeper meaning is extracted in each
        encoder/decoder, much like a CNN passing through several rounds of filters to identify more abstract
        representations.

    Overall, though, this function is the "equation" (*) described above.

    Note that in this case the query, key and value parameters represent the inputs (x) multiplied by weight matrices
        Q, K and V. Each head has their own Q, K and V.

    I should also note that while x is what is multiplied by these matrices in most cases, THIS IS NOT ALWAYS THE CASE.
        As we'll see in decoders, sometimes you can feed different inputs into the Q, K and V multiplications.

    I HIGHLY recommend you look at https://jalammar.github.io/illustrated-transformer/, even if you feel you already
        get the gist of it. It is helpful to see abstract representations (boxes and arrows), but when you get into
        the weeds you also need to see the matrices and numbers being calculated in a straightforward manner. This
        blog post combines both into a beautiful tapestry.

    SOURCE MASKING ADDITION

    Padding is added to the end of sentences until all sentences in the batch are of the same length. Think of them
        as blank words. When performing attention, we want to make sure that these blank words are NOT treated as
        other words - we want to blank them out, so to speak, and not calculate any attention for them. To explain
        how we're going to do that, note the "scores" variable below.

    scores is the query matrix times the transposed key matrix divided by the square root of the head dimension. In
        effect, it is a square matrix that maps out how much each word/token cares about every other word/token in the
        sentence.

    Imagine, then, an 11x11 matrix as the "attention score" matrix. In this matrix:
        -rows are the "main" token.
        -columns indicate how much attention the "main" token pays to the column token.

    The QxK multiplication makes this matrix initially, but we are about to make it as succint as possible - we will
        apply a softmax so that each row adds up to 1. Basically, we will know on a scale of 0 to 1 how much one
        specific word/token pays to every other word/token.

    We want to mark padded tokens so that all tokens pay
        0% attention to them. So, in the attention score before softmax, we will set columns representing padded tokens
        to -infinity (or negative a really, really large number). When softmax is run, these padded tokens will get
        a big fat 0 in their respective columns.

    Note that you could, technically, also mask out the rows - but this is a waste of time, because no tokens are
        paying any attention to them anyways. YOU CANNOT, however, mask out the rows without also masking out the
        columns. This would basically say "padded tokens should pay attention to all tokens equally, and all
        non-padding tokens SHOULD TREAT PADDED TOKENS AS MEANINGFUL (give them a non-zero score). Bad idea. Blank
        spaces don't have special meaning in any language.

    DECODER MHA ADDITION

    In the source masking addition above, I mentioned how the attention matrix is square. This may be true for the
        encoder, but for the second MHA in the decoder this could be different. In that case, the key and value
        matrices actually come from the encoder, and therefore have dimensions related to the number of tokens provided
        in each sample of a batch. The query, however, has dimensions related to the number of tokens that have been
        heretofore generated by the decoder. Ultimately, the same math applies, and the output winds up being the same
        shape as the query (ie, relating to the number of dimensions of the decoder input).

    I'd imagine, then, you'd want to review the 11x11 matrix row vs column breakdown. It still applies - the "main"
        token just comes from the decoder input, the column token comes from the encoder input.
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
