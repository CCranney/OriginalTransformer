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
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
