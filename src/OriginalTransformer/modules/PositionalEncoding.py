import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    """
    I have written pages in my journal on this subject. On YouTube I recommend the AI Coffee Break with Letitia
        positional encoding videos, as well as CodeEmporium's video on the subject. The fact of the matter is that this
        is a field with much nuance laced with research and debate. What is introduced here is a good method that
        works well for things like language sentences - but is it the best for your application? We shall see.

    With sentences, the meaning of each word is reduced to a vector of numbers. Related words have similar vectors,
        but ultimately in about 512 numbers the computer can generally understand what a word is and how it is used
        in a language. The problem with transformers is that it's so efficient at analyzing sentences all in one go
        that it loses its ability to parse out sequential meaning. "The dog chased the cat" is hard to distinguish
        from "the cat chased the dog," for instance. In this sense, we need to distinguish the words by their order,
        somehow. That's where positional encodings come into play.

    Just like we have a list of 512 numbers that denote what a word means, we can also have 512 numbers that denote
        what it's position means. You may be asking, as I did, why you need 512 numbers to do so. I refer you to the
        first AI Coffee break for a breakdown of why that may be, but in a nutshell, simply listing the order from
        1-300 as the 513th number does not play well with weight matrices. We need to portray position in a way that
        the machine can extract significance from it from several different angles. We need to keep the numbers low and
        easy to parse/multiply by weights, but still be clearly distinguished from each other. And we need to do it in
        a way that complements the word vector meaning (not drown it out).

    The Attention is All You Need paper does this using an alternating sine/cosine method, and is what is implemented
        below. Each sine and cosine gets a longer frequency than the last, leading to a tangled (but patterned) web
        of flow that can stretch into eternity, providing unique, varied and bound values that denote clear sequential
        order. I note again, however, that there are alternatives, like learned positional embeddings - this is a
        field rich in research topics and debate.

    To explain in plain english what this accomplishes, however, let's return to the above example - "the dog chased
        the cat." Each of these words will be given semantic meaning in the form of a 512-number'd vector. However, we
        aregoing to ADD the 512 positional encoding numbers directly to this word vectors, changing their meaning
        slightly. It now becomes, basically, "(first word version of 'the') (second word version of 'dog') (third word
        version of'chased') (fourth word version of 'the') (fifth word version of 'cat')."

    Why add them together and risk changing the meaning directly? Would you not risk overwhelming the semantic meaning
        with the positional meaning, or vice versa? This is a genuine concern, and you can imagine how much time the
        original authors spent fine-tuning their sine/cosine equation to strike the perfect balance between the two.
        The alternative is to cut the BS and concatenate the word and position embedding vectors to make a 1048-long
        vector for each word. However, this comes at a dramatic computational cost. Adding is for the poor trying to
        account for limited resources - concatenation would be for those with too much supercomputing power to care.
        Both can, ultimately, work, and it all depends on your use case. Here, the authors went with adding.

    So that's what this is for. Give it a position like 0, 5 or 300, and it will give you a vector of numbers meant
        to represent the meaning of that position.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)