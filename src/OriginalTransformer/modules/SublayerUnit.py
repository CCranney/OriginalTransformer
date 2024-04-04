class SublayerUnit(nn.Module):
    """
    In a transformer, there are encoder layers and decoder layers. However, each of these layers has sublayers to them.
        For example, encoders have a Multi-Head Attention (MHA) sublayer as well as a normal fully-conn. neural network
        (NN) sublayer. Each of these sublayers, however, goes through the same process. There's a residual connection
        added, plus layer normalization and dropout are applied. This class takes the specific "sublayer" (ie, MHA or NN
        in the case of an encoder) and adds the residual connection, layer norm, and dropout.

    If you look at figure 1 of the original paper you can see these sublayers pretty clearly. Notice how the encoder
        layer shown has an "orange" (MHA) and "blue" (NN) sublayer, while decoders have 2 "orange" and 1 "blue." They
        all have very similar arrows denoting what this class does.

    I'm changing the term from "SublayerConnection" to "SublayerUnit" because that's more intuitive to me. This is an
        encapsulation of everything a sublayer needs, forming a sublayer unit, so to speak.
    """

    def __init__(self, size, dropout):
        super(SublayerUnit, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
