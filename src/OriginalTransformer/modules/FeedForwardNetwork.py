from torch import nn

class FeedForwardNetwork(nn.Module):
    """
    This is perhaps the most straightforward code addition.

    The multihead attention covers the "attention" part - a global comprehension of what was put in. This part covers
        a more local understanding. It's a simple 2-layer neural network, with the output being the same shape as the
        input (but with a larger representation in between).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
