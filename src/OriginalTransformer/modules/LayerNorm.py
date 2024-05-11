from torch import nn
import torch
class LayerNorm(nn.Module):
    """
    Taken from annotation blog https://nlp.seas.harvard.edu/annotated-transformer/
    I found this AI Assembly YouTube video quite helpful in understanding the concept.
    https://www.youtube.com/watch?v=2V3Uduw1zwQ
    I'm not finding a good description of the a_2 and b_2 variables, but hazarding a guess:
        a_2 is multiplied by the normalization and is originally set to ones (ie, does nothing). I assume it can change
            during backpropogation to have an impact?
        b_2 is added to the end result of a_2 x normalization, originally set to zeros (ie, does nothing). I assume it
            can change during backpropogation to have an impact?
        eps = epsilon so you never divide by 0.
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

