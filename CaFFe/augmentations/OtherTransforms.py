from torch import nn

"""
    Filler Identity functions
"""


class NoTransform(nn.Module):
    def __init__(self):
        super(NoTransform, self).__init__()

    def forward(self, x):
        return x


class NoTransform2(nn.Module):
    def __init__(self):
        super(NoTransform2, self).__init__()

    def forward(self, x, y, **kwargs):
        return x, y


class NoTransform4(nn.Module):
    def __init__(self):
        super(NoTransform4, self).__init__()

    def forward(self, x, y, z, w, **kwargs):
        return x, y, z, w
