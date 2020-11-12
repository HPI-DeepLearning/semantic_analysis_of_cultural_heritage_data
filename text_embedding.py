import torch
import torch.nn as nn
from utils import l2norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextEmbedder(nn.Module):
    def __init__(self, vocab_dim, l2_norm=False):
        super(TextEmbedder, self).__init__()

        self.l2_norm = l2_norm

        # Linear projection to the joint embedding space
        self.linear = nn.Linear(vocab_dim, 512, bias=False)

    def forward(self, input):
        output = self.linear(input)

        if self.l2_norm:
            output = l2norm(output)

        return output
