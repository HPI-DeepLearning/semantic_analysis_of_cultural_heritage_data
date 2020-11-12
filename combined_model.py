import torch.nn as nn
from image_embedding import ImageEmbedder
from text_embedding import TextEmbedder


class CombinedModel(nn.Module):
    def __init__(self, vocab_dim, device, resnet, l2_norm=False):
        super(CombinedModel, self).__init__()

        self.image_embedder = ImageEmbedder(resnet, l2_norm=l2_norm).to(device)
        self.text_embedder = TextEmbedder(vocab_dim, l2_norm=l2_norm).to(device)

    def forward(self, text_input, image_input):
        text_output = self.text_embedder(text_input)
        image_output = self.image_embedder(image_input)

        return text_output, image_output