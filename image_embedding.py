import torch
import torch.nn as nn
import torchvision

from utils import l2norm


class ImageEmbedder(nn.Module):
    def __init__(self, resnet_location, l2_norm=False):
        super(ImageEmbedder, self).__init__()

        self.l2_norm = l2_norm

        if resnet_location == 'pretrained':
            resnet = torchvision.models.resnet152(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        else:
            resnet = torchvision.models.resnet152()
            resnet = torch.nn.DataParallel(resnet)
            checkpoint = torch.load(resnet_location)
            resnet.load_state_dict(checkpoint['state_dict'])

            # Our resnet models were trained with DataParallel, hence the resnet.module.children()
            self.feature_extractor = nn.Sequential(*list(resnet.module.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Linear projection to the joint embedding space
        self.linear = nn.Linear(2048, 512, bias=False)

    def forward(self, input):
        features = self.feature_extractor(input)

        output = self.linear(features.squeeze())

        if self.l2_norm:
            output = l2norm(output)

        return output
