import os
import torch
import torch.nn as nn
import torchvision

from collections import OrderedDict

class DRModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=None):
        super().__init__()
        model = torchvision.models.resnet18()
        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    new_state_dict[k[len('module.encoder_q.'):]] = state_dict[k]
            msg = model.load_state_dict(new_state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            print("=> loaded pre-trained model '{}'".format(pretrained))

            # for name, param in model.named_parameters():
            #     if name not in ['fc.weight', 'fc.bias']:
            #         param.requires_grad = False
            
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.cls = nn.Linear(model.fc.in_features, num_classes)
        self.cls.weight.data.normal_(mean=0.0, std=0.01)
        self.cls.bias.data.zero_()
        self.activation = torch.nn.Sigmoid()

    def forward(self, input):
        features = self.features(input)
        out = self.cls(torch.flatten(features, 1))
        out = self.activation(out)
        return out


