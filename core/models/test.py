import torch
import torchvision
from einops import rearrange

efficientb0 = torchvision.models.efficientnet_b0(pretrained=True)
# for name, parameter in efficientb7.named_parameters():
#     parameter.requires_grad = False
# feature_size = efficientb7.classifier[1].in_features
# print(feature_size)
print(efficientb0)
# print(list(efficientb7.children())[:-1])
