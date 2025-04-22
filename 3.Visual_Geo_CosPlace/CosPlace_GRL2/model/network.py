
import torch
import logging
import torchvision
from torch import nn
from model.layers import revgrad

from model.layers import Flatten, L2Norm, GeM


CHANNELS_NUM_IN_LAST_CONV = {
        "resnet18": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
        "vgg16": 512,
    }



class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)

class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone, fc_output_dim,alpha,GRL_output =False):
        super().__init__()
        self.GRL_output =GRL_output

        self.backbone, features_dim = get_backbone(backbone)
        self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
                L2Norm()
            )
       
       
        self.GRL_aggregation = nn.Sequential(   
                L2Norm(),
                GeM(),            
                Flatten(),
                GradientReversal(alpha),
                nn.Linear(features_dim, 128),
                nn.Sigmoid(),
                nn.Linear(128, 2) 
                            
            )
        
    
    def forward(self, x):
        x = self.backbone(x)       
        if self.GRL_output ==False:
                  return  self.aggregation(x)

        if self.GRL_output:
          
          return  self.aggregation(x), self.GRL_aggregation(x)


           


def get_backbone(backbone_name):
    if backbone_name.startswith("resnet"):
        if backbone_name == "resnet18":
            backbone = torchvision.models.resnet18(pretrained=True)
        elif backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=True)
        elif backbone_name == "resnet101":
            backbone = torchvision.models.resnet101(pretrained=True)
        elif backbone_name == "resnet152":
            backbone = torchvision.models.resnet152(pretrained=True)
        
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "vgg16":
        backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
    backbone = torch.nn.Sequential(*layers)
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim
