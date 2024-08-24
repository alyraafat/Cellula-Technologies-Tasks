import torchvision.models as models
from torch import nn
import torch

def download_resnet(resnet_type: str) -> nn.Module:
    if resnet_type == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif resnet_type == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif resnet_type == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif resnet_type == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif resnet_type == 'resnet152':
        model = models.resnet152(pretrained=True)
    else:
        raise ValueError(f"Invalid ResNet type: {resnet_type}")

    return model

def create_resnet(resnet_type: str, num_classes: int) -> nn.Module:
    resnet_model = download_resnet(resnet_type)
    in_features = resnet_model.fc.in_features
    model = nn.Sequential(*list(resnet_model.children())[:-1])
    model = nn.Sequential(
        *list(resnet_model.children())[:-1],
        nn.Flatten(),
        nn.Linear(in_features, num_classes)  
    )
    return model