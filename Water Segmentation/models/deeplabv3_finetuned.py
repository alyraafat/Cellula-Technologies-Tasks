import torch
import torchvision

def create_deeplab_resnet_model(num_classes: int, pretrained: bool=True, resnet_type: str='resnet50') -> torch.nn.Module:
    if resnet_type == 'resnet50':
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
    elif resnet_type == 'resnet101':
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
    model.classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

class DeepLabV3Finetuned(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int, pretrained: bool=True, resnet_type: str='resnet50'):
        super(DeepLabV3Finetuned, self).__init__()
        self.in_channels = in_channels
        if in_channels != 3:
            self.conv1 = torch.nn.Conv2d(in_channels, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.model = create_deeplab_resnet_model(num_classes=num_classes, pretrained=pretrained, resnet_type=resnet_type)

    def forward(self, x):
        if self.in_channels != 3:
            x = self.conv1(x)
        x = self.model(x)['out']
        return x
