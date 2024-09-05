import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import torchvision.models as models

class Atrous_Convolution(nn.Module):

    def __init__(
            self, input_channels: int, kernel_size: int, pad, dilation_rate: int,
            output_channels: int=256) -> None:
        super(Atrous_Convolution, self).__init__()

        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=kernel_size, padding=pad,
                              dilation=dilation_rate, bias=False)

        self.batchnorm = nn.BatchNorm2d(output_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
class ASSP(nn.Module):
    """
   Encoder of DeepLabv3+.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Atrous Spatial Pyramid pooling layer
        Args:
            in_channles (int): No of input channel for Atrous_Convolution.
            out_channles (int): No of output channel for Atrous_Convolution.
        """
        super(ASSP, self).__init__()
        self.conv_1x1 = Atrous_Convolution(
            input_channels=in_channels, output_channels=out_channels,
            kernel_size=1, pad=0, dilation_rate=1)

        self.conv_6x6 = Atrous_Convolution(
            input_channels=in_channels, output_channels=out_channels,
            kernel_size=3, pad=6, dilation_rate=6)

        self.conv_12x12 = Atrous_Convolution(
            input_channels=in_channels, output_channels=out_channels,
            kernel_size=3, pad=12, dilation_rate=12)

        self.conv_18x18 = Atrous_Convolution(
            input_channels=in_channels, output_channels=out_channels,
            kernel_size=3, pad=18, dilation_rate=18)

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.final_conv = Atrous_Convolution(
            input_channels=out_channels * 5, output_channels=out_channels,
            kernel_size=1, pad=0, dilation_rate=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1x1 = self.conv_1x1(x)
        x_6x6 = self.conv_6x6(x)
        x_12x12 = self.conv_12x12(x)
        x_18x18 = self.conv_18x18(x)
        img_pool_opt = self.image_pool(x)
        img_pool_opt = F.interpolate(
            img_pool_opt, size=x_18x18.size()[2:],
            mode='bilinear', align_corners=True)
        concat = torch.cat(
            (x_1x1, x_6x6, x_12x12, x_18x18, img_pool_opt),
            dim=1)
        x_final_conv = self.final_conv(concat)
        return x_final_conv
    
class ResNet_50(nn.Module):
    def __init__(self, output_layer: str=None) -> None:
        """
        ResNet-50 model with pretrained weights up to the specified output layer.
        """
        super(ResNet_50, self).__init__()
        self.pretrained = models.resnet50(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

        # Modify the first convolutional layer to accept 12 channels
        # Get the original conv1 layer
        original_conv1 = self.net.conv1

        # Create a new conv1 layer with 12 input channels
        self.net.conv1 = nn.Conv2d(
            in_channels=12,       # Change input channels from 3 to 12
            out_channels=original_conv1.out_channels,  # Keep the same number of output channels
            kernel_size=original_conv1.kernel_size,    # Keep the same kernel size (7x7)
            stride=original_conv1.stride,              # Keep the same stride
            padding=original_conv1.padding,            # Keep the same padding
            bias=original_conv1.bias is not None       # If the original conv1 had bias, keep it
        )
        with torch.no_grad():
            # Assign the weights from the pretrained ResNet-50's conv1 to the appropriate satellite channels
            # Map: RGB (pre-trained conv1) -> BGR in your satellite image
            self.net.conv1.weight[:, 3] = original_conv1.weight[:, 0]  
            self.net.conv1.weight[:, 2] = original_conv1.weight[:, 1]  
            self.net.conv1.weight[:, 1] = original_conv1.weight[:, 2]  
            
            # Initialize the rest of the channels (indices 4 to 11) with random or custom initialization
            nn.init.kaiming_normal_(self.net.conv1.weight[:, 4:], mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.net.conv1.weight[:, 0], mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    

class Deeplabv3Plus(nn.Module):
    def __init__(self, num_classes: int) -> None:

        super(Deeplabv3Plus, self).__init__()

        self.backbone = ResNet_50(output_layer='layer3') # high level features

        self.low_level_features = ResNet_50(output_layer='layer1') # low level features

        self.assp = ASSP(in_channels=1024, out_channels=256)

        self.conv1x1 = Atrous_Convolution(
            input_channels=256, output_channels=48, kernel_size=1,
            dilation_rate=1, pad=0)

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifer = nn.Conv2d(256, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_backbone = self.backbone(x)
        x_low_level = self.low_level_features(x)
        x_assp = self.assp(x_backbone)
        x_assp_upsampled = F.interpolate(
            x_assp, scale_factor=(4, 4),
            mode='bilinear', align_corners=True)
        x_conv1x1 = self.conv1x1(x_low_level)
        x_cat = torch.cat([x_conv1x1, x_assp_upsampled], dim=1)
        x_3x3 = self.conv_3x3(x_cat)
        x_3x3_upscaled = F.interpolate(
            x_3x3, scale_factor=(4, 4),
            mode='bilinear', align_corners=True)
        x_out = self.classifer(x_3x3_upscaled)
        return x_out