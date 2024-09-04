import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, downsample: nn.Module=None) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        identity = x
        # print(f'identity shape: {identity.shape}, type: {type(identity)}')
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        skip = out
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, skip
    

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: List[int], block: nn.Module, layers: List[int]) -> None:
        super(ResNetEncoder, self).__init__()
        self.in_channels = out_channels[0]
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layers = nn.ModuleList()
        for i in range(len(out_channels)):
            self.layers.append(self._make_layer(block, out_channels[i], layers[i], stride=2))
        

    def _make_layer(self, block: nn.Module, out_channels: int, blocks: int, stride: int=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.conv1(x)
        # print(f'conv1 shape: {x.shape}')
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(f'maxpool shape: {x.shape}')

        skips = []
        for i , layer in enumerate(self.layers):
            for i, block in enumerate(layer):  # Iterate over individual blocks in the layer
                x, skip = block(x)  # Here, x and skip are returned as expected from each ResidualBlock
                # print(f'block {i} shape: {x.shape}, skip shape: {skip.shape}')
                if i==0:
                    skips.append(skip)

        return x, skips

class BaseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, padding: int=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 transpose_kernel_size: int=2, 
                 transpose_stride: int=2,
                 kernel_size: int=3,
                 padding: int=1) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=transpose_kernel_size, stride=transpose_stride)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        # print(f'x shape: {x.shape}, skip shape: {skip.shape}')
        concat_x = torch.cat((x, skip), dim=1)
        x = self.conv(concat_x)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder_channels: List[int], num_classes: int=1) -> None:
        super(Decoder, self).__init__()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_channels)-1):
            in_channels = decoder_channels[i]
            out_channels = decoder_channels[i+1]
            self.decoder_blocks.append(DecoderBlock(in_channels, out_channels))        
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x, skips[i])
        out = self.final_conv(x)
        return out



class ResNetUNet(nn.Module):
    def __init__(self, in_channels: int, resnet_out_channels: List[int], layers: List[int], num_classes: int=1, num_base_blocks: int=2) -> None:
        super(ResNetUNet, self).__init__()
        self.encoder = ResNetEncoder(in_channels=in_channels, out_channels=resnet_out_channels, block=ResidualBlock, layers=layers)
        self.base_blocks = nn.ModuleList()
        base_block_in_channels = resnet_out_channels[-1]
        base_block_out_channels = base_block_in_channels * 2
        for _ in range(num_base_blocks):
            self.base_blocks.append(BaseBlock(base_block_in_channels, base_block_out_channels))
            base_block_in_channels = base_block_out_channels
        decoder_channels = [base_block_out_channels]+resnet_out_channels[::-1]
        self.decoder = Decoder(decoder_channels=decoder_channels, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_out, skips = self.encoder(x)
        base_block_out = encoder_out
        # print(f'encoder_out shape: {encoder_out.shape}')
        for base_block in self.base_blocks:
            base_block_out = base_block(base_block_out)
        # print(f'base_block_out shape: {base_block_out.shape}')
        out = self.decoder(base_block_out, skips[::-1])
        return out

