import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import List, Dict, Union, Tuple
class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, padding: int=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        pre_pooled = F.relu(x)
        x = self.maxpool(pre_pooled)
        return x, pre_pooled

class BaseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, padding: int=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x
class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, padding: int=1) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1) # across the channels dimension
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x
class UNet(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: List[int], 
                 num_base_blocks: int,
                 base_block_kwargs: Dict[str, Union[str,int]],
                 num_encoder_blocks: int,
                 encoder_block_kwargs: Dict[str, Union[str,int]]={},
                 decoder_block_kwargs: Dict[str, Union[str,int]]={}) -> None:
        super().__init__()
        self.encoders = nn.ModuleList()
        for i in range(num_encoder_blocks):
            encoder_in_channels = in_channels if i == 0 else out_channels[i-1]
            self.encoders.append(EncoderBlock(encoder_in_channels, out_channels[i], **encoder_block_kwargs))

        self.base_blocks = nn.ModuleList()
        base_block_channels = base_block_kwargs['out_channels']
        for i in range(num_base_blocks):
            base_block_in_channels = out_channels[-1] if i == 0 else base_block_channels
            self.base_blocks.append(BaseBlock(base_block_in_channels, base_block_channels))

        self.decoders = nn.ModuleList()
        for i in range(num_encoder_blocks):
            decoder_in_channels = out_channels[-i] if i!=0 else base_block_channels
            self.decoders.append(DecoderBlock(decoder_in_channels, out_channels[-i-1], **decoder_block_kwargs))

        self.final_conv = nn.Conv2d(out_channels[0], 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x, pre_pooled = encoder(x)
            skips.append(pre_pooled)
            # print(f'In Encoder: x shape: {x.shape}, skip shape: {pre_pooled.shape}')
        
        for base_block in self.base_blocks:
            x = base_block(x)
            # print(f'In BaseBlock: x shape: {x.shape}')

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[-i-1])
            # print(f'In Decoder: x shape: {x.shape}')
        
        x = self.final_conv(x)
        # x = torch.sigmoid(x)
        return x