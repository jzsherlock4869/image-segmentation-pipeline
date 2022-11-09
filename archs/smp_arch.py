import torch
import torch.nn as nn
# import segmentation_models_pytorch as smp

import importlib

class SMPArch(nn.Module):
    """
        Use the smp library to construct arch
        simple modify in this class if necessary 
    """
    def __init__(self, backbone='UnetPlusPlus', **kwargs):
        super(SMPArch, self).__init__()
        self.arch = getattr(importlib.import_module('segmentation_models_pytorch'), backbone)(**kwargs)

    def forward(self, x):
        out = self.arch(x)
        return out

if __name__ == "__main__":
    
    arch = SMPArch(backbone='UnetPlusPlus',
                encoder_name='resnet34',
                encoder_weights="imagenet",
                decoder_attention_type=None,
                in_channels=3,
                classes=6)

    dummy_input = torch.rand(1, 3, 64, 64)
    arch.eval()
    output = arch(dummy_input)
    print(output.size())