from pretrains.configs import *
from models.medicalnet import *
from typing import Literal
from torch import nn

### medicalnet-like classifier models

def get_fresh_medicalnet(depth: Literal[18, 50] = 18,):
    if depth == 18:
        model_fn = resnet18
    elif depth == 50:
        model_fn = resnet50
    elif depth == 34:
        model_fn = resnet34
    else:
        raise ValueError(f"Unsupported depth: {depth}")

    model = model_fn(sample_input_D = 64,
                     sample_input_H = 128,
                     sample_input_W = 128,
                     num_seg_classes= 2,
                     shortcut_type=MedicalNet18Config.shortcut,)
    
    # replace the classifier
    input_size = model.conv_seg[0].in_channels
    model.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(start_dim=1),
        nn.Linear(input_size, 1)
    )

    return model