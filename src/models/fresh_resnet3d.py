from pretrains.configs import *
from models.resnet3d import *
from typing import Literal
from torch import nn

def get_fresh_resnet3d(depth: Literal[18, 50] = 18,
                       n_input_channels: int = 1):
    
    if depth == 18:
        config = R3d18Config
    elif depth == 50:
        config = R3d50Config
    else:
        raise ValueError(f"Unsupported depth: {depth}")

    model = generate_model(depth,
                           shortcut_type=config.shortcut,
                           n_input_channels=n_input_channels,
                           n_classes=1,)
    return model