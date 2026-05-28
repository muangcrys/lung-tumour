from IPython.core.completerlib import module_list

from models.medicalnet import *
from pathlib import Path
from pretrains.configs import *
import torch.nn as nn
import torch
from typing import Literal
from luna_dataset.dataset import LunaDataset
from luna_dataset.transforms import *
from torch.utils.data.dataloader import DataLoader

def get_pretrained_medicalnet(depth: Literal[18, 50],
                              ckt_path: str|Path|None = None,
                              disable_cuda_loading: bool = None,
                              remove_module_prefix: bool = True) -> nn.Module:
    if depth == 18:
        config = MedicalNet18Config
        model_fn = resnet18
    elif depth == 50:
        config = MedicalNet50Config
        model_fn = resnet50
    else:
        raise ValueError(f"Unsupported depth: {depth}")

    if ckt_path is None:
        ckt_path = config.ckt
    else:
        ckt_path = Path(ckt_path)

    if disable_cuda_loading is None:
        disable_cuda_loading = not torch.cuda.is_available()

    model = model_fn(sample_input_D = 64,
                     sample_input_H = 128,
                     sample_input_W = 128,
                     num_seg_classes= 2,
                     shortcut_type=config.shortcut,)

    if disable_cuda_loading:
        ckt = torch.load(ckt_path, map_location='cpu')
    else:
        ckt = torch.load(ckt_path)
    state_dict = ckt['state_dict']

    if remove_module_prefix:
        new_state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }
    else:
        new_state_dict = state_dict

    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in  new_state_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    return model

def replace_medicalnet_classifier(model: nn.Module,
                                  num_classes: int = 1):
    input_size = model.conv_seg[0].in_channels
    model.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(start_dim=1),
        nn.Linear(input_size, num_classes)
    )

def main():
    print("Testing model loading...")
    print("=" * 20)
    print("Testing Medicalnet-18")
    medicalnet18 = get_pretrained_medicalnet(depth=18)
    print("Successfully loaded Medicalnet-18 with pretrained weights.")
    print("Replacing final layer.")
    replace_medicalnet_classifier(medicalnet18)
    print("Successfully replaced Medicalnet-18 fully connected layer.")
    print(medicalnet18)
    print("=" * 20)
    print("Testing Medicalnet-50")
    medicalnet50 = get_pretrained_medicalnet(depth=50)
    print("Successfully loaded Medicalnet-50 with pretrained weights.")
    print("Replacing final layer.")
    replace_medicalnet_classifier(medicalnet50)
    print("Successfully replaced Medicalnet-50 fully connected layer.")
    print(medicalnet50)
    print("=" * 20)
    print("Loading test data...")

    train_dataset = LunaDataset.get_dataset_with_transform(split='train',
                                                           model='medical_pretrained',)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=16,)
    next_batch, y = next(iter(train_dataloader))

    print("Successfully loaded test data.")
    print("=" * 20)
    print("Testing forward pass...")
    medicalnet18.eval()
    medicalnet50.eval()

    with torch.no_grad():
        medicalnet18(next_batch)
        print("Medicalnet-18 forward pass OK")
        medicalnet50(next_batch)
        print("Medicalnet-50 forward pass OK")


if __name__ == "__main__":
    main()