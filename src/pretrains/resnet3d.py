from models.resnet3d import *
from pathlib import Path
from pretrains.configs import *
import torch.nn as nn
from typing import Literal
from luna_dataset.dataset import LunaDataset
from luna_dataset.transforms import *
from torch.utils.data.dataloader import DataLoader

def get_pretrained_r3d(depth: Literal[18, 50],
                       ckt_path: str|Path|None = None,
                       num_classes: int = None,):
    if depth == 18:
        config = R3d18Config
    elif depth == 50:
        config = R3d50Config
    else:
        raise ValueError(f"Unsupported depth: {depth}")

    if ckt_path is None:
        ckt_path = config.ckt
    else:
        ckt_path = Path(ckt_path)

    if num_classes is None:
        num_classes = config.pretrained_classes

    model = generate_model(depth,
                           shortcut_type=config.shortcut,
                           n_classes=num_classes)
    ckt = torch.load(ckt_path)
    state_dict = ckt['state_dict']
    model.load_state_dict(state_dict)
    return model



def replace_resnet3d_classifier(model, num_classes=1):
    fc_input_size = model.fc.in_features
    model.fc = nn.Linear(fc_input_size, num_classes)


def main():
    print("Testing model loading...")
    print("=" * 20)
    print("Testing ResNet3D-18")
    r3d18 = get_pretrained_r3d(depth=18)
    print("Successfully loaded ResNet3D-18 with pretrained weights.")
    print(r3d18)
    print("Replacing final layer.")
    replace_resnet3d_classifier(r3d18)
    print("Successfully replaced ResNet3D-18 fully connected layer.")
    print(r3d18)
    print("=" * 20)
    print("Testing ResNet3D-50")
    r3d50 = get_pretrained_r3d(depth=50)
    print("Successfully loaded ResNet3D-50 with pretrained weights.")
    print(r3d50)
    print("Replacing final layer.")
    replace_resnet3d_classifier(r3d50)
    print("Successfully replaced ResNet3D-50 fully connected layer.")

    print("=" * 20)
    print("Loading test data...")

    train_dataset = LunaDataset.get_dataset_with_transform(split='train',
                                                           model='video_pretrained', )
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=16, )
    next_batch, y = next(iter(train_dataloader))

    print("Successfully loaded test data.")
    print("=" * 20)
    print("Testing forward pass...")
    r3d18.eval()
    r3d50.eval()

    with torch.no_grad():
        r3d18(next_batch)
        print("Resnet3D-18 forward pass OK")
        r3d50(next_batch)
        print("Resnet3D-50 forward pass OK")

if __name__ == "__main__":
    main()