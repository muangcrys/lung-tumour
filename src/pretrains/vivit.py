from torch import nn

from models.vivit import _vivit_dir
from transformers import VivitConfig, VivitForVideoClassification
from luna_dataset.dataset import LunaDataset
from torch.utils.data import DataLoader
import torch



def get_config_pretrained():
    config = VivitConfig.from_pretrained(_vivit_dir)

    return config

def get_model_pretrained(config=None,
                         replace_classifier: bool = True,
                         num_classes: int = 1):
    if config is None:
        config = get_config_pretrained()

    model = VivitForVideoClassification.from_pretrained(_vivit_dir, config=config)

    # replace classifier
    if replace_classifier:
        fc_in = model.classifier.in_features
        model.classifier = nn.Linear(in_features=fc_in, out_features=num_classes)
    
    return model

def main():
    print("Testing model loading (ViViT)")
    vivit = get_model_pretrained()
    print("Successfully loaded ViViT with pretrained weights.")
    print("Testing forward pass...")
    print("=" * 20)
    print("Loading test data...")

    train_dataset = LunaDataset.get_dataset_with_transform(split='train',
                                                           model='medical_pretrained', )
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=16, )
    next_batch, y = next(iter(train_dataloader))

    print("Successfully loaded test data.")
    print("=" * 20)
    print("Testing forward pass...")