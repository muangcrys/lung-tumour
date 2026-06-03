from torch import nn

from models.vivit import _vivit_dir
from transformers import VivitConfig, VivitForVideoClassification

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