from transformers import VivitForVideoClassification, VivitConfig
from pathlib import Path

_model_on_hub = "google/vivit-b-16x2-kinetics400"
_vivit_dir = Path(__file__).parent.parent.parent.resolve() / "checkpoints" / "vivit" / "vivit-b-16x2-kinetics400"

def get_config(num_channels = 1):
    config = VivitConfig.from_pretrained(_vivit_dir)

    # replacing
    config.num_labels = 1
    config.num_frames = 64
    config.image_size = 128
    config.num_channels = num_channels

    return config

def get_vivit(config = None):
    if config is None:
        config = get_config()

    model = VivitForVideoClassification(config)
    return model


