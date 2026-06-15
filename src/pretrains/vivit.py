from torch import nn
from models.vivit import _vivit_dir, get_vivit
from transformers import VivitConfig, VivitForVideoClassification
from luna_dataset.dataset import LunaDataset
from torch.utils.data import DataLoader
import torch



import torch
import torch.nn.functional as F


def interpolate_vivit_pos_embed(
    old_model,
    new_model,
    tubelet_size: int = 2,
    patch_size: int = 16,
):
    old_cfg = old_model.config
    new_cfg = new_model.config

    # Derive grid dimensions from model configs
    old_t = old_cfg.num_frames // tubelet_size
    old_h = old_cfg.image_size // patch_size
    old_w = old_cfg.image_size // patch_size

    new_t = new_cfg.num_frames // tubelet_size
    new_h = new_cfg.image_size // patch_size
    new_w = new_cfg.image_size // patch_size

    old_pos = old_model.vivit.embeddings.position_embeddings  # (1, 1 + T*H*W, C)
    hidden = old_pos.shape[-1]

    expected_tokens = 1 + old_t * old_h * old_w
    if old_pos.shape[1] != expected_tokens:
        raise ValueError(
            f"Old embedding has {old_pos.shape[1]} tokens, "
            f"but config implies {expected_tokens} (1 CLS + {old_t}×{old_h}×{old_w})."
        )

    cls_token = old_pos[:, :1]          # (1, 1, C)  — preserve CLS unchanged
    patch_embed = old_pos[:, 1:]        # (1, T*H*W, C)

    # Reshape to spatial volume and cast to float32 for interpolation
    patch_embed = (
        patch_embed
        .reshape(1, old_t, old_h, old_w, hidden)
        .permute(0, 4, 1, 2, 3)        # (1, C, T, H, W)
        .float()
    )

    patch_embed_new = F.interpolate(
        patch_embed,
        size=(new_t, new_h, new_w),
        mode="trilinear",
        align_corners=False,
    )                                   # (1, C, new_T, new_H, new_W)

    patch_embed_new = (
        patch_embed_new
        .permute(0, 2, 3, 4, 1)        # (1, new_T, new_H, new_W, C)
        .reshape(1, new_t * new_h * new_w, hidden)
        .to(old_pos.dtype)             # restore original dtype
    )

    new_pos_embed = torch.cat([cls_token, patch_embed_new], dim=1)

    with torch.no_grad():
        new_model.vivit.embeddings.position_embeddings.copy_(new_pos_embed)

    # new_model is already mutated above; returning it allows method chaining
    return new_model

def get_config_pretrained():
    config = VivitConfig.from_pretrained(_vivit_dir)
    config.num_labels = 1
    config.num_frames = 64
    config.image_size = 128

    return config

def get_model_pretrained(config=None,):
    if config is None:
        config = get_config_pretrained()

    pretrained_model = VivitForVideoClassification.from_pretrained(_vivit_dir)
    model = VivitForVideoClassification.from_pretrained(_vivit_dir, config=config, ignore_mismatched_sizes=True)

    print("Successfully loaded ViViT with pretrained weights.")
    print("Interpolating positional embeddings...")
    new_model = interpolate_vivit_pos_embed(
        old_model=pretrained_model,
        new_model=model,
        tubelet_size=2,
        patch_size=16
    )
    print("Success")

    # replace classifier
    # if replace_classifier:
    #     fc_in = model.classifier.in_features
    #     model.classifier = nn.Linear(in_features=fc_in, out_features=num_classes)
    
    return new_model

def main():
    print("Testing model loading (ViViT)")
    vivit = get_model_pretrained()
    print("Successfully loaded ViViT with pretrained weights.")
    fresh_vivit = get_vivit()
    print("Successfully loaded Fresh ViViT.")
    print("Testing forward pass...")
    print("=" * 20)
    print("Loading test data...")


    train_random_dataset = LunaDataset.get_dataset_with_transform(split='train',
                                                                  model="vivit_random")
    train_random_dataloader = DataLoader(train_random_dataset,
                                           batch_size=16)
    train_pretrain_dataset = LunaDataset.get_dataset_with_transform(split='train',
                                                           model='vivit_pretrained',)
    train_pretrain_dataloader = DataLoader(train_pretrain_dataset,
                                  batch_size=16, )
    next_batch_random, _ = next(iter(train_random_dataloader))
    next_batch_pretrained, _ = next(iter(train_pretrain_dataloader))
    print("Successfully loaded test data.")
    print("=" * 20)
    print("Testing forward pass...")
    vivit.eval()
    fresh_vivit.eval()

    with torch.no_grad():
        fresh_output = fresh_vivit(next_batch_random)
        print("Fresh ViVit forward pass OK.")
        pretrained_output = vivit(next_batch_pretrained, interpolate_pos_encoding=True)
        print("Pretrained ViViT forward pass OK.")

        print(fresh_output)
        print(pretrained_output)


if __name__ == "__main__":
    main()