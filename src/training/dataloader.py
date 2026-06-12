import torch
from torch.utils.data import DataLoader
from luna_dataset.dataset import LunaDataset
from pathlib import Path
from typing import Literal
from training.sampler import get_weighted_sampler_luna
from utility.reproducibility import seed_worker


def get_train_dataloader(model_type: Literal["video_pretrained", "medical_pretrained", "random_init"],
                         n_input_channels: int = None,
                         annotation: str | Path | None = None,
                         image_dir: str | Path | None = None,
                         batch_size: int = 16,
                         num_workers: int = 0,
                         pos_weight: float = 2.0,
                         seed: int = 4242,
                         deterministic: bool = True):
    generator = torch.Generator()

    if deterministic:
        generator.manual_seed(seed)
        train_dataset = LunaDataset.get_dataset_with_transform(
            annotation_file=annotation,
            image_dir=image_dir,
            split="train",
            n_input_channels=n_input_channels,
            seed=seed,
            model=model_type,
        )
        train_sampler = get_weighted_sampler_luna(train_dataset,
                                                  pos_weight=pos_weight,
                                                  generator=generator, )
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  generator=generator,
                                  worker_init_fn=seed_worker)
    else:
        train_dataset = LunaDataset.get_dataset_with_transform(
            annotation_file=annotation,
            image_dir=image_dir,
            split="train",
            model=model_type,
            n_input_channels=n_input_channels,
        )
        train_sampler = get_weighted_sampler_luna(train_dataset,
                                                  pos_weight=pos_weight, )
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  worker_init_fn=seed_worker,
                                  generator=generator)

    return train_loader


def get_validate_loader(model_type: Literal["video_pretrained", "medical_pretrained", "random_init"],
                        n_input_channels: int = None,
                        annotation: str | Path | None = None,
                        image_dir: str | Path | None = None,
                        batch_size: int = 16,
                        num_workers: int = 0, ):
    validate_dataset = LunaDataset.get_dataset_with_transform(
        annotation_file=annotation,
        image_dir=image_dir,
        split="validate",
        model=model_type,
        n_input_channels=n_input_channels
    )
    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers, )
    return validate_loader


def get_train_val_loaders(model_type: Literal["video_pretrained", "medical_pretrained", "random_init"],
                          n_input_channels: int = None,
                          train_annotation: str | Path | None = None,
                          train_image_dir: str | Path | None = None,
                          validate_annotation: str | Path | None = None,
                          validate_image_dir: str | Path | None = None,
                          batch_size: int = 16,
                          num_workers: int = 0,
                          pos_weight: float = 2.0,
                          seed: int = 4242,
                          deterministic: bool = True
                          ):
    train_loader = get_train_dataloader(model_type=model_type,
                                        n_input_channels=n_input_channels,
                                        annotation=train_annotation,
                                        image_dir=train_image_dir,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        pos_weight=pos_weight,
                                        seed=seed,
                                        deterministic=deterministic)

    validate_loader = get_validate_loader(model_type=model_type,
                                          n_input_channels=n_input_channels,
                                          annotation=validate_annotation,
                                          image_dir=validate_image_dir,
                                          batch_size=batch_size,
                                          num_workers=num_workers, )

    return train_loader, validate_loader
