import torch
from torch.utils.data import WeightedRandomSampler
from luna_dataset.dataset import LunaDataset


def get_weighted_sampler(labels: torch.Tensor,
                         pos_weight: float = 2.0,
                         generator: torch.Generator = None,):

    weights = torch.where(labels == 1, pos_weight, 1.0)
    num_samples = labels.size(0)

    return WeightedRandomSampler(weights=weights,
                                 num_samples=num_samples,
                                 generator=generator,)

def get_weighted_sampler_luna(dataset: LunaDataset,
                              **kwargs):
    return get_weighted_sampler(dataset.get_all_labels(), **kwargs)