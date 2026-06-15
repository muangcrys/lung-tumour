from monai.transforms import Compose, ScaleIntensityRange, NormalizeIntensity, CenterSpatialCrop, RandSpatialCrop, \
    EnsureChannelFirst, RandFlip, RandRotate90, Resize, RepeatChannel, RandScaleIntensity, RandShiftIntensity, \
    RandAdjustContrast, RandGaussianNoise, RandGaussianSmooth, Transpose

from typing import Literal, List

def get_means_stds(statistics: Literal["kinetics", "0.5", "activitynet", "imagenet"], channel: int = 1):
    # from ResNet3D PyTorch
    if statistics == 'activitynet':
        mean = [0.4477, 0.4209, 0.3906]
        std = [0.2767, 0.2695, 0.2714]
    elif statistics == 'kinetics':
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
    elif statistics == '0.5':
        mean = [0.5] * channel
        std = [0.5] * channel
    elif statistics == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unknown statistics: {statistics}")

    return mean, std

def clip_and_scale(npzarray, maxHU=400.0, minHU=-1000.0):
    # from LUNA25 repo
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.0
    npzarray[npzarray < 0] = 0.0
    return npzarray

def get_monai_clip_scale(maxHU=400.0,
                         minHU=-1000.0,
                         mean=None,
                         std=None,
                         statistics: Literal["kinetics", "0.5", "imagenet"] = "0.5",
                         channel: int = 1,
                         return_type: Literal["list", "compose"] = "list"):
    if mean is None or std is None:
        print(f"No mean or std provided, using default values based on provided statistics: {statistics}.")
        mean, std = get_means_stds(statistics=statistics, channel=channel)
        print("Mean:", mean)
        print("Std:", std)

    compose_list = [
        ScaleIntensityRange(a_min=minHU, a_max=maxHU, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True)
    ]

    if return_type == "list":
        return compose_list
    elif return_type == "compose":
        return Compose(compose_list)
    else:
        # default fallback
        print(f"Unknown return type: {return_type}, returning as default (list)")
        return compose_list

def get_transforms(augmentation: bool = True,
                   spatial_augmentation: bool = True,
                   grey_values_augmentation: bool = True,
                   crop: bool = False,
                   crop_size: int = 112,
                   maxHU=400.0,
                   minHU=-1000.0,
                   mean=None,
                   std=None,
                   statistics: Literal["kinetics", "0.5", "imagenet"] = "kinetics",
                   scale: bool = False,
                   scale_to_size: int = 112,
                   replicate_channels: bool = False,
                   num_channels: int = 3,
                   output_depth_first: bool = False):
    ### EXPECTING: (1, D, H ,W)

    compose = []
    # cropping
    if replicate_channels:
        compose.append(RepeatChannel(repeats=num_channels))

    if crop:
        if augmentation:
            compose.append(RandSpatialCrop(roi_size=(-1, -1, crop_size, crop_size), random_size=False))
        else:
            compose.append(CenterSpatialCrop(roi_size=(-1, -1, crop_size, crop_size)))

    # clip and scale
    clip_scale_transform = get_monai_clip_scale(maxHU=maxHU,
                                                minHU=minHU,
                                                mean=mean,
                                                std=std,
                                                statistics=statistics,
                                                channel=num_channels,
                                                return_type="list")
    compose.extend(clip_scale_transform)

    if augmentation:
        # data is (C, D, H, W)
        if spatial_augmentation:
            spatial_augs_transforms = [
                RandFlip(prob=0.5, spatial_axis=0),    # flip along depth
                RandFlip(prob=0.5, spatial_axis=1),    # flip along height
                RandFlip(prob=0.5, spatial_axis=2),    # flip along width

                RandRotate90(prob=0.3, max_k=3,
                             spatial_axes=(1, 2))      # rotate H and W
            ]
            compose.extend(spatial_augs_transforms)
        if grey_values_augmentation:
            grey_augs_transforms = [
                RandScaleIntensity(factors=0.1, prob=0.5),
                RandShiftIntensity(offsets=0.1, prob=0.5),
                RandAdjustContrast(gamma=(0.8, 1.2), prob=0.3),
                RandGaussianNoise(mean=0.0, std=0.01, prob=0.3),
                RandGaussianSmooth(sigma_x=(0.25, 0.75),
                                   sigma_y=(0.25, 0.75),
                                   sigma_z=(0, 0),
                                   prob=0.3)
            ]
            compose.extend(grey_augs_transforms)

    if scale:
        compose.append(Resize(spatial_size=(-1, -1, scale_to_size, scale_to_size)))

    if output_depth_first:
        compose.append(Transpose(indices=(1, 0, 2, 3)))  # (C, D, H, W) -> (D, C, H, W)


    return Compose(compose)




