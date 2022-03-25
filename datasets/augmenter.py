from .video_transforms import (GroupRandomHorizontalFlip,
                               GroupMultiScaleCrop, GroupScale, GroupCenterCrop, GroupRandomCrop,
                               GroupNormalize, Stack, ToTorchFormatTensor, GroupRandomScale)




def get_augmentor(is_train, image_size, mean=None,
                  std=None, disable_scaleup=False, is_flow=False,
                  threed_data=False, version='v1', scale_range=None):

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    scale_range = [256, 320] if scale_range is None else scale_range
    augments = []

    if is_train:
        if version == 'v1':
            augments += [
                GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
            ]
        elif version == 'v2':
            augments += [
                GroupRandomScale(scale_range),
                GroupRandomCrop(image_size),
            ]
        augments += [GroupRandomHorizontalFlip(is_flow=is_flow)]
    else:
        scaled_size = image_size if disable_scaleup else int(image_size / 0.875 + 0.5)
        augments += [
            GroupScale(scaled_size),
            GroupCenterCrop(image_size)
        ]
    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]

    augmentor = transforms.Compose(augments)
    return augmentor
