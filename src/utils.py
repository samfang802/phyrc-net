import os, glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import v2

REGRESSOR_CKPT_PATTERNS = (
    "**/best.ckpt",
    "**/best_*.ckpt",
    "**/best-*.ckpt",
    "**/*r2*.ckpt",
    "**/epoch=*.ckpt",
)


def _normalize_blur_kernel(kernel_size):
    if kernel_size < 3:
        return 3
    if kernel_size % 2 == 0:
        return kernel_size + 1
    return kernel_size


def build_damage_image_transform(
    image_size,
    use_augmentation=False,
    output_scale=True,
    normalize_mean=None,
    normalize_std=None,
    aug_prob=0.5,
    aug_blur_prob=0.15,
    aug_brightness=0.12,
    aug_contrast=0.12,
    aug_saturation=0.08,
    aug_hue=0.02,
    aug_translate=0.02,
    aug_scale_min=0.98,
    aug_scale_max=1.02,
    aug_blur_kernel=3,
    aug_blur_sigma_min=0.1,
    aug_blur_sigma_max=0.5,
    aug_erase_prob=0.1,
    aug_erase_scale_min=0.01,
    aug_erase_scale_max=0.08,
    aug_erase_ratio_min=0.3,
    aug_erase_ratio_max=3.3,
    aug_erase_random_value=False,
):
    transforms = [
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
    ]

    if use_augmentation:
        transforms.extend(
            [
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=aug_brightness,
                            contrast=aug_contrast,
                            saturation=aug_saturation,
                            hue=aug_hue,
                        )
                    ],
                    p=aug_prob,
                ),
                v2.RandomApply(
                    [
                        v2.RandomAffine(
                            degrees=0,
                            translate=(aug_translate, aug_translate),
                            scale=(aug_scale_min, aug_scale_max),
                        )
                    ],
                    p=aug_prob,
                ),
                v2.RandomApply(
                    [
                        v2.GaussianBlur(
                            kernel_size=_normalize_blur_kernel(aug_blur_kernel),
                            sigma=(aug_blur_sigma_min, aug_blur_sigma_max),
                        )
                    ],
                    p=aug_blur_prob,
                ),
            ]
        )

    transforms.append(v2.ToDtype(torch.float32, scale=output_scale))
    if use_augmentation and aug_erase_prob > 0:
        erase_value = "random" if aug_erase_random_value else 0.0
        transforms.append(
            v2.RandomErasing(
                p=aug_erase_prob,
                scale=(aug_erase_scale_min, aug_erase_scale_max),
                ratio=(aug_erase_ratio_min, aug_erase_ratio_max),
                value=erase_value,
            )
        )
    if normalize_mean is not None and normalize_std is not None:
        transforms.append(v2.Normalize(mean=normalize_mean, std=normalize_std))

    return v2.Compose(transforms)


def resolve_checkpoint(result_dir, patterns, description, allow_latest_fallback=True):
    for pattern in patterns:
        matches = glob.glob(os.path.join(result_dir, pattern), recursive=True)
        if not matches:
            continue
        if "epoch=" in pattern:
            return max(matches, key=os.path.getmtime)
        return sorted(matches)[0]

    all_ckpts = [
        path for path in glob.glob(os.path.join(result_dir, "**/*.ckpt"), recursive=True)
        if not path.endswith("last.ckpt")
    ]
    if all_ckpts:
        if allow_latest_fallback:
            latest = max(all_ckpts, key=os.path.getmtime)
            print(f"Warning: no preferred {description} found; using latest checkpoint: {latest}")
            return latest
        raise FileNotFoundError(
            f"Could not find {description} with patterns {patterns}. "
            f"Checkpoints present: {sorted(all_ckpts)}"
        )
    raise FileNotFoundError(
        f"Could not find {description} under {result_dir}. "
        f"No .ckpt files found — ensure training finished successfully."
    )


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
            
