import os, glob, math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF

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


def _parse_clahe_tile_grid_size(tile_grid_size):
    if isinstance(tile_grid_size, str):
        parts = [p.strip() for p in tile_grid_size.replace("x", ",").split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid CLAHE tileGridSize: {tile_grid_size!r}")
        return (int(parts[0]), int(parts[1]))
    if isinstance(tile_grid_size, (list, tuple)) and len(tile_grid_size) == 2:
        return (int(tile_grid_size[0]), int(tile_grid_size[1]))
    raise ValueError(f"Invalid CLAHE tileGridSize: {tile_grid_size!r}")


def _clahe_numpy(gray_u8, clip_limit=2.0, tile_grid_size=(8, 8)):
    """OpenCV-style CLAHE on a uint8 grayscale image (pure NumPy)."""
    image = np.asarray(gray_u8, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError(f"CLAHE expects 2D grayscale, got shape {image.shape}")

    height, width = image.shape
    tiles_y, tiles_x = int(tile_grid_size[0]), int(tile_grid_size[1])
    tiles_y = max(1, tiles_y)
    tiles_x = max(1, tiles_x)

    tile_h = int(np.ceil(height / tiles_y))
    tile_w = int(np.ceil(width / tiles_x))
    pad_h = tile_h * tiles_y - height
    pad_w = tile_w * tiles_x - width
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")
    padded_h, padded_w = image.shape

    # OpenCV clipLimit is relative to tile area / 256.
    clip_threshold = max(1, int(np.ceil(float(clip_limit) * tile_h * tile_w / 256.0)))
    luts = np.zeros((tiles_y, tiles_x, 256), dtype=np.float32)

    for ty in range(tiles_y):
        y0, y1 = ty * tile_h, (ty + 1) * tile_h
        for tx in range(tiles_x):
            x0, x1 = tx * tile_w, (tx + 1) * tile_w
            hist, _ = np.histogram(image[y0:y1, x0:x1], bins=256, range=(0, 256))
            hist = hist.astype(np.float32)

            excess = np.maximum(hist - clip_threshold, 0.0).sum()
            hist = np.minimum(hist, clip_threshold)
            hist += excess / 256.0

            cdf = np.cumsum(hist)
            cdf_min = cdf[np.nonzero(cdf)[0][0]] if np.any(cdf) else 0.0
            denom = max((tile_h * tile_w) - cdf_min, 1.0)
            lut = np.clip(((cdf - cdf_min) * 255.0) / denom, 0, 255)
            luts[ty, tx] = lut

    ys = np.arange(height, dtype=np.float32) / float(tile_h) - 0.5
    xs = np.arange(width, dtype=np.float32) / float(tile_w) - 0.5
    ys = np.clip(ys, 0.0, tiles_y - 1.0001)
    xs = np.clip(xs, 0.0, tiles_x - 1.0001)

    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, tiles_y - 1)
    x1 = np.minimum(x0 + 1, tiles_x - 1)
    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)

    src = image[:height, :width]
    vals = src.astype(np.int32)
    out = np.empty((height, width), dtype=np.float32)
    for i in range(height):
        yi0, yi1 = int(y0[i]), int(y1[i])
        wyi = float(wy[i])
        row_vals = vals[i]
        a = (1.0 - wx) * luts[yi0, x0, row_vals] + wx * luts[yi0, x1, row_vals]
        b = (1.0 - wx) * luts[yi1, x0, row_vals] + wx * luts[yi1, x1, row_vals]
        out[i] = (1.0 - wyi) * a + wyi * b
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


class ApplyCLAHE:
    """Apply CLAHE on grayscale images to enhance micro-cracks.

    Expects a CHW image tensor. RGB inputs are converted to luminance first.
    Uses OpenCV-compatible CLAHE parameters: clipLimit / tileGridSize.
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = _parse_clahe_tile_grid_size(tile_grid_size)

    def __call__(self, image):
        if not torch.is_tensor(image):
            raise TypeError(f"ApplyCLAHE expects a torch.Tensor, got {type(image)}")
        if image.ndim != 3:
            raise ValueError(f"ApplyCLAHE expects CHW tensor, got shape {tuple(image.shape)}")

        was_float = torch.is_floating_point(image)
        img = image.detach()
        if was_float:
            # Support both [0,1] and [0,255] float tensors.
            scale = 255.0 if float(img.max()) <= 1.0 + 1e-6 else 1.0
            img_u8 = (img * scale).clamp(0, 255).to(torch.uint8)
        else:
            img_u8 = img.to(torch.uint8)

        if img_u8.shape[0] == 3:
            r, g, b = img_u8[0].float(), img_u8[1].float(), img_u8[2].float()
            gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).round().to(torch.uint8).cpu().numpy()
        elif img_u8.shape[0] == 1:
            gray = img_u8[0].cpu().numpy()
        else:
            raise ValueError(f"ApplyCLAHE expects 1 or 3 channels, got {img_u8.shape[0]}")

        enhanced = _clahe_numpy(
            gray,
            clip_limit=self.clip_limit,
            tile_grid_size=self.tile_grid_size,
        )
        out = torch.from_numpy(enhanced).to(device=image.device).unsqueeze(0)
        if was_float:
            out = out.to(dtype=image.dtype) / 255.0
        else:
            out = out.to(dtype=image.dtype)
        return out


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
    aug_affine=True,
    aug_blur_kernel=3,
    aug_blur_sigma_min=0.1,
    aug_blur_sigma_max=0.5,
    aug_erase_prob=0.0,
    aug_erase_scale_min=0.1,
    aug_erase_scale_max=0.2,
    aug_erase_ratio_min=0.3,
    aug_erase_ratio_max=3.3,
    aug_erase_random_value=False,
    include_random_erasing=True,
    use_grayscale_input=True,
    use_clahe=False,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
):
    use_grayscale_input = bool(use_grayscale_input)
    use_clahe = bool(use_clahe)
    # CLAHE is defined on grayscale luminance; force grayscale when enabled.
    if use_clahe:
        use_grayscale_input = True

    transforms = [
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
    ]
    if use_grayscale_input:
        transforms.append(v2.Grayscale(num_output_channels=1))
    if use_clahe:
        transforms.append(
            ApplyCLAHE(
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )
        )

    if use_augmentation:
        # In grayscale mode saturation/hue are disabled automatically.
        if aug_brightness > 0 or aug_contrast > 0:
            jitter_saturation = 0 if use_grayscale_input else aug_saturation
            jitter_hue = 0 if use_grayscale_input else aug_hue
            transforms.append(
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=aug_brightness,
                            contrast=aug_contrast,
                            saturation=jitter_saturation,
                            hue=jitter_hue,
                        )
                    ],
                    p=aug_prob,
                )
            )
        if aug_affine:
            transforms.append(
                v2.RandomApply(
                    [
                        v2.RandomAffine(
                            degrees=0,
                            translate=(aug_translate, aug_translate),
                            scale=(aug_scale_min, aug_scale_max),
                        )
                    ],
                    p=aug_prob,
                )
            )
        transforms.append(
            v2.RandomApply(
                [
                    v2.GaussianBlur(
                        kernel_size=_normalize_blur_kernel(aug_blur_kernel),
                        sigma=(aug_blur_sigma_min, aug_blur_sigma_max),
                    )
                ],
                p=aug_blur_prob,
            )
        )

    transforms.append(v2.ToDtype(torch.float32, scale=output_scale))
    if use_grayscale_input:
        transforms.append(v2.Lambda(lambda x: x.repeat(3, 1, 1)))
    if use_augmentation and include_random_erasing and aug_erase_prob > 0:
        erase_value = "random" if aug_erase_random_value else 0.0
        transforms.append(
            v2.RandomErasing(
                p=aug_erase_prob,
                scale=(aug_erase_scale_min, aug_erase_scale_max),
                ratio=(aug_erase_ratio_min, aug_erase_ratio_max),
                value=erase_value,
            )
        )

    return v2.Compose(transforms)


class MaskedDamageImageTransform:
    """Apply image augmentation and return a masked image with its valid-pixel mask.

    ``loss_mask`` has shape ``(1, H, W)`` and uses 1 for valid pixels.
    RandomAffine translation is applied jointly to image + all mask tensors.
    """

    def __init__(
        self,
        image_transform,
        use_augmentation=False,
        horizontal_flip_prob=0.5,
        aug_affine=False,
        aug_prob=0.5,
        aug_translate=0.10,
        aug_scale_min=1.0,
        aug_scale_max=1.0,
        aug_erase_prob=0.25,
        aug_erase_scale_min=0.01,
        aug_erase_scale_max=0.08,
        aug_erase_ratio_min=0.3,
        aug_erase_ratio_max=3.3,
        normalize_mean=None,
        normalize_std=None,
        grayscale_expanded_rgb=True,
    ):
        self.image_transform = image_transform
        self.use_augmentation = use_augmentation
        self.horizontal_flip_prob = horizontal_flip_prob
        self.aug_affine = bool(aug_affine)
        self.aug_prob = float(aug_prob)
        self.aug_translate = float(aug_translate)
        self.aug_scale_min = float(aug_scale_min)
        self.aug_scale_max = float(aug_scale_max)
        self.aug_erase_prob = float(aug_erase_prob)
        self.aug_erase_scale = (
            float(aug_erase_scale_min),
            float(aug_erase_scale_max),
        )
        self.aug_erase_ratio = (
            float(aug_erase_ratio_min),
            float(aug_erase_ratio_max),
        )
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.grayscale_expanded_rgb = bool(grayscale_expanded_rgb)

    def _maybe_normalize(self, target_image):
        if self.normalize_mean is None or self.normalize_std is None:
            return target_image
        return TF.normalize(target_image, mean=list(self.normalize_mean), std=list(self.normalize_std))

    def _maybe_apply_random_affine(self, target_image, *mask_tensors):
        if not self.aug_affine or torch.rand(()) >= self.aug_prob:
            return (target_image, *mask_tensors)

        height, width = target_image.shape[-2], target_image.shape[-1]
        max_dx = int(round(self.aug_translate * width))
        max_dy = int(round(self.aug_translate * height))
        # Asymmetrical translation:
        # - X: symmetric [-max_dx, max_dx]
        # - Y: only upward / zero [-max_dy, 0] (dy < 0 moves content up in TF.affine).
        dx = int(torch.randint(-max_dx, max_dx + 1, (1,)).item()) if max_dx > 0 else 0
        dy = int(torch.randint(-max_dy, 1, (1,)).item()) if max_dy > 0 else 0
        translate = [dx, dy]
        scale = float(torch.empty(1).uniform_(self.aug_scale_min, self.aug_scale_max).item())

        affined_image = TF.affine(
            target_image,
            angle=0.0,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
        )
        affined_masks = []
        for mask in mask_tensors:
            affined_masks.append(
                TF.affine(
                    mask,
                    angle=0.0,
                    translate=translate,
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.NEAREST,
                )
            )
        return (affined_image, *affined_masks)

    @staticmethod
    def _sample_erase_region(height, width, scale, ratio):
        area = height * width
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(*scale).item()
            aspect_ratio = math.exp(torch.empty(1).uniform_(*log_ratio).item())
            erase_height = int(round(math.sqrt(erase_area * aspect_ratio)))
            erase_width = int(round(math.sqrt(erase_area / aspect_ratio)))
            if 0 < erase_height <= height and 0 < erase_width <= width:
                top = torch.randint(0, height - erase_height + 1, (1,)).item()
                left = torch.randint(0, width - erase_width + 1, (1,)).item()
                return top, left, erase_height, erase_width
        return None

    def _gaussian_noise_like(self, region, reference):
        if self.grayscale_expanded_rgb:
            # Keep random erasing luminance-consistent with grayscale-expanded RGB.
            channel_mean = reference.mean(dim=(-2, -1), keepdim=True).mean(dim=0, keepdim=True)
            channel_std = reference.std(dim=(-2, -1), keepdim=True).mean(dim=0, keepdim=True).clamp_min(1e-6)
            mono_noise = torch.randn(
                (1, region.shape[-2], region.shape[-1]),
                device=region.device,
                dtype=region.dtype,
            ) * channel_std + channel_mean
            return mono_noise.repeat(region.shape[0], 1, 1)

        channel_mean = reference.mean(dim=(-2, -1), keepdim=True)
        channel_std = reference.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return torch.randn_like(region) * channel_std + channel_mean

    def _maybe_apply_random_erasing(self, target_image, loss_mask):
        if self.aug_erase_prob <= 0.0 or torch.rand(()) >= self.aug_erase_prob:
            return target_image, loss_mask

        height, width = target_image.shape[-2], target_image.shape[-1]
        region = self._sample_erase_region(
            height=height,
            width=width,
            scale=self.aug_erase_scale,
            ratio=self.aug_erase_ratio,
        )
        if region is None:
            return target_image, loss_mask

        top, left, erase_height, erase_width = region
        erased = target_image[
            :,
            top : top + erase_height,
            left : left + erase_width,
        ]
        noise = self._gaussian_noise_like(erased, target_image)
        target_image[
            :,
            top : top + erase_height,
            left : left + erase_width,
        ] = noise
        # Exclude erased pixels from FM loss.
        loss_mask[
            :,
            top : top + erase_height,
            left : left + erase_width,
        ] = 0.0
        return target_image, loss_mask

    def __call__(self, image, object_mask=None):
        target_image = self.image_transform(image)
        if target_image.ndim != 3:
            raise ValueError(
                "MaskedDamageImageTransform expects a CHW image tensor; "
                f"received shape {tuple(target_image.shape)}."
            )

        loss_mask = torch.ones(
            (1, target_image.shape[-2], target_image.shape[-1]),
            dtype=target_image.dtype,
            device=target_image.device,
        )
        if object_mask is not None:
            object_mask = object_mask.to(device=target_image.device, dtype=target_image.dtype)
            if object_mask.ndim == 2:
                object_mask = object_mask.unsqueeze(0)
            if object_mask.shape[-2:] != target_image.shape[-2:]:
                object_mask = TF.resize(
                    object_mask,
                    size=[target_image.shape[-2], target_image.shape[-1]],
                    interpolation=InterpolationMode.NEAREST,
                )
        flipped = False
        if not self.use_augmentation:
            if object_mask is not None:
                loss_mask = loss_mask * object_mask
            target_image = self._maybe_normalize(target_image)
            return {"target_image": target_image, "loss_mask": loss_mask, "flipped": flipped}

        mask_stack = [loss_mask] + ([object_mask] if object_mask is not None else [])
        target_image, *mask_stack = self._maybe_apply_random_affine(target_image, *mask_stack)
        loss_mask = mask_stack[0]
        if object_mask is not None:
            object_mask = mask_stack[1]

        if torch.rand(()) < self.horizontal_flip_prob:
            target_image = torch.flip(target_image, dims=(-1,))
            loss_mask = torch.flip(loss_mask, dims=(-1,))
            if object_mask is not None:
                object_mask = torch.flip(object_mask, dims=(-1,))
            flipped = True
        target_image, loss_mask = self._maybe_apply_random_erasing(target_image, loss_mask)
        if object_mask is not None:
            loss_mask = loss_mask * object_mask
        target_image = self._maybe_normalize(target_image)
        return {"target_image": target_image, "loss_mask": loss_mask, "flipped": flipped}


def build_damage_image_transform_with_loss_mask(
    image_size,
    use_augmentation=False,
    output_scale=True,
    normalize_mean=None,
    normalize_std=None,
    **augmentation_kwargs,
):
    """Build an on-the-fly image transform that also emits a binary loss mask."""
    hflip_prob = float(augmentation_kwargs.pop("aug_hflip_prob", 0.5))
    use_random_affine = augmentation_kwargs.pop("use_random_affine", None)
    if use_random_affine is None:
        use_random_affine = augmentation_kwargs.pop("aug_affine", True)
    aug_affine = bool(use_random_affine)
    use_grayscale_input = bool(augmentation_kwargs.pop("use_grayscale_input", True))
    use_clahe = bool(augmentation_kwargs.pop("use_clahe", False))
    clahe_clip_limit = float(augmentation_kwargs.pop("clahe_clip_limit", 2.0))
    clahe_tile_grid_size = augmentation_kwargs.pop("clahe_tile_grid_size", (8, 8))
    aug_prob = float(augmentation_kwargs.pop("aug_prob", 0.5))
    aug_translate = float(augmentation_kwargs.pop("aug_translate", 0.10))
    aug_scale_min = float(augmentation_kwargs.pop("aug_scale_min", 1.0))
    aug_scale_max = float(augmentation_kwargs.pop("aug_scale_max", 1.0))
    # RandomAffine is handled jointly with loss/object masks in MaskedDamageImageTransform.
    image_transform = build_damage_image_transform(
        image_size=image_size,
        use_augmentation=use_augmentation,
        output_scale=output_scale,
        normalize_mean=None,
        normalize_std=None,
        include_random_erasing=False,
        aug_affine=False,
        use_grayscale_input=use_grayscale_input,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
        **augmentation_kwargs,
    )
    return MaskedDamageImageTransform(
        image_transform=image_transform,
        use_augmentation=use_augmentation,
        horizontal_flip_prob=hflip_prob,
        aug_affine=aug_affine,
        aug_prob=aug_prob,
        aug_translate=aug_translate,
        aug_scale_min=aug_scale_min,
        aug_scale_max=aug_scale_max,
        aug_erase_prob=augmentation_kwargs.get("aug_erase_prob", 0.25),
        aug_erase_scale_min=augmentation_kwargs.get("aug_erase_scale_min", 0.01),
        aug_erase_scale_max=augmentation_kwargs.get("aug_erase_scale_max", 0.08),
        aug_erase_ratio_min=augmentation_kwargs.get("aug_erase_ratio_min", 0.3),
        aug_erase_ratio_max=augmentation_kwargs.get("aug_erase_ratio_max", 3.3),
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        grayscale_expanded_rgb=use_grayscale_input or use_clahe,
    )


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
            
