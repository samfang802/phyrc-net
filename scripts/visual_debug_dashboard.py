"""Visual debugging dashboard for InterferenceMaskedDIEvaluator.

Run ``run_visual_dashboard(dataset, evaluator)`` after creating a dataset with
DI labels and an initialized InterferenceMaskedDIEvaluator.
"""

import random
from collections import defaultdict
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor


_COLUMN_ID_KEYS = ("column_id", "columnId", "id", "exp")
_DI_KEYS = ("DI", "di", "damage_index", "damageIndex")
_IMAGE_KEYS = ("target_image", "image", "images")


def _first_present(container: Any, keys: Tuple[str, ...], field_name: str) -> Any:
    for key in keys:
        if key in container:
            return container[key]
    raise KeyError(f"Could not find {field_name}; expected one of {keys}.")


def _sample_fields(sample: Any) -> Tuple[Any, Any, Any]:
    """Extract image, column id, and GT DI from project or mapping datasets."""
    if isinstance(sample, dict):
        image = _first_present(sample, _IMAGE_KEYS, "an image")
        column_id = _first_present(sample, _COLUMN_ID_KEYS, "a column id")
        gt_di = _first_present(sample, _DI_KEYS, "a DI value")
        return image, column_id, gt_di

    # Existing project tuple formats:
    # PatternLoopDataset: image, hidden_state, direction, exp, timestep, DI, info
    # DamageIndexDataset: image, features, exp, DI
    if isinstance(sample, (tuple, list)) and len(sample) >= 7:
        return sample[0], sample[3], sample[5]
    if isinstance(sample, (tuple, list)) and len(sample) == 4:
        return sample[0], sample[2], sample[3]
    raise TypeError(
        "Dataset samples must be mappings containing image/column_id/DI fields, "
        "or use this project's PatternLoopDataset or DamageIndexDataset tuple format."
    )


def _label_fields(dataset: Any, index: int) -> Tuple[Any, Any]:
    """Read column id and DI from ``dataset.label`` without loading its image."""
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return _label_fields(dataset.dataset, int(dataset.indices[index]))

    if not hasattr(dataset, "label"):
        raise AttributeError("Dataset has no label metadata.")
    row = dataset.label.iloc[index]
    column_id = _first_present(row, _COLUMN_ID_KEYS, "a column id")
    gt_di = _first_present(row, _DI_KEYS, "a DI value")
    return column_id, gt_di


def _collect_candidates(dataset: Any) -> Dict[str, list]:
    """Group local dataset indices by column id, loading samples only if needed."""
    by_column = defaultdict(list)
    cached_samples = {}

    for index in range(len(dataset)):
        try:
            column_id, gt_di = _label_fields(dataset, index)
        except (AttributeError, KeyError):
            sample = dataset[index]
            image, column_id, gt_di = _sample_fields(sample)
            cached_samples[index] = (image, column_id, gt_di)
        by_column[str(column_id)].append((index, gt_di))
    return by_column, cached_samples


def _image_to_unit_tensor(image: Any) -> torch.Tensor:
    """Convert a CHW tensor or PIL image to a CPU float tensor in [0, 1]."""
    if not isinstance(image, torch.Tensor):
        image = to_tensor(image)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected an RGB CHW image, got shape {tuple(image.shape)}.")

    image = image.detach().float().cpu()
    # Project image datasets often normalize RGB data into [-1, 1].
    if image.min() < 0:
        image = (image + 1.0) * 0.5
    return image.clamp(0, 1)


def _evaluator_device(evaluator: Any) -> torch.device:
    try:
        return next(evaluator.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _mask_to_image(mask: torch.Tensor, grid_size: Tuple[int, int], image_size: Tuple[int, int]) -> torch.Tensor:
    """Expand a flat row-major patch mask to a blocky image-size binary mask."""
    grid_height, grid_width = grid_size
    if mask.numel() != grid_height * grid_width:
        raise ValueError(
            f"Patch mask has {mask.numel()} values, incompatible with grid {grid_size}."
        )
    return F.interpolate(
        mask.reshape(1, 1, grid_height, grid_width).float(),
        size=image_size,
        mode="nearest",
    )[0, 0].bool()


def run_visual_dashboard(dataset, evaluator):
    """Visualize exactly one random image per column id through the DI pipeline.

    The evaluator must return the detail fields produced by
    ``InterferenceMaskedDIEvaluator(..., return_details=True)``. For every
    unique ``column_id`` (or this project's ``exp``), the function creates one
    1x4 figure containing original, detection, patch alignment, and masked view.

    Returns:
        Dictionary mapping each column id to its matplotlib ``Figure``.
    """
    by_column, cached_samples = _collect_candidates(dataset)
    device = _evaluator_device(evaluator)
    figures = {}
    evaluator.eval()

    for column_id, candidates in by_column.items():
        index, gt_di = random.choice(candidates)
        if index in cached_samples:
            image, _, gt_di = cached_samples[index]
        else:
            image, _, sampled_gt_di = _sample_fields(dataset[index])
            gt_di = sampled_gt_di

        image_01 = _image_to_unit_tensor(image)
        with torch.no_grad():
            result = evaluator(
                image_01.unsqueeze(0).to(device),
                input_range="zero_one",
                return_details=True,
            )

        required = {
            "predicted_di",
            "boxes_xyxy",
            "interference_patch_mask",
            "patch_grid_size",
        }
        missing = required.difference(result)
        if missing:
            raise KeyError(
                f"Evaluator detail output is missing {sorted(missing)}. "
                "Use InterferenceMaskedDIEvaluator with return_details=True."
            )

        predicted_di = float(result["predicted_di"][0, 0].detach().cpu())
        boxes = result["boxes_xyxy"][0].detach().cpu()
        patch_mask = result["interference_patch_mask"][0].detach().cpu()
        image_mask = _mask_to_image(
            patch_mask,
            result["patch_grid_size"],
            image_01.shape[-2:],
        )

        image_hwc = image_01.permute(1, 2, 0).numpy()
        masked_view = image_01.clone()
        masked_view[:, image_mask] = 0

        figure, axes = plt.subplots(1, 4, figsize=(18, 4.5))

        axes[0].imshow(image_hwc)
        axes[0].set_title(f"GT DI: {float(torch.as_tensor(gt_di).squeeze()):.3f}")

        axes[1].imshow(image_hwc)
        for x1, y1, x2, y2 in boxes.tolist():
            axes[1].add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
            )
        axes[1].set_title("Detected Interferences")

        axes[2].imshow(image_hwc)
        axes[2].imshow(
            image_mask.numpy(),
            cmap="Reds",
            alpha=0.45,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        axes[2].set_title("Patch Mask Alignment")

        axes[3].imshow(masked_view.permute(1, 2, 0).numpy())
        prediction_title = f"Predicted DI: {predicted_di:.3f}"
        if not getattr(evaluator, "di_predictor_is_trained", True):
            prediction_title += "\n(untrained MLP: visual test only)"
        axes[3].set_title(prediction_title)

        for axis in axes:
            axis.axis("off")
        figure.suptitle(
            f"Column: {column_id} | masked patches: "
            f"{int(patch_mask.sum())}/{patch_mask.numel()}"
        )
        figure.tight_layout()
        plt.show()
        figures[column_id] = figure

    return figures
