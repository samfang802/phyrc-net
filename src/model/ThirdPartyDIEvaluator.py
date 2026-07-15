"""Third-party DI evaluation with Grounding DINO and an injected DINOv3 model.

The detector identifies non-structural objects, then their spatial regions are
excluded from DINO patch-token pooling before the learned DI regressor runs.
"""

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class DamageIndexMLP(nn.Module):
    """MLP trained on clean DINO pooled patch features and DI labels."""

    def __init__(self, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class InterferenceMaskedDIEvaluator(nn.Module):
    """Predict DI while ignoring DINO patches covered by detected interference.

    Args:
        detector_model_id: Hugging Face Grounding DINO model identifier.
        dino: Already-loaded DINOv3 Vision Transformer.
        dino_processor: The processor paired with the supplied DINOv3 model.
        box_threshold: Minimum Grounding DINO query confidence for an interference box.
        text_threshold: Minimum prompt-token confidence for an interference box.
        max_interference_area_ratio: Reject boxes larger than this fraction of
            the source image. Interference objects should not cover a column.
        freeze_dino: Freeze DINO features while training only ``di_predictor``.
        di_predictor: A trained ``nn.Module`` mapping pooled DINO features to
            ``[batch, 1]`` DI. If omitted, a new MLP is created and must be
            trained before its predictions are meaningful.
    """

    def __init__(
        self,
        dino: nn.Module,
        dino_processor: Any,
        detector_model_id: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        max_interference_area_ratio: float = 0.5,
        freeze_dino: bool = True,
        di_predictor: Optional[nn.Module] = None,
    ):
        super().__init__()
        try:
            from transformers import (
                GroundingDinoForObjectDetection,
                GroundingDinoProcessor,
            )
        except ImportError as error:
            raise ImportError(
                "Install transformers to use InterferenceMaskedDIEvaluator: "
                "pip install transformers"
            ) from error

        self.detector_processor = GroundingDinoProcessor.from_pretrained(detector_model_id)
        self.detector = GroundingDinoForObjectDetection.from_pretrained(detector_model_id)
        self.dino_processor = dino_processor
        self.dino = dino

        self.detector.eval()
        self.detector.requires_grad_(False)
        self.freeze_dino = freeze_dino
        if freeze_dino:
            self.dino.eval()
            self.dino.requires_grad_(False)

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_interference_area_ratio = max_interference_area_ratio
        feature_dim = getattr(self.dino.config, "hidden_size", None)
        if feature_dim is None:
            feature_dim = getattr(self.dino.config, "embed_dim", None)
        if feature_dim is None:
            raise ValueError(
                "Could not determine the DINOv3 feature dimension. "
                "Expected dino.config.hidden_size or dino.config.embed_dim."
            )
        self.di_predictor = di_predictor or DamageIndexMLP(feature_dim)

    @staticmethod
    def _as_01(images: torch.Tensor, input_range: str) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected generated_images shaped [B, 3, H, W], got {tuple(images.shape)}.")
        if input_range == "zero_one":
            return images.clamp(0, 1)
        if input_range == "minus_one_one":
            return ((images + 1) * 0.5).clamp(0, 1)
        raise ValueError("input_range must be 'zero_one' or 'minus_one_one'.")

    @staticmethod
    def boxes_to_patch_mask(
        boxes_xyxy: torch.Tensor,
        source_size: Tuple[int, int],
        dino_size: Tuple[int, int],
        patch_size: Union[int, Tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """Return True for patch tokens whose centers fall inside any box.

        The patch sequence is row-major: ``flat_index = row * grid_width + col``.
        For a DINO sequence containing prefix tokens, callers must apply this
        mask after those prefix tokens; this method only addresses patch tokens.
        """
        source_height, source_width = source_size
        dino_height, dino_width = dino_size
        if isinstance(patch_size, int):
            patch_height = patch_width = patch_size
        else:
            patch_height, patch_width = patch_size

        grid_height, grid_width = dino_height // patch_height, dino_width // patch_width
        if boxes_xyxy.numel() == 0:
            return torch.zeros(grid_height * grid_width, dtype=torch.bool, device=device)

        # Grounding DINO boxes are post-processed into original-image pixels.
        # Rescale x and y independently because DINO preprocessing may resize
        # the image to a different aspect ratio.
        boxes = boxes_xyxy.to(device=device, dtype=torch.float32).clone()
        boxes[:, [0, 2]] *= dino_width / source_width
        boxes[:, [1, 3]] *= dino_height / source_height

        rows = torch.arange(grid_height, device=device)
        cols = torch.arange(grid_width, device=device)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        center_x = (xx + 0.5) * patch_width
        center_y = (yy + 0.5) * patch_height

        # [num_boxes, grid_height, grid_width], then collapse boxes with any().
        covered = (
            (center_x.unsqueeze(0) >= boxes[:, 0, None, None])
            & (center_x.unsqueeze(0) < boxes[:, 2, None, None])
            & (center_y.unsqueeze(0) >= boxes[:, 1, None, None])
            & (center_y.unsqueeze(0) < boxes[:, 3, None, None])
        )
        return covered.any(dim=0).reshape(-1)

    @staticmethod
    def _processor_tensors_on_device(inputs: Any, device: torch.device) -> Dict[str, Any]:
        return {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }

    @staticmethod
    def _normalized_cxcywh_to_absolute_xyxy(
        normalized_boxes: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        """Convert Grounding DINO ``[cx, cy, w, h]`` boxes to pixel ``xyxy``.

        Grounding DINO emits normalized coordinates in [0, 1]. Conversion must
        happen before patch-grid mapping; treating cx/cy as x1/y1 causes
        oversized or misplaced masks.
        """
        cx, cy, width, height = normalized_boxes.unbind(dim=-1)
        boxes = torch.stack(
            (
                (cx - width * 0.5) * image_width,
                (cy - height * 0.5) * image_height,
                (cx + width * 0.5) * image_width,
                (cy + height * 0.5) * image_height,
            ),
            dim=-1,
        )
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image_height)
        return boxes

    @torch.no_grad()
    def detect_interference(
        self,
        images_01: torch.Tensor,
        text_prompt: str,
    ) -> List[torch.Tensor]:
        """Detect interference and return original-pixel ``[x1,y1,x2,y2]`` boxes.

        This intentionally performs the normalized Grounding DINO box
        conversion explicitly instead of relying on processor post-processing.
        """
        detector_device = next(self.detector.parameters()).device
        image_height, image_width = images_01.shape[-2:]
        # The processor accepts a list of CHW tensors in [0, 1].
        image_list = [image.cpu() for image in images_01]
        inputs = self.detector_processor(
            images=image_list,
            text=[text_prompt] * len(image_list),
            return_tensors="pt",
        )
        inputs = self._processor_tensors_on_device(inputs, detector_device)
        outputs = self.detector(**inputs)

        # outputs.pred_boxes: normalized [cx, cy, width, height].
        # outputs.logits: grounding scores per query and prompt token.
        token_scores = outputs.logits.sigmoid()
        query_scores = token_scores.amax(dim=-1)
        has_prompt_token = (token_scores >= self.text_threshold).any(dim=-1)
        keep = (query_scores >= self.box_threshold) & has_prompt_token

        boxes_per_image = []
        image_area = float(image_height * image_width)
        for normalized_boxes, keep_for_image in zip(outputs.pred_boxes, keep):
            boxes = self._normalized_cxcywh_to_absolute_xyxy(
                normalized_boxes[keep_for_image],
                image_height=image_height,
                image_width=image_width,
            )
            if boxes.numel() > 0:
                box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                # A near-full-frame region is a background false positive, not
                # a screw/rebar. Rejecting it prevents all DINO tokens masking.
                boxes = boxes[box_area <= self.max_interference_area_ratio * image_area]
            boxes_per_image.append(boxes)
        return boxes_per_image

    def forward(
        self,
        generated_images: torch.Tensor,
        text_prompt: str = "metal screw . rusty rebar loop .",
        input_range: str = "minus_one_one",
        return_details: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run detection → masked DINO patch pooling → DI prediction.

        Returns:
            By default, DI shaped ``[batch, 1]``. With ``return_details=True``,
            also returns detector boxes and the boolean flattened patch masks.
        """
        images_01 = self._as_01(generated_images, input_range)
        source_size = (images_01.shape[-2], images_01.shape[-1])
        boxes_per_image = self.detect_interference(images_01, text_prompt)

        dino_device = next(self.dino.parameters()).device
        dino_inputs = self.dino_processor(
            images=[image.cpu() for image in images_01],
            return_tensors="pt",
        )
        dino_inputs = self._processor_tensors_on_device(dino_inputs, dino_device)
        dino_context = torch.no_grad() if self.freeze_dino else nullcontext()
        with dino_context:
            dino_outputs = self.dino(**dino_inputs)

        sequence = dino_outputs.last_hidden_state
        dino_height, dino_width = dino_inputs["pixel_values"].shape[-2:]
        patch_size = getattr(self.dino.config, "patch_size", None)
        if patch_size is None:
            patch_size = getattr(self.dino, "patch_size", None)
        if patch_size is None:
            raise ValueError(
                "Could not determine the DINOv3 patch size. "
                "Expected dino.config.patch_size or dino.patch_size."
            )
        patch_height, patch_width = (
            (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        )
        grid_height, grid_width = dino_height // patch_height, dino_width // patch_width
        patch_count = grid_height * grid_width
        prefix_count = sequence.shape[1] - patch_count
        if prefix_count < 0:
            raise RuntimeError(
                f"DINO returned {sequence.shape[1]} tokens but the {grid_height}x{grid_width} "
                f"patch grid requires {patch_count} patch tokens."
            )

        patch_tokens = sequence[:, prefix_count:, :]
        if patch_tokens.shape[1] != patch_count:
            raise RuntimeError("Unable to isolate DINO patch tokens from the output sequence.")

        patch_masks = torch.stack(
            [
                self.boxes_to_patch_mask(
                    boxes,
                    source_size=source_size,
                    dino_size=(dino_height, dino_width),
                    patch_size=patch_size,
                    device=patch_tokens.device,
                )
                for boxes in boxes_per_image
            ]
        )

        # patch_masks=True means interference. Zero those D-dimensional tokens.
        clean_patch_tokens = patch_tokens.masked_fill(patch_masks.unsqueeze(-1), 0.0)
        valid_patch_count = (~patch_masks).sum(dim=1, keepdim=True).clamp_min(1)
        clean_features = clean_patch_tokens.sum(dim=1) / valid_patch_count
        predicted_di = self.di_predictor(clean_features)

        if return_details:
            return {
                "predicted_di": predicted_di,
                "boxes_xyxy": boxes_per_image,
                "interference_patch_mask": patch_masks,
                "patch_grid_size": (grid_height, grid_width),
                "clean_patch_tokens": clean_patch_tokens,
                "clean_features": clean_features,
            }
        return predicted_di
