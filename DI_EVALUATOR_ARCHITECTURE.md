# Interference-Masked Damage Index Evaluator

## Purpose

`src/model/ThirdPartyDIEvaluator.py` evaluates the Damage Index (DI) of generated RC-column images while suppressing image regions that contain non-structural interference, such as metal screws or steel rebars.

The evaluator does not load DINOv3 itself. It receives the project’s already-loaded DINOv3 model and its matching processor. Grounding DINO remains responsible only for open-vocabulary interference detection.

> The DI MLP must be trained with DI-labelled data, or loaded from a trained checkpoint. Grounding DINO and DINOv3 alone cannot predict DI.

## Architecture

```mermaid
flowchart LR
    I[Generated RC image<br/>B × 3 × H × W] --> R[Convert range<br/>[-1, 1] → [0, 1]]
    R --> GD[Grounding DINO<br/>prompt: metal screw. steel rebar.]
    GD --> BX[Interference boxes<br/>B × N × 4 in original pixels]
    R --> DP[DINOv3 processor]
    DP --> DV[DINOv3 Vision Transformer]
    DV --> PT[Patch tokens<br/>B × P × D]
    BX --> MAP[Map boxes to<br/>DINO patch grid]
    MAP --> PM[Boolean patch mask<br/>B × P]
    PT --> Z[Zero masked tokens]
    PM --> Z
    Z --> POOL[Masked mean pooling<br/>B × D]
    POOL --> MLP[Trained DI MLP]
    MLP --> DI[Predicted DI<br/>B × 1]
```

### Components

| Component | Responsibility | Trainable during DI-MLP training |
|---|---|---|
| Grounding DINO | Detect prompt-matched interference boxes | No |
| DINOv3 | Extract visual patch-token features | No by default |
| Patch masking | Set interference-token feature vectors to zero | No |
| `DamageIndexMLP` | Map clean pooled features to DI | Yes |

## Input and Output Contract

```python
result = evaluator(
    generated_images,                       # Tensor [B, 3, H, W]
    text_prompt="metal screw. steel rebar.",
    input_range="minus_one_one",            # or "zero_one"
    return_details=True,
)
```

| Field | Shape | Meaning |
|---|---:|---|
| `generated_images` | `[B, 3, H, W]` | Generated RGB images |
| `result["predicted_di"]` | `[B, 1]` | Predicted DI |
| `result["boxes_xyxy"][b]` | `[N_b, 4]` | Detected `[x1, y1, x2, y2]` boxes in original-image pixels |
| `result["interference_patch_mask"]` | `[B, P]` | `True` where a DINO patch is interference |
| `result["clean_patch_tokens"]` | `[B, P, D]` | Patch features after interference features are zeroed |

## Patch-Token Mapping

Grounding DINO produces boxes in original image coordinates:

```text
box = [x1, y1, x2, y2]
original image = Hsrc × Wsrc
DINO input      = Hdino × Wdino
```

The evaluator rescales each box independently:

```text
x' = x × Wdino / Wsrc
y' = y × Hdino / Hsrc
```

For patch size `(Ph, Pw)`, the DINO grid dimensions are:

```text
Gh = Hdino / Ph
Gw = Wdino / Pw
P  = Gh × Gw
```

The center of patch `(row, col)` is:

```text
cx = (col + 0.5) × Pw
cy = (row + 0.5) × Ph
```

A patch is masked when its center lies inside any rescaled interference box:

```text
x1' ≤ cx < x2'  and  y1' ≤ cy < y2'
```

DINO patch tokens are flattened in row-major order:

```text
flat_patch_index = row × Gw + col
sequence_index   = prefix_count + flat_patch_index
```

`prefix_count` accounts for CLS and any DINOv3 register/prefix tokens. Prefix tokens are not masked; only the isolated patch-token sequence is modified.

## Pseudocode

```text
function FORWARD(generated_images, prompt):
    images_01 = convert_to_unit_range(generated_images)

    boxes_per_image = GroundingDINO.detect(images_01, prompt)

    dino_pixels = DINOv3Processor(images_01)
    sequence = DINOv3(dino_pixels).last_hidden_state

    patch_tokens = remove_prefix_tokens(sequence)
    patch_mask = empty_boolean_mask(batch_size, num_patch_tokens)

    for each image b:
        for each detected box in boxes_per_image[b]:
            box_dino = rescale_box(box, original_size, dino_input_size)

            for each patch location (row, col):
                center = patch_center(row, col, patch_size)
                flat_index = row * grid_width + col
                if center is inside box_dino:
                    patch_mask[b, flat_index] = True

    clean_patch_tokens = set_to_zero(patch_tokens, where=patch_mask)
    valid_count = max(sum(not patch_mask), 1)
    clean_feature = sum(clean_patch_tokens) / valid_count

    return DI_MLP(clean_feature)
```

## Trace Code

The following trace corresponds to `InterferenceMaskedDIEvaluator.forward()`.

```python
# 1. Normalize model output for both external vision processors.
images_01 = self._as_01(generated_images, input_range)

# 2. Grounding DINO returns boxes in original image coordinates.
boxes_per_image = self.detect_interference(images_01, text_prompt)

# 3. Reuse the already-loaded DINOv3 processor and model.
dino_inputs = self.dino_processor(
    images=[image.cpu() for image in images_01],
    return_tensors="pt",
)
dino_outputs = self.dino(**dino_inputs)
sequence = dino_outputs.last_hidden_state

# 4. Derive the patch grid and remove non-spatial prefix tokens.
grid_height = dino_height // patch_height
grid_width = dino_width // patch_width
patch_count = grid_height * grid_width
prefix_count = sequence.shape[1] - patch_count
patch_tokens = sequence[:, prefix_count:, :]  # [B, P, D]

# 5. Construct one flattened boolean mask per image.
patch_masks = torch.stack([
    self.boxes_to_patch_mask(
        boxes,
        source_size=source_size,
        dino_size=(dino_height, dino_width),
        patch_size=patch_size,
        device=patch_tokens.device,
    )
    for boxes in boxes_per_image
])  # [B, P]

# 6. Suppress interference features and average valid tokens only.
clean_patch_tokens = patch_tokens.masked_fill(
    patch_masks.unsqueeze(-1), 0.0
)
valid_patch_count = (~patch_masks).sum(dim=1, keepdim=True).clamp_min(1)
clean_features = clean_patch_tokens.sum(dim=1) / valid_patch_count

# 7. Predict one DI value per image.
predicted_di = self.di_predictor(clean_features)  # [B, 1]
```

## Example Initialization

```python
from src.model.ThirdPartyDIEvaluator import InterferenceMaskedDIEvaluator

# dinov3_model and dinov3_processor are loaded elsewhere in the project.
evaluator = InterferenceMaskedDIEvaluator(
    dino=dinov3_model,
    dino_processor=dinov3_processor,
    box_threshold=0.35,
    text_threshold=0.25,
).cuda().eval()
```

## Validation Checks

- The generated image must have shape `[B, 3, H, W]`.
- `DINOv3.config.hidden_size` or `DINOv3.config.embed_dim` must be available.
- `DINOv3.config.patch_size` (or `DINOv3.patch_size`) must be available.
- The DINO output must contain at least `Gh × Gw` patch tokens.
- If no interference is detected, the patch mask is all `False` and all patch tokens participate in pooling.
- If every patch is masked, the denominator is clamped to `1` to avoid division by zero; this case should be monitored because it produces a zero pooled feature.
