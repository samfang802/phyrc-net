"""
DINOv3 + SD3 VAE reconstruction test for RC-column crack details.

How to use:
1) Update --image_path to your RC damage image.
2) Run:
   python3 test_dinov3_vae_cracks.py --image_path "Data/DamagePattern/images/your_image.jpg"

Optional:
- Change output path with --output_path.
- Change resize target with --height and --width.
- If your environment needs auth for model weights, login first:
  huggingface-cli login
"""

import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from diffusers import AutoencoderKL
    from transformers import AutoImageProcessor, AutoModel
except ImportError as exc:
    raise ImportError(
        "Please install dependencies first: pip install diffusers transformers"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test DINOv3 consistency after SD3 VAE reconstruction."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to a single RC damage image for reconstruction test.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="dinov3_vae_reconstruction_result.png",
        help="Output path for side-by-side comparison image.",
    )
    parser.add_argument(
        "--sd3_model_id",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Diffusers model repo id that contains SD3 VAE.",
    )
    parser.add_argument(
        "--vae_subfolder",
        type=str,
        default="vae",
        help="Subfolder containing VAE weights in the model repo.",
    )
    parser.add_argument(
        "--dino_model_id",
        type=str,
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="Hugging Face repo id for DINOv3 model.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Resize height before encoding.")
    parser.add_argument("--width", type=int, default=1024, help="Resize width before encoding.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference, e.g., cuda or cpu.",
    )
    return parser.parse_args()


def preprocess_image(
    image_path: str, height: int, width: int, device: str
) -> Tuple[Image.Image, torch.Tensor]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    resized = image.resize((width, height), Image.Resampling.LANCZOS)

    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return resized, tensor.to(device)


def tensor_to_pil_image(image_tensor: torch.Tensor) -> Image.Image:
    x = image_tensor.detach().cpu().float()
    x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    x = x.squeeze(0).permute(1, 2, 0).numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return Image.fromarray(x)


def latent_encode(vae: AutoencoderKL, image_tensor: torch.Tensor) -> torch.Tensor:
    posterior = vae.encode(image_tensor).latent_dist
    latent = posterior.mean
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    shift = float(getattr(vae.config, "shift_factor", 0.0))
    return (latent - shift) * scaling


def latent_decode(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    shift = float(getattr(vae.config, "shift_factor", 0.0))
    model_input = latents / scaling + shift
    return vae.decode(model_input).sample


def build_pixel_error_heatmap(original_tensor: torch.Tensor, reconstructed_tensor: torch.Tensor) -> Image.Image:
    abs_diff = (original_tensor - reconstructed_tensor).abs()
    abs_diff = abs_diff.mean(dim=1, keepdim=False).squeeze(0)
    abs_diff = abs_diff / (abs_diff.max() + 1e-8)
    heat = (abs_diff.detach().cpu().numpy() * 255.0).astype(np.uint8)
    r = heat
    g = np.minimum(255, (heat * 1.5).astype(np.uint8))
    b = np.zeros_like(heat, dtype=np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1))


def _resolve_patch_hw(dino_model) -> Tuple[int, int]:
    patch_size = getattr(dino_model.config, "patch_size", None)
    if patch_size is None:
        patch_size = getattr(dino_model, "patch_size", None)
    if patch_size is None:
        raise ValueError("Cannot determine DINO patch size from model/config.")
    if isinstance(patch_size, (tuple, list)):
        return int(patch_size[0]), int(patch_size[1])
    return int(patch_size), int(patch_size)


def _extract_patch_tokens(outputs: torch.Tensor, expected_patch_tokens: int) -> torch.Tensor:
    if outputs.ndim != 3:
        raise ValueError(f"Unexpected DINO output shape: {tuple(outputs.shape)}")
    if outputs.shape[1] < expected_patch_tokens:
        raise ValueError(
            f"DINO sequence too short: seq={outputs.shape[1]}, expected patches={expected_patch_tokens}"
        )
    # DINOv3 sequence may include cls/register tokens before patch tokens.
    return outputs[:, -expected_patch_tokens:, :]


def build_dino_error_heatmap(
    dino_processor,
    dino_model,
    original_pil: Image.Image,
    reconstructed_pil: Image.Image,
    device: str,
) -> Image.Image:
    batch = dino_processor(images=[original_pil, reconstructed_pil], return_tensors="pt")
    pixel_values = batch["pixel_values"].to(device)
    h_in, w_in = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
    patch_h, patch_w = _resolve_patch_hw(dino_model)
    grid_h, grid_w = h_in // patch_h, w_in // patch_w
    expected_patch_tokens = grid_h * grid_w

    with torch.no_grad():
        outputs = dino_model(pixel_values=pixel_values).last_hidden_state

    patch_tokens = _extract_patch_tokens(outputs, expected_patch_tokens=expected_patch_tokens)
    feat_orig = patch_tokens[0]
    feat_recon = patch_tokens[1]
    token_diff = torch.norm(feat_orig - feat_recon, p=2, dim=-1)

    diff_map = token_diff.view(1, 1, grid_h, grid_w)
    diff_map = F.interpolate(
        diff_map,
        size=(original_pil.height, original_pil.width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    diff_map = diff_map / (diff_map.max() + 1e-8)

    heat = (diff_map.detach().cpu().numpy() * 255.0).astype(np.uint8)
    r = heat
    g = np.minimum(255, (heat * 1.5).astype(np.uint8))
    b = np.zeros_like(heat, dtype=np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1))


def save_side_by_side(
    original_img: Image.Image,
    reconstructed_img: Image.Image,
    pixel_error_map: Image.Image,
    dino_error_map: Image.Image,
    output_path: str,
) -> None:
    if not (
        original_img.size
        == reconstructed_img.size
        == pixel_error_map.size
        == dino_error_map.size
    ):
        raise ValueError(
            "Image sizes do not match for concatenation: "
            f"original={original_img.size}, reconstructed={reconstructed_img.size}, "
            f"pixel_error={pixel_error_map.size}, dino_error={dino_error_map.size}"
        )

    w, h = original_img.size
    canvas = Image.new("RGB", (w * 4, h))
    canvas.paste(original_img, (0, 0))
    canvas.paste(reconstructed_img, (w, 0))
    canvas.paste(pixel_error_map, (w * 2, 0))
    canvas.paste(dino_error_map, (w * 3, 0))
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError(f"Height/width must be positive, got height={args.height}, width={args.width}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no CUDA device is available.")

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"[Info] Loading SD3 VAE from: {args.sd3_model_id}/{args.vae_subfolder}")
    print(f"[Info] Loading DINOv3 from: {args.dino_model_id}")
    print(f"[Info] Device: {device}, dtype: {dtype}")

    try:
        vae = AutoencoderKL.from_pretrained(
            args.sd3_model_id,
            subfolder=args.vae_subfolder,
            torch_dtype=dtype,
        ).to(device)
        vae.eval()

        dino_processor = AutoImageProcessor.from_pretrained(args.dino_model_id)
        dino_model = AutoModel.from_pretrained(args.dino_model_id).to(device).eval()

        original_pil, image_tensor = preprocess_image(args.image_path, args.height, args.width, device)

        with torch.no_grad():
            if device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=dtype):
                    latents = latent_encode(vae, image_tensor)
                    recon_tensor = latent_decode(vae, latents)
            else:
                latents = latent_encode(vae, image_tensor)
                recon_tensor = latent_decode(vae, latents)

        reconstructed_pil = tensor_to_pil_image(recon_tensor)
        pixel_error_map = build_pixel_error_heatmap(image_tensor, recon_tensor)
        dino_error_map = build_dino_error_heatmap(
            dino_processor=dino_processor,
            dino_model=dino_model,
            original_pil=original_pil,
            reconstructed_pil=reconstructed_pil,
            device=device,
        )

        save_side_by_side(
            original_img=original_pil,
            reconstructed_img=reconstructed_pil,
            pixel_error_map=pixel_error_map,
            dino_error_map=dino_error_map,
            output_path=args.output_path,
        )
        print(f"[Done] Saved comparison image to: {args.output_path}")
        print("[Done] Layout: [Original | Reconstructed | Pixel Error | DINOv3 Error]")

    except Exception as exc:
        raise RuntimeError(f"DINOv3 + SD3 VAE reconstruction test failed: {exc}") from exc
    finally:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
