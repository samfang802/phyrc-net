"""Stage-3 CFM command line."""

import argparse
import json
import os
from distutils.util import strtobool

from src.model.cfm.condition import scalar_condition_includes_di


DEFAULT_COLUMN_INFO = [
    "id",
    "concreteStrength",
    "transverse_yield",
    "transverse_strength",
    "longitudinal_corner_yield",
    "longitudinal_corner_strength",
    "longitudinal_intermediate_yield",
    "longitudinal_intermediate_strength",
    "lMeasured",
    "axialLoad",
    "fcc",
    "epscc",
    "fcu",
    "epscu",
    "Vn",
    "Vst",
    "reinforcementRatio",
    "volTransReinfRatio",
]

def add_core_arguments(parser):
    parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--train_virtual_repeat', type=int, default=100, help='Virtual expansion factor for train dataset length')
    parser.add_argument('--max_steps', type=int, default=15000, help='Maximum optimizer steps (step-based training clock)')
    parser.add_argument('--warmup_steps', type=int, default=800, help='Warmup steps for step-based scheduler')
    parser.add_argument('--val_check_interval', type=int, default=5, help='Run validation every N training steps')
    parser.add_argument('--use_wandb', action="store_true", help='Enable Weights & Biases logging')
    parser.add_argument('--architecture', type=str, default="CFMADM", help='The architecture of model')
    parser.add_argument(
        '--run_tag',
        type=str,
        default='',
        help='Appended to the result directory and the W&B run name.',
    )

def add_conditioning_arguments(parser):
    parser.add_argument('--latent_conditioning', type=lambda x: bool(strtobool(x)), default=False, help='Whether to use latent condition')
    parser.add_argument('--column_info', nargs='+', type=str, default=DEFAULT_COLUMN_INFO, help='The column information of data')
    parser.add_argument('--direction_conditioning', type=lambda x: bool(strtobool(x)), default=False, help='Whether to use direction')
    parser.add_argument('--include_di_conditioning', type=lambda x: bool(strtobool(x)), default=False, help='Append DI to latent conditioning (hidden_state [+ direction + DI])')
    parser.add_argument(
            '--condition_mode',
            type=str,
            default="full",
            choices=["full", "di_only", "latent_direction", "split", "adagn"],
            help=(
                'Condition mode: full (hidden_state[+direction[+DI]] per flags), '
                'di_only (DI scalar), latent_direction (hidden_state+direction, no DI), '
                'split (zero-init gated AdaGN: e_c=LN(MLP(t[,DI],dir)), e_g=LN(Linear(g)), '
                'h_cond=e_c+alpha*e_g; no cross-attn), '
                'adagn (legacy concat: DI+direction + projected g → AdaGN; no cross-attn)'
            ),
        )
    parser.add_argument(
            '--zero_di_condition',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Drop the DI channel from the split/adagn scalar path (e_c). '
                'Does not keep a zero-padded slot. Damage amount stays on physics g.'
            ),
        )
    parser.add_argument(
            '--cfg_g_drop_prob',
            type=float,
            default=0.0,
            help=(
                'Train-only CFG: per-sample probability of zeroing GRU g (physics). '
                'Does not drop patch XY or direction. 0 disables. Typical 0.1.'
            ),
        )
    parser.add_argument(
            '--gated_fusion_alpha_dim',
            type=str,
            default="scalar",
            choices=["scalar", "vector"],
            help=(
                'For condition_mode=split gated fusion: learnable alpha as a scalar (1,) '
                'or a vector matching cfm_cond_dim. Always zero-initialized.'
            ),
        )
    parser.add_argument('--extractor_path', type=str, default="", help='The path of extractor checkpoint directory')
    parser.add_argument('--regressor_path', type=str, default="", help='The path of DI regressor checkpoint directory')
    parser.add_argument('--resume', type=str, default="", help='Checkpoint path for resuming full Lightning training state')
    parser.add_argument('--data_path', type=str, default="id_list.json", help='The path of data split json under Data/DamagePattern')
    parser.add_argument(
            '--loss_type',
            type=str,
            default="CFMADM",
            help='Result-group name under results/DamagePatternGeneration (default: CFMADM)',
        )
    parser.add_argument('--optimizer', type=str, default="adamw", choices=["adamw", "adam"], help='Optimizer type for training')
    parser.add_argument('--add_DI_loss', type=lambda x: bool(strtobool(x)), default=False, help='Add auxiliary MSE on DI predicted by a regressor from generated images (requires --regressor_path)')
    parser.add_argument('--image_size', type=int, default=128, help='The size of input image')
    parser.add_argument('--img_aug', action="store_true", help='Enable damage image augmentation for training split')
    parser.add_argument('--use_grayscale_input', type=lambda x: bool(strtobool(x)), default=True, help='Convert inputs to grayscale and expand to 3 channels')
    parser.add_argument(
            '--use_clahe',
            type=lambda x: bool(strtobool(x)),
            default=True,
            help='Apply CLAHE before augmentation. Grayscale mode: luminance CLAHE. Color mode (use_grayscale_input=False): Lab-L CLAHE while preserving RGB chroma.',
        )
    parser.add_argument('--clahe_clip_limit', type=float, default=2.0, help='CLAHE clipLimit (OpenCV-compatible)')
    parser.add_argument('--clahe_tile_grid_size', type=str, default='8,8', help='CLAHE tileGridSize as H,W (e.g. 8,8)')
    parser.add_argument('--aug_prob', type=float, default=0.5, help='Probability for color and affine augmentation')
    parser.add_argument('--aug_blur_prob', type=float, default=0.15, help='Probability for Gaussian blur augmentation')
    parser.add_argument('--aug_brightness', type=float, default=0.12, help='ColorJitter brightness delta')
    parser.add_argument('--aug_contrast', type=float, default=0.12, help='ColorJitter contrast delta')
    parser.add_argument('--aug_saturation', type=float, default=0.0, help='ColorJitter saturation delta (ignored when grayscale is enabled)')
    parser.add_argument('--aug_hue', type=float, default=0.0, help='ColorJitter hue delta (ignored when grayscale is enabled)')
    parser.add_argument('--aug_hflip_prob', type=float, default=0.5, help='Probability of horizontal flip (0 disables)')
    parser.add_argument('--aug_blur_kernel', type=int, default=3, help='Gaussian blur kernel size (odd)')
    parser.add_argument('--aug_blur_sigma_min', type=float, default=0.1, help='Gaussian blur minimum sigma')
    parser.add_argument('--aug_blur_sigma_max', type=float, default=0.5, help='Gaussian blur maximum sigma')
    parser.add_argument('--aug_erase_prob', type=float, default=0.25, help='Probability for random erasing (cutout-like augmentation)')
    parser.add_argument('--aug_erase_scale_min', type=float, default=0.01, help='Random erasing minimum erased area ratio')
    parser.add_argument('--aug_erase_scale_max', type=float, default=0.08, help='Random erasing maximum erased area ratio')
    parser.add_argument('--aug_erase_ratio_min', type=float, default=0.3, help='Random erasing minimum aspect ratio')
    parser.add_argument('--aug_erase_ratio_max', type=float, default=3.3, help='Random erasing maximum aspect ratio')
    parser.add_argument('--aug_erase_random_value', action="store_true", help='Use random fill value for random erasing')
    parser.add_argument('--condition_perturb', action="store_true", help='Enable small perturbation on conditioning features during train split')
    parser.add_argument('--condition_perturb_dist', type=str, default="uniform", choices=["uniform", "normal"], help='Distribution for condition perturbation noise')
    parser.add_argument('--condition_perturb_min', type=float, default=0.01, help='Minimum relative perturbation magnitude (e.g. 0.01 = 1%%)')
    parser.add_argument('--condition_perturb_max', type=float, default=0.03, help='Maximum relative perturbation magnitude (e.g. 0.03 = 3%%)')
    parser.add_argument('--condition_perturb_on_hidden', type=lambda x: bool(strtobool(x)), default=True, help='Apply condition perturbation on hidden state')
    parser.add_argument('--condition_perturb_on_info', type=lambda x: bool(strtobool(x)), default=True, help='Apply condition perturbation on physics/info features')
    parser.add_argument('--condition_perturb_on_di', type=lambda x: bool(strtobool(x)), default=True, help='Apply condition perturbation on DI condition')
    parser.add_argument(
            '--g_traj_mix',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Train only: with probability g_traj_mix_max_lambda, replace g by a '
                'same-(exp, direction) neighbor. The image, mask, and DI stay.'
            ),
        )
    parser.add_argument(
            '--g_traj_mix_delta',
            type=int,
            default=2,
            help='Max neighbor offset in labeled frames (uses ±1..delta along the chain)',
        )
    parser.add_argument(
            '--g_traj_mix_max_lambda',
            type=float,
            default=0.3,
            help='Probability of replacing g with a neighbor g. 0.5 uses a neighbor half the time.',
        )
    parser.add_argument(
            '--g_pca_dim',
            type=int,
            default=0,
            help=(
                'If >0, project GRU g with train-fitted PCA (center + SVD, no whitening) '
                'to this many components, then Linear(k, cfm_cond_dim). 0 disables. '
                'Writes/loads result_dir/g_pca.npz. Do not combine with physics_bottleneck_dim.'
            ),
        )
    parser.add_argument(
            '--oracle_dino',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Oracle upper bound: replace GRU g with a frozen DINOv3 embedding of the GT frame '
                '(same physics Linear → gated AdaGN path). physics_dim is set from the encoder.'
            ),
        )
    parser.add_argument(
            '--oracle_dino_model_id',
            type=str,
            default='facebook/dinov3-vits16-pretrain-lvd1689m',
            help='HF repo id of the DINOv3 encoder used by --oracle_dino.',
        )
    parser.add_argument(
            '--oracle_dino_pool',
            type=str,
            default='cls_mean',
            choices=['cls', 'mean', 'cls_mean'],
            help='cls: CLS only. mean: mean patch token only. cls_mean: concat both (2x hidden).',
        )
    parser.add_argument(
            '--oracle_dino_size',
            type=int,
            default=224,
            help='Resize the GT frame to this square size before DINOv3.',
        )
    parser.add_argument(
            '--oracle_dino_pos_blur',
            type=str,
            default='none',
            choices=['none', 'shift', 'shuffle'],
            help=(
                'Blur WHERE in the GT frame before DINOv3. shift: random translate+flip (train only). '
                'shuffle: permute grid x grid tiles (random in train, fixed in eval).'
            ),
        )
    parser.add_argument(
            '--oracle_dino_shift_frac',
            type=float,
            default=0.3,
            help='Max translation as a fraction of image size for pos_blur=shift.',
        )
    parser.add_argument(
            '--oracle_dino_shuffle_grid',
            type=int,
            default=4,
            help='Tiles per side for pos_blur=shuffle (oracle_dino_size must be divisible).',
        )
    parser.add_argument(
            '--use_damage_mask_condition',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Inject object mask from Data/DamagePattern/mask as an extra U-Net condition channel '
                '(train: augmented object mask; val/test/inference: all-ones).'
            ),
        )
    parser.add_argument(
            '--use_object_loss_mask',
            type=lambda x: bool(strtobool(x)),
            default=True,
            help='Load object masks and apply valid-pixel loss masking (concrete=1, rebar=0). Disable when using mask-as-condition only.',
        )

def add_architecture_arguments(parser):
    parser.add_argument('--z_dim', type=int, default=128, help='The size of latent dimension')
    parser.add_argument('--G_h_size', type=int, default=128, help='The size of hidden state in generator')
    parser.add_argument('--D_h_size', type=int, default=128, help='The size of hidden state in discriminator')
    parser.add_argument('--SELU', dest='SELU', type=lambda x: bool(strtobool(x)), help='Whether to use SELU activation function instead of batch normalization and ReLU')
    parser.add_argument('--spectral_D', dest='spectral_D', type=lambda x: bool(strtobool(x)), help='Whether to use spectral normalization in discriminator')
    parser.add_argument('--no_batch_norm', dest='no_batch_norm', type=lambda x: bool(strtobool(x)), help='Whether to use batch normalization in discriminator')
    parser.add_argument('--cfm_model_channels', type=int, default=32, help='Base channel width for CFM ADM U-Net')
    parser.add_argument('--cfm_num_res_blocks', type=int, default=1, help='Number of ResBlocks per level in CFM ADM U-Net')
    parser.add_argument('--cfm_cond_dim', type=int, default=128, help='Shared conditioning hidden dimension for timestep and condition')
    parser.add_argument('--cfm_dropout', type=float, default=0.0, help='Dropout in CFM ADM ResBlocks')
    parser.add_argument('--cfm_mid_attn_heads', type=int, default=4, help='Number of attention heads in CFM U-Net bottleneck')
    parser.add_argument('--cfm_physics_num_tokens', type=int, default=16, help='GRU feature token count for split-conditioning cross-attention')
    parser.add_argument('--cfm_physics_token_dim', type=int, default=128, help='Per-token dimension for GRU physics tokens (should match cfm_cond_dim)')
    parser.add_argument('--physics_dim', type=int, default=1024, help='GRU hidden-state dimension for split/adagn physics path')
    parser.add_argument(
            '--physics_bottleneck_dim',
            type=int,
            default=0,
            help=(
                'If 0 < k < physics_dim, map g with LN-Linear(k)-LN-Linear(cond_dim) '
                'instead of a single Linear(physics_dim, cond_dim). 0 disables.'
            ),
        )
    parser.add_argument(
            '--cfm_physics_adagn_dim',
            type=int,
            default=128,
            help='Projected GRU dim for adagn mode before concat with DI/direction (typically = cfm_cond_dim)',
        )
    parser.add_argument(
            '--use_physics_cross_attn',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Dual-Track Phase-0: inject GRU physics as Cross-Attention tokens at mid/U3/U2, '
                'in addition to gated AdaGN (requires --condition_mode split).'
            ),
        )
    parser.add_argument(
            '--use_vss_l3_u3',
            type=lambda x: bool(strtobool(x)),
            default=True,
            help='Enable Vision Mamba VSS at L3 (down) and doc-U3 @ P/8 (up)',
        )
    parser.add_argument(
            '--use_vss_u2',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='Enable VSS at doc-U2 @ P/4 (decoder long-range)',
        )
    parser.add_argument(
            '--use_vss_l2',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='Enable VSS at doc-L2 @ P/4 (encoder spatial understanding)',
        )
    parser.add_argument(
            '--use_u1_extra_resblock',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='Add one extra AdaGN ResBlock at Decoder U1 (128x128) after skip-merge',
        )
    parser.add_argument(
            '--use_wavelet_unet',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Replace strided U-Net with Haar WaveletUNet: DWT down / IWT up, '
                'LL-only frequency bottleneck, high-frequency skip (Patch-CFM).'
            ),
        )
    parser.add_argument(
            '--wavelet_hf_boost',
            type=float,
            default=0.0,
            help=(
                'Extra L2 on Haar LH/HL/HH of the velocity residual (0 = equal-weight MSE; '
                'no JPEG-style HF suppression). Typical 0.1–0.3.'
            ),
        )
    parser.add_argument(
            '--use_freeu',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='Training-free FreeU on decoder skip merge (backbone scale + skip FFT).',
        )
    parser.add_argument('--freeu_b1', type=float, default=1.2, help='FreeU backbone scale at decoder stage 0')
    parser.add_argument('--freeu_b2', type=float, default=1.4, help='FreeU backbone scale at decoder stage 1')
    parser.add_argument('--freeu_s1', type=float, default=0.9, help='FreeU skip low-freq scale at stage 0')
    parser.add_argument('--freeu_s2', type=float, default=0.2, help='FreeU skip low-freq scale at stage 1')
    parser.add_argument(
            '--freeu_r_thresh',
            type=float,
            default=0.25,
            help='Normalized Fourier radius for skip low-freq mask (after fftshift).',
        )
    parser.add_argument('--grad_clip_max_norm', type=float, default=2.0, help='Max L2 norm for gradient clipping (0 disables clipping)')

def add_sampling_arguments(parser):
    parser.add_argument('--cfm_ode_steps', type=int, default=20, help='ODE integration steps for CFM inference during validation')
    parser.add_argument('--cfm_solver', type=str, default="heun", choices=["heun", "euler"], help='ODE solver for CFM sampling')
    parser.add_argument(
            '--ref_guidance_weight',
            type=float,
            default=0.0,
            help=(
                'Training-free OT reference guidance strength α∈[0,1] during ODE inference. '
                '0 disables. No FFT / frequency filters.'
            ),
        )
    parser.add_argument(
            '--ref_guidance_mode',
            type=str,
            default='latent',
            choices=['latent', 'velocity', 'inversion', 'chain', 'none'],
            help=(
                'Structure reuse at inference: '
                'latent = blend query toward z_t_ref; '
                'velocity = blend v toward (z1_ref-z0); '
                'inversion = invert DI=0 → z0 then forward under target DI; '
                'chain = sequential img2img DI evolution (partial noise at chain_t_start); '
                'none = disabled'
            ),
        )
    parser.add_argument(
            '--chain_t_start',
            type=float,
            default=0.5,
            help=(
                'For ref_guidance_mode=chain: OT partial-noise time in (0,1). '
                'Higher = keep more structure from previous DI; lower = more re-noise.'
            ),
        )
    parser.add_argument(
            '--use_patch_diffusion',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='Enable Patch Diffusion: concat XY coordinate channels and stochastic multi-scale crops during CFM training',
        )
    parser.add_argument(
            '--self_condition',
            '--use_self_conditioning',
            dest='use_self_conditioning',
            type=lambda x: bool(strtobool(x)),
            default=True,
            help=(
                'Architecture bool for classic CFM self-conditioning (V9 / selfcondnoDI default True). '
                'True: extra in_conv channels [z_image | self_cond | coords | mask]; '
                'train uses a scheduled batch-level Bernoulli; ODE inference carries z1-hat. '
                'False: no self-cond channels (incompatible with selfcondnoDI checkpoints).'
            ),
        )
    parser.add_argument(
            '--use_cascade_sr',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Cascaded super-resolution CFM: concat bicubic-upsampled LR as extra channels '
                'and (optionally) predict HR residual. Train at --image_size with LR from '
                '--cascade_lr_size. Mutually exclusive with VAE latent and AR chain.'
            ),
        )
    parser.add_argument(
            '--cascade_lr_size',
            type=int,
            default=128,
            help='Low-res spatial size for cascade SR (downsample HR → this, then upsample back).',
        )
    parser.add_argument(
            '--cascade_noise_std',
            type=float,
            default=0.05,
            help=(
                'Max Gaussian std added to LR cond during training (uniform [0, std]). '
                'Keep small; Stage-1 mismatch is blur, not grain. 0 disables.'
            ),
        )
    parser.add_argument(
            '--cascade_blur_sigma_min',
            type=float,
            default=0.8,
            help='Min Gaussian blur sigma applied on the cascade 128 LR during training.',
        )
    parser.add_argument(
            '--cascade_blur_sigma_max',
            type=float,
            default=2.0,
            help=(
                'Max Gaussian blur sigma on cascade 128 LR (uniform [min, max] in train; '
                'midpoint at val). 0 disables extra blur. Inference Stage-1 path does not blur.'
            ),
        )
    parser.add_argument(
            '--cascade_stage1_checkpoint',
            type=str,
            default='',
            help=(
                'Frozen Stage-1 128 CFM checkpoint for two-stage cascade inference '
                '(train panels and test_cfm). Required when use_cascade_sr=True.'
            ),
        )
    parser.add_argument(
            '--cascade_stage1_result_dir',
            type=str,
            default='',
            help='Run dir with args.json for Stage-1 (default: parent of cascade_stage1_checkpoint).',
        )
    parser.add_argument(
            '--cascade_predict_residual',
            type=lambda x: bool(strtobool(x)),
            default=True,
            help=(
                'If True, CFM target is HR - upsample(downsample(HR)); inference adds residual '
                'onto the LR upsample. Keeps Stage-1 geometry in the LR image.'
            ),
        )
    parser.add_argument(
            '--use_frecas',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Enable FreCaS two-stage training-free 64→128 sampling via sample_with_frecas '
                '(SNR shift + HF velocity scaling + Mid self-attention reuse).'
            ),
        )
    parser.add_argument('--frecas_stage1_size', type=int, default=64)
    parser.add_argument('--frecas_stage2_size', type=int, default=128)
    parser.add_argument(
            '--frecas_stage1_stop_index',
            type=int,
            default=-1,
            help='Stage-1 ODE schedule stop index L (-1 → num_steps // 2).',
        )
    parser.add_argument('--frecas_gamma', type=float, default=2.0, help='SNR shift exponent.')
    parser.add_argument(
            '--frecas_w_h',
            type=float,
            default=2.0,
            help='Stage-2 high-frequency velocity scaling weight.',
        )
    parser.add_argument(
            '--frecas_w_c',
            type=float,
            default=0.6,
            help='Stage-2 Mid self-attention fusion weight (Stage-1 layout fraction).',
        )
    parser.add_argument('--frecas_pool_kernel', type=int, default=3)
    parser.add_argument(
            '--use_mafp',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'MAFP before VAE encode: p_diff translation (Diff-Augment) and p_ada ADA '
                'at fixed probability (no adaptive controller). Concat aug params onto c.'
            ),
        )
    parser.add_argument('--mafp_p_diff', type=float, default=0.1, help='Diff-Augment translation probability')
    parser.add_argument('--mafp_p_ada', type=float, default=0.1, help='Fixed ADA probability per enabled op')
    parser.add_argument(
            '--mafp_tx_max',
            type=float,
            default=0.125,
            help='Max horizontal translation as fraction of width',
        )
    parser.add_argument(
            '--mafp_ty_max',
            type=float,
            default=0.02,
            help='Max vertical translation as fraction of height (keep small for base spall)',
        )

def add_experiments_arguments(parser):
    parser.add_argument(
            '--self_cond_train_prob',
            type=float,
            default=0.5,
            help='Plateau probability of self-conditioning after warmup (default 0.5)',
        )
    parser.add_argument(
            '--self_cond_warmup_start',
            type=int,
            default=2000,
            help='Global step before which self-cond probability is 0',
        )
    parser.add_argument(
            '--self_cond_warmup_end',
            type=int,
            default=4000,
            help='Global step at which self-cond probability reaches self_cond_train_prob (linear ramp)',
        )
    parser.add_argument(
            '--use_ar_self_conditioning',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Enable temporal AR self-conditioning: teacher-force noisy GT z_prev as self_cond '
                'with curriculum dropout. Implies use_self_conditioning and PatternLoopDataset '
                'temporal prev pairing. Mutually exclusive with --use_ar_chain_ot.'
            ),
        )
    parser.add_argument(
            '--use_ar_chain_ot',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Train prev→target chain OT matching inference img2img: '
                'z_start=(1-α)ε+α·z_prev, integrate τ∈[α,1] to z_target (α=chain_t_start). '
                'Enables temporal pairing. Mutually exclusive with --use_ar_self_conditioning.'
            ),
        )
    parser.add_argument(
            '--use_paired_bridge',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Paired OT bridge: t=0 is the previous frame, t=1 is this frame, '
                'v = x_next - x_prev. First frames stay noise-to-image. '
                'No spall mask.'
            ),
        )
    parser.add_argument(
            '--paired_bridge_noise_max',
            type=float,
            default=0.15,
            help='Training only. Mix this much Gaussian into the real previous frame.',
        )
    parser.add_argument(
            '--use_refiner_residual',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Residual temporal refiner. Noise start, S_{t-1} locked as an input channel, '
                'target x_t - S_{t-1}. S_{t-1} comes from a stop-gradient state track: the frozen '
                'single-frame model, optionally chained through this network. Needs --refiner_base_checkpoint.'
            ),
        )
    parser.add_argument(
            '--use_delta_g',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='Add a zero-gated AdaGN branch on g_t - g_{t-1} (refiner only; needs split conditioning).',
        )
    parser.add_argument(
            '--refiner_chain_prob',
            type=float,
            default=0.5,
            help='Per-batch probability of the two-hop chain (frame t-2 -> this network -> S_{t-1}).',
        )
    parser.add_argument(
            '--refiner_state_noise_max',
            type=float,
            default=0.05,
            help='Training only. Max std of Gaussian noise on the S_{t-1} input channel.',
        )
    parser.add_argument(
            '--refiner_zero_init_out',
            type=lambda x: bool(strtobool(x)),
            default=True,
            help='Refiner warm start: zero the output conv, so the initial velocity is 0 (a noise-start sample then returns its own noise).',
        )
    parser.add_argument(
            '--cache_data',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='Decode every image, object mask and info row once at dataset build. Augmentation still runs per call.',
        )
    parser.add_argument(
            '--refiner_base_checkpoint',
            type=str,
            default='',
            help=(
                'Frozen single-frame CFM. With --use_paired_bridge, the bridge start '
                'S_{t-1} is this model\'s own sample for the previous frame instead of the real photo.'
            ),
        )
    parser.add_argument(
            '--refiner_base_result_dir',
            type=str,
            default='',
            help='Run dir with args.json for the refiner base (default: parent of its checkpoint).',
        )
    parser.add_argument(
            '--refiner_roll_prob',
            type=float,
            default=0.15,
            help='Per-batch probability that S_{t-1} comes from a full ODE rollout (else one Euler step).',
        )
    parser.add_argument(
            '--refiner_roll_nfe',
            type=int,
            default=10,
            help='Network evaluations of the full rollout (Heun, so steps = nfe // 2).',
        )
    parser.add_argument(
            '--use_spall_growth_mask',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Coarse accumulated-damage mask M. Target is pooled luminance below '
                'spall_dark_level (fixed, no per-image max). G sees relu(m_t - m_prev) '
                'and is trained outside the old dark cells. Those cells are written '
                'back onto x_prev at every ODE step. Incompatible with AR, '
                'self-conditioning, and cascade SR.'
            ),
        )
    parser.add_argument('--spall_mask_grid', type=int, default=16)
    parser.add_argument('--spall_shift_px', type=int, default=4)
    parser.add_argument('--spall_mask_threshold', type=float, default=0.5)
    parser.add_argument(
            '--spall_dark_level',
            type=float,
            default=0.0,
            help=(
                'Fixed coarse-luminance cutoff in model space (about [-1, 1]). '
                'Cells darker than this are accumulated damage. 0 means mid-gray.'
            ),
        )
    parser.add_argument(
            '--spall_tv_weight',
            type=float,
            default=0.1,
            help='Weight of neighbor total variation added to the mask BCE. 0 disables it.',
        )
    parser.add_argument(
            '--spall_new_cell_weight',
            type=float,
            default=8.0,
            help='BCE weight on cells that are dark now and were not dark in the previous frame.',
        )
    parser.add_argument('--spall_bg_quantile', type=float, default=0.70)
    parser.add_argument('--spall_bg_target', type=float, default=0.30)
    parser.add_argument('--spall_border_gap', type=float, default=0.55)
    parser.add_argument(
            '--spall_seed_margin',
            type=float,
            default=0.2,
            help=(
                'How much darker than spall_dark_level a cell must be to seed m*. '
                'Mildly dark cells above that seed are dropped.'
            ),
        )
    parser.add_argument('--spall_dilate_k', type=int, default=1)
    parser.add_argument(
            '--spall_close_k',
            type=int,
            default=3,
            help=(
                'Odd kernel for morphological closing on the inferred extent '
                'before the growth mask. 1 disables. Fills gaps inside the spall.'
            ),
        )
    parser.add_argument('--spall_feather_px', type=float, default=4.0)
    parser.add_argument('--spall_pool_noise_std', type=float, default=0.05)
    parser.add_argument('--spall_loss_weight', type=float, default=1.0)
    parser.add_argument(
            '--spall_mask_only',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Train only the coarse mask head. Skips the velocity net. '
                'Use this to check m against the accumulated-extent target.'
            ),
        )
    parser.add_argument(
            '--freeze_spall_mask',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='Keep the loaded mask head fixed while training G.',
        )
    parser.add_argument(
            '--spall_cond_prob',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='Feed M sigmoid to G. Skips the hard mask and m* target.',
        )
    parser.add_argument(
            '--spall_soft_paint',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Train and sample G on the feathered paint weight. Old spall and '
                'healthy concrete stay on the previous frame. Incompatible with '
                '--spall_cond_prob.'
            ),
        )
    parser.add_argument(
            '--spall_lock_healthy',
            type=lambda x: bool(strtobool(x)),
            default=True,
            help=(
                'With --spall_soft_paint, also lock coarse cells that are bright '
                'relative to this frame p70. False locks only old spall and the '
                'left/right backdrop.'
            ),
        )
    parser.add_argument(
            '--spall_healthy_margin',
            type=float,
            default=0.10,
            help=(
                'A coarse cell is healthy when it is at least this much above '
                'its own frame p70, measured downward. 0.10 locks concrete that '
                'is up to 0.10 darker than that percentile.'
            ),
        )
    parser.add_argument(
            '--spall_ctx_noise_max',
            type=float,
            default=0.15,
            help=(
                'Training only. Max OT mix toward noise on the previous frame '
                'seen by G. 0 disables. M still uses the real previous frame.'
            ),
        )
    parser.add_argument(
            '--spall_mask_ckpt',
            type=str,
            default='',
            help='Checkpoint whose spall_mask_head weights initialize M.',
        )
    parser.add_argument(
            '--velocity_init',
            type=str,
            default='',
            help='Checkpoint whose velocity_net weights initialize G. Not a Lightning resume.',
        )
    parser.add_argument(
            '--ar_drop_prob_start',
            type=float,
            default=0.5,
            help='AR self-cond drop probability at step 0 (linear decay toward ar_drop_prob_end)',
        )
    parser.add_argument(
            '--ar_drop_prob_end',
            type=float,
            default=0.15,
            help='AR self-cond drop probability at/after ar_drop_prob_end_step',
        )
    parser.add_argument(
            '--ar_drop_prob_end_step',
            type=int,
            default=80000,
            help='Global step at which AR drop_prob reaches ar_drop_prob_end',
        )
    parser.add_argument(
            '--ar_noise_std_min',
            type=float,
            default=0.05,
            help='Min per-sample Gaussian std added to GT z_prev self_cond',
        )
    parser.add_argument(
            '--ar_noise_std_max',
            type=float,
            default=0.10,
            help='Max per-sample Gaussian std added to GT z_prev self_cond',
        )
    parser.add_argument(
            '--patch_size_probs',
            type=str,
            default='0.5,0.3,0.2',
            help=(
                'Patch Diffusion full/half/quarter probabilities. Used when '
                '--use_patch_prob_schedule False, and as the two-segment mid/plateau.'
            ),
        )
    parser.add_argument(
            '--use_patch_prob_schedule',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='Enable a time-varying patch-size schedule (see --patch_schedule_mode).',
        )
    parser.add_argument(
            '--patch_schedule_mode',
            type=str,
            default='two_segment',
            choices=['two_segment', 'pixel_mass'],
            help=(
                'two_segment: start→mid hold→end probability triples. '
                'pixel_mass: linearly interpolate pixel mass M over all steps '
                '(half-first, quarter capped).'
            ),
        )
    parser.add_argument(
            '--patch_size_probs_start',
            type=str,
            default='0.8,0.2,0.0',
            help='Start full/half/quarter probs when --patch_schedule_mode two_segment',
        )
    parser.add_argument(
            '--patch_size_probs_end',
            type=str,
            default='0.7,0.25,0.05',
            help='End full/half/quarter probs when --patch_schedule_mode two_segment',
        )
    parser.add_argument(
            '--patch_schedule_mid_frac',
            type=float,
            default=0.2,
            help='two_segment: fraction of max_steps at which the start→mid ramp finishes',
        )
    parser.add_argument(
            '--patch_schedule_end_frac',
            type=float,
            default=0.7,
            help='two_segment: fraction of max_steps at which the mid→end ramp starts',
        )
    parser.add_argument(
            '--patch_pixel_mass_start',
            type=float,
            default=0.55,
            help='pixel_mass mode: M at step 0 (static 0.5/0.3/0.2 is ~0.59)',
        )
    parser.add_argument(
            '--patch_pixel_mass_end',
            type=float,
            default=0.64,
            help='pixel_mass mode: M at max_steps (static 0.5/0.3/0.2 is ~0.59)',
        )
    parser.add_argument(
            '--patch_quarter_prob_start',
            type=float,
            default=0.15,
            help='pixel_mass mode: p_quarter at step 0',
        )
    parser.add_argument(
            '--patch_quarter_prob_end',
            type=float,
            default=0.10,
            help='pixel_mass mode: p_quarter at max_steps (keep near 50/30/20; 32px is crop not texture)',
        )
    parser.add_argument(
            '--use_patch_t_schedule',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'After the iteration schedule, move full→half from each sample CFM t '
                '(t≈0 noise keeps full; t→1 allows more 64px half), then average '
                'the batch probs for one shared crop. Never boosts quarter.'
            ),
        )
    parser.add_argument(
            '--patch_t_crop_min',
            type=float,
            default=0.5,
            help='CFM t below this keeps the iteration-only probs; above, linear extra half.',
        )
    parser.add_argument(
            '--patch_t_half_boost',
            type=float,
            default=0.20,
            help='Extra p_half taken from p_full at t=1 when --use_patch_t_schedule True.',
        )
    parser.add_argument(
            '--use_velocity_min_snr_weighting',
            type=lambda x: bool(strtobool(x)),
            default=True,
            help='Apply Min-SNR-gamma weighting on CFM velocity loss',
        )
    parser.add_argument(
            '--min_snr_gamma',
            type=float,
            default=5.0,
            help='Gamma value used in Min-SNR weighting (clamp max)',
        )
    parser.add_argument(
            '--use_undamaged_ref',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Other-specimen undamaged GT as a 128-d AdaGN branch with independent '
                'zero-init alpha_ref (h = e_c + alpha*e_g + alpha_ref*e_ref). Never same exp. '
                'Does not grow condition_dim. Train mask concat only; val/test mask is all-ones. '
                'Pixel Patch CFM only.'
            ),
        )
    parser.add_argument('--undamaged_ref_dim', type=int, default=128)
    parser.add_argument('--undamaged_ref_di_max', type=float, default=0.05)
    parser.add_argument('--undamaged_ref_tx_max', type=float, default=0.08)
    parser.add_argument('--undamaged_ref_ty_max', type=float, default=0.02)
    parser.add_argument('--undamaged_ref_p_diff', type=float, default=0.5)
    parser.add_argument('--undamaged_ref_p_ada', type=float, default=0.3)
    parser.add_argument(
            '--use_gamma_fm_loss',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Weight per-sample CFM MSE by batch min-max of Var(z1) over (C,H,W) '
                '(patch complexity proxy). Floor at --gamma_fm_eps then re-mean to 1.'
            ),
        )
    parser.add_argument(
            '--gamma_fm_eps',
            type=float,
            default=0.1,
            help='Lower bound for min-max gamma weights before mean-normalization (smooth patches still get gradient)',
        )
    parser.add_argument(
            '--use_hf_aux_loss',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help=(
                'Late-t pixel aux on OT clean estimate x1_hat=z_t+(1-t)v: '
                'Sobel edge match + high-pass match to GT image (no crack/spall mask). '
                'Does not change velocity target u=x1-eps. Pixel-space CFM only.'
            ),
        )
    parser.add_argument(
            '--hf_aux_lambda',
            type=float,
            default=0.02,
            help='Weight of late-t HF/edge aux vs velocity FM loss',
        )
    parser.add_argument(
            '--hf_aux_t_min',
            type=float,
            default=0.7,
            help='Only apply HF aux for samples with t >= this threshold',
        )
    parser.add_argument(
            '--hf_aux_edge_weight',
            type=float,
            default=1.0,
            help='Relative weight of Sobel edge L1 inside HF aux',
        )
    parser.add_argument(
            '--hf_aux_hp_weight',
            type=float,
            default=0.5,
            help='Relative weight of high-pass residual L1 inside HF aux',
        )
    parser.add_argument(
            '--hf_aux_hp_kernel',
            type=int,
            default=3,
            help='Box-filter kernel size for high-pass residual (odd)',
        )
    parser.add_argument('--use_sd3_vae_latent', type=lambda x: bool(strtobool(x)), default=False, help='Train/sample CFM in SD3 VAE latent space instead of pixel space')
    parser.add_argument('--sd3_vae_model_id', type=str, default="stabilityai/stable-diffusion-3-medium-diffusers", help='Diffusers model id containing SD3 VAE weights')
    parser.add_argument('--sd3_vae_subfolder', type=str, default="vae", help='Subfolder for SD3 VAE weights in the model repository')
    parser.add_argument('--sd3_vae_use_fp16', type=lambda x: bool(strtobool(x)), default=True, help='Use fp16 for SD3 VAE on CUDA to reduce memory')
    parser.add_argument('--lr_schedule', type=str, default="constant", choices=["constant", "warmup_cosine"], help='Learning-rate schedule type')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs for warmup_cosine schedule')
    parser.add_argument('--min_lr_ratio', type=float, default=0.1, help='Final LR ratio for warmup_cosine schedule')
    parser.add_argument('--val_every_n_epoch', type=int, default=10, help='Run validation every N epochs')
    parser.add_argument('--inference_every_n_epochs', type=int, default=10, help='Save/log multi-sample CFM inference images every N epochs')
    parser.add_argument('--inference_num_images', type=int, default=4, help='Number of fixed DI values per split (train + val) in CFM inference panel')
    parser.add_argument('--inference_anchor_exp', type=str, default="", help='Deprecated alias for --inference_val_anchor_exp')
    parser.add_argument(
            '--inference_train_anchor_exp',
            type=str,
            default="R615",
            help='Fixed bridge column (exp id) for train-set inference; must exist in train split',
        )
    parser.add_argument(
            '--inference_val_anchor_exp',
            type=str,
            default="R307",
            help='Fixed bridge column (exp id) for val GT/inference panel; must exist in val split',
        )
    parser.add_argument('--inference_anchor_timestep', type=int, default=-1, help='Optional fixed timestep for inference anchor (-1 for auto)')
    parser.add_argument(
            '--metrics_every_n_epochs',
            type=int,
            default=0,
            help='Deprecated: FID/PSNR/SSIM are disabled during CFM training (kept for CLI compat)',
        )
    parser.add_argument('--checkpoint_every_n_epochs', type=int, default=0, help='Save checkpoint_epoch_X.pt every N epochs (set 0 to disable)')
    parser.add_argument(
            '--checkpoint_every_n_steps',
            type=int,
            default=0,
            help='Save checkpoint_step_XXXXXX.pt every N training steps (set 0 to disable)',
        )
    parser.add_argument(
            '--checkpoint_start_step',
            type=int,
            default=10000,
            help='Only save periodic step checkpoints when global_step >= this value',
        )
    parser.add_argument(
            '--use_early_stopping',
            type=lambda x: bool(strtobool(x)),
            default=True,
            help='Enable EarlyStopping callback (requires early_stop_patience > 0)',
        )
    parser.add_argument('--early_stop_patience', type=int, default=0, help='Early stopping patience on val/loss (ignored if use_early_stopping=False)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0, help='Minimum val/loss improvement to reset early stopping')
    parser.add_argument(
            '--reset_early_stop_patience',
            type=lambda x: bool(strtobool(x)),
            default=True,
            help='When resuming from --resume, reset EarlyStopping wait_count to 0 (default True)',
        )
    parser.add_argument(
            '--reset_early_stop_best_score',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='Also reset EarlyStopping best_score on resume (default False; keeps historical best val/loss)',
        )
    parser.add_argument('--fixed_di_values', type=str, default="0.0,0.4,0.75,1.0", help='Comma-separated DI values used for fixed CFM inference panel')
    parser.add_argument(
            '--laplacian_band',
            type=str,
            default='d1',
            choices=['d1', 'd2', 'd3'],
            help='Laplacian-CFM band: d1=128 HF residual, d2=64 mid residual, d3=32 LF topology',
        )
    parser.add_argument(
            '--laplacian_dummy',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='Use a 3-layer conv stub instead of D1/D2/D3 velocity nets (loop smoke test)',
        )
    parser.add_argument(
            '--use_laplacian_patch',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='Laplacian-CFM: random spatial crop of (x_t, v_t). Off = always full band map.',
        )

def parse_args():
    desc = "Pytorch implementation of Hysteretic Loop Based Prediction and Generation"
    parser = argparse.ArgumentParser(description=desc)
    add_core_arguments(parser)
    add_conditioning_arguments(parser)
    add_architecture_arguments(parser)
    add_sampling_arguments(parser)
    add_experiments_arguments(parser)
    return parser.parse_args()

def check_args(args, current_time):
    result_dir = ["./results", "DamagePatternGeneration", args.loss_type]
    tag = str(getattr(args, "run_tag", "") or "").strip()
    tag = tag.replace("\\", "_").replace("/", "_").replace(" ", "_")
    folder = f"{current_time}_{tag}" if tag else current_time
    args.result_dir = os.path.join(*result_dir, folder)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(os.path.join(args.result_dir, "data_check")):
        os.makedirs(os.path.join(args.result_dir, "data_check"))
    return args


def save_args(args):
    dict_args = vars(args)
    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(dict_args, f, indent=2)


def set_dataset_args(args, train_dataloader):
    batch = next(iter(train_dataloader))
    if isinstance(batch, dict):
        # AR temporal pairing returns a named dict batch.
        image = batch["image_target"] if "image_target" in batch else batch["image"]
        hidden_state = (
            batch["hidden_state_target"]
            if "hidden_state_target" in batch
            else batch["hidden_state"]
        )
        direction = (
            batch["direction_target"] if "direction_target" in batch else batch["direction"]
        )
        DI = batch["DI_target"] if "DI_target" in batch else batch["DI"]
        info = batch["info"]
    else:
        image, hidden_state, direction, exp, timestep, DI, info = batch[:7]
    if getattr(args, "condition_mode", "full") == "di_only":
        args.condition_dim = DI.shape[-1]
        return
    if getattr(args, "condition_mode", "full") in ("split", "adagn"):
        args.condition_dim = 0
        if scalar_condition_includes_di(
            getattr(args, "include_di_conditioning", False),
            getattr(args, "zero_di_condition", False),
            keep_zero_di_slot=getattr(args, "keep_zero_di_slot", False),
        ):
            args.condition_dim += DI.shape[-1]
        if args.direction_conditioning:
            args.condition_dim += direction.shape[-1]
        if args.condition_dim <= 0:
            raise ValueError("split/adagn condition_dim is 0: enable DI or direction")
        args.physics_dim = hidden_state.shape[-1]
        return
    if getattr(args, "condition_mode", "full") == "latent_direction":
        args.condition_dim = hidden_state.shape[-1] + direction.shape[-1]
        return
    if args.latent_conditioning:
        args.condition_dim = hidden_state.shape[-1]
        if args.direction_conditioning:
            args.condition_dim += direction.shape[-1]
        if args.include_di_conditioning:
            args.condition_dim += DI.shape[-1]
    else:
        args.condition_dim = info.shape[-1]
        if args.direction_conditioning:
            args.condition_dim += direction.shape[-1]
        args.condition_dim += DI.shape[-1]

