import argparse, os, json
from distutils.util import strtobool

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


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of Hysteretic Loop Based Prediction and Generation"
    parser = argparse.ArgumentParser(description=desc)  
    parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--use_wandb', action="store_true", help='Enable Weights & Biases logging')
    parser.add_argument('--architecture', type=str, default="HystereticCGAN", help='The architecture of model')
    
    parser.add_argument('--latent_conditioning', type=lambda x: bool(strtobool(x)), default=False, help='Whether to use latent condition')
    parser.add_argument('--column_info', nargs='+', type=str, default=DEFAULT_COLUMN_INFO, help='The column information of data')
    parser.add_argument('--direction_conditioning', type=lambda x: bool(strtobool(x)), default=False, help='Whether to use direction')
    parser.add_argument('--include_di_conditioning', type=lambda x: bool(strtobool(x)), default=False, help='Append DI to latent conditioning (hidden_state [+ direction + DI])')
    parser.add_argument('--extractor_path', type=str, default="", help='The path of extractor checkpoint directory')
    parser.add_argument('--regressor_path', type=str, default="", help='The path of DI regressor checkpoint directory')
    parser.add_argument('--resume', type=str, default="", help='Checkpoint path for resuming full Lightning training state')
    parser.add_argument('--data_path', type=str, default="id_list.json", help='The path of data split json under Data/DamagePattern')
    
    parser.add_argument('--loss_type', type=str, default="SGAN", help='The type of loss function')
    parser.add_argument('--optimizer', type=str, default="adamw", choices=["adamw", "adam"], help='Optimizer type for training')
    parser.add_argument('--add_DI_loss', type=lambda x: bool(strtobool(x)), help='Whether to add DI loss')
    parser.add_argument('--image_size', type=int, default=128, help='The size of input image')
    parser.add_argument('--img_aug', action="store_true", help='Enable damage image augmentation for training split')
    parser.add_argument('--aug_prob', type=float, default=0.5, help='Probability for color and affine augmentation')
    parser.add_argument('--aug_blur_prob', type=float, default=0.15, help='Probability for Gaussian blur augmentation')
    parser.add_argument('--aug_brightness', type=float, default=0.12, help='Color jitter brightness delta')
    parser.add_argument('--aug_contrast', type=float, default=0.12, help='Color jitter contrast delta')
    parser.add_argument('--aug_saturation', type=float, default=0.08, help='Color jitter saturation delta')
    parser.add_argument('--aug_hue', type=float, default=0.02, help='Color jitter hue delta')
    parser.add_argument('--aug_translate', type=float, default=0.02, help='Random affine translate ratio')
    parser.add_argument('--aug_scale_min', type=float, default=0.98, help='Random affine minimum scale')
    parser.add_argument('--aug_scale_max', type=float, default=1.02, help='Random affine maximum scale')
    parser.add_argument('--aug_affine', type=lambda x: bool(strtobool(x)), default=True, help='Enable RandomAffine geometric augmentation')
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
    parser.add_argument('--condition_perturb_min', type=float, default=0.01, help='Minimum relative perturbation magnitude (e.g. 0.01 = 1%)')
    parser.add_argument('--condition_perturb_max', type=float, default=0.03, help='Maximum relative perturbation magnitude (e.g. 0.03 = 3%)')
    parser.add_argument('--condition_perturb_on_hidden', type=lambda x: bool(strtobool(x)), default=True, help='Apply condition perturbation on hidden state')
    parser.add_argument('--condition_perturb_on_info', type=lambda x: bool(strtobool(x)), default=True, help='Apply condition perturbation on physics/info features')
    parser.add_argument('--condition_perturb_on_di', type=lambda x: bool(strtobool(x)), default=True, help='Apply condition perturbation on DI condition')
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
    parser.add_argument('--cfm_ode_steps', type=int, default=20, help='ODE integration steps for CFM inference during validation')
    parser.add_argument('--cfm_solver', type=str, default="heun", choices=["heun", "euler"], help='ODE solver for CFM sampling')
    parser.add_argument('--use_sd3_vae_latent', type=lambda x: bool(strtobool(x)), default=False, help='Train/sample CFM in SD3 VAE latent space instead of pixel space')
    parser.add_argument('--sd3_vae_model_id', type=str, default="stabilityai/stable-diffusion-3-medium-diffusers", help='Diffusers model id containing SD3 VAE weights')
    parser.add_argument('--sd3_vae_subfolder', type=str, default="vae", help='Subfolder for SD3 VAE weights in the model repository')
    parser.add_argument('--sd3_vae_use_fp16', type=lambda x: bool(strtobool(x)), default=True, help='Use fp16 for SD3 VAE on CUDA to reduce memory')
    parser.add_argument('--lr_schedule', type=str, default="constant", choices=["constant", "warmup_cosine"], help='Learning-rate schedule type')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs for warmup_cosine schedule')
    parser.add_argument('--min_lr_ratio', type=float, default=0.1, help='Final LR ratio for warmup_cosine schedule')
    parser.add_argument('--val_every_n_epoch', type=int, default=10, help='Run validation every N epochs')
    parser.add_argument('--inference_every_n_epochs', type=int, default=10, help='Save/log multi-sample CFM inference images every N epochs')
    parser.add_argument('--inference_num_images', type=int, default=6, help='Number of inference samples to visualize per logging epoch')
    parser.add_argument('--metrics_every_n_epochs', type=int, default=100, help='Compute/log FID/PSNR/SSIM every N validation epochs')
    parser.add_argument('--checkpoint_every_n_epochs', type=int, default=0, help='Save checkpoint_epoch_X.pt every N epochs (set 0 to disable)')
    parser.add_argument('--early_stop_patience', type=int, default=0, help='Early stopping patience on val/loss (set 0 to disable)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0, help='Minimum val/loss improvement to reset early stopping')
    parser.add_argument('--fixed_di_values', type=str, default="0.0,0.4,0.75,0.95", help='Comma-separated DI values used for fixed CFM inference panel')
    
    return parser.parse_args()


"""checking arguments"""
def check_args(args, current_time):
    result_dir = ["./results", "DamagePatternGeneration", args.loss_type]
    args.result_dir = os.path.join(*result_dir, current_time)
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
    image, hidden_state, direction, exp, timestep, DI, info = batch[:7]
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

    