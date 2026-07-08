import argparse, os, json
from distutils.util import strtobool


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of Hysteretic Loop Based Prediction and Generation"
    parser = argparse.ArgumentParser(description=desc)  
    parser.add_argument('--lr', type=float, default=4.5e-06, help='The learning rate')
    parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--use_wandb', action="store_true", help='Enable Weights & Biases logging')
    parser.add_argument('--seed', type=int, default=8465468, help='Random seed')
    parser.add_argument('--embed_dim', type=int, default=4, help='The dimension of embedding')
    parser.add_argument('--n_embed', type=int, default=256, help='The number of embedding')
    parser.add_argument('--disc_start', type=int, default=0, help='The start of discriminator')
    parser.add_argument('--resolution', type=int, default=256, help='The resolution of image')
    parser.add_argument('--architecture', type=str, default="VQGAN", help='The architecture of model')
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
    parser.add_argument('--aug_blur_kernel', type=int, default=3, help='Gaussian blur kernel size (odd)')
    parser.add_argument('--aug_blur_sigma_min', type=float, default=0.1, help='Gaussian blur minimum sigma')
    parser.add_argument('--aug_blur_sigma_max', type=float, default=0.5, help='Gaussian blur maximum sigma')
    parser.add_argument('--aug_erase_prob', type=float, default=0.1, help='Probability for random erasing (cutout-like augmentation)')
    parser.add_argument('--aug_erase_scale_min', type=float, default=0.01, help='Random erasing minimum erased area ratio')
    parser.add_argument('--aug_erase_scale_max', type=float, default=0.08, help='Random erasing maximum erased area ratio')
    parser.add_argument('--aug_erase_ratio_min', type=float, default=0.3, help='Random erasing minimum aspect ratio')
    parser.add_argument('--aug_erase_ratio_max', type=float, default=3.3, help='Random erasing maximum aspect ratio')
    parser.add_argument('--aug_erase_random_value', action="store_true", help='Use random fill value for random erasing')
    

    
    return parser.parse_args()


"""checking arguments"""
def check_args(args, current_time):
    result_dir = ["./results", "ImageReconstruction", args.architecture]
    args.result_dir = os.path.join(*result_dir, current_time)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    return args

def save_args(args):
    dict_args = {}
    dict_args = namespace_to_dict(args, dict_args)
    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(dict_args, f, indent=2)
        
    
def namespace_to_dict(namespace, dict):
    # iterate over the namespace and return a dictionary
    for key, value in namespace.__dict__.items():
        if isinstance(value, argparse.Namespace):
            dict[key] = {}
            dict[key] = namespace_to_dict(value, dict[key])
        else:
            dict[key] = value
    return dict
    
    
    