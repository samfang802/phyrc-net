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
    
    
    