import argparse, os, json

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of Hysteretic Loop Based Prediction and Generation"
    parser = argparse.ArgumentParser(description=desc)  
    parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch')
    
    parser.add_argument('--backbone', type=str, default="resnet101", help='The backbone of model')
    parser.add_argument('--pretrained', action="store_true", help='whether to use pretrained model')

    return parser.parse_args()


"""checking arguments"""
def check_args(args, current_time):
    result_dir = ["./results", "DamageQuantification", args.backbone]
    args.result_dir = os.path.join(*result_dir, current_time)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    return args

def save_args(args):
    dict_args = vars(args)
    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(dict_args, f, indent=2)
        
    
def set_dataset_args(args, train_dataloader):
    image, features, exp, DI = next(iter(train_dataloader))
    args.features_dim = features.shape[-1]
