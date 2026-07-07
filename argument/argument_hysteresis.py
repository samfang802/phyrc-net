import argparse, os, json

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of Hysteretic Loop Based Prediction and Generation"
    parser = argparse.ArgumentParser(description=desc)  
    parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch')
    
    parser.add_argument('--data_size', type=str, default="small", help='The size of data')
    parser.add_argument('--architecture', type=str, default="HystereticLSTM", help='The architecture of model')
    parser.add_argument('--hidden_state_dim', type=int, default=512, help='The dimension of hidden state')
    parser.add_argument('--layers', type=int, default=2, help='The number of lstm layers')
    parser.add_argument('--MLP_activation', type=str, default="relu", help='The activation function of MLP')
    parser.add_argument('--maximum', type=float, default=1.0, help='The unit of maximum value. (kN)')
    parser.add_argument('--bidirectional', action="store_true", help='The direction of lstm')
    parser.add_argument('--simulate', action="store_true", help='If add the simulated force or not')
    parser.add_argument('--scale', action="store_true", help='If scale the data or not')

    return parser.parse_args()


"""checking arguments"""
def check_args(args, current_time):
    result_dir = ["./results", "HystereticPrediction", args.architecture]
    args.result_dir = os.path.join(*result_dir, current_time)
    if not os.path.exists(os.path.join(args.result_dir, "HystereticLoop")):
        os.makedirs(os.path.join(args.result_dir, "HystereticLoop", "train"))
        os.makedirs(os.path.join(args.result_dir, "HystereticLoop", "val"))
    return args

def save_args(args):
    dict_args = vars(args)
    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(dict_args, f, indent=2)
        
    
def set_dataset_args(args, train_dataloader):
    simulated_loop, test_force, length, max_simulate_force, id_list = next(iter(train_dataloader))
    args.signal_feat = simulated_loop.shape[-1]
