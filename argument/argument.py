import argparse, os, json

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of Hysteretic Loop Based Prediction and Generation"
    parser = argparse.ArgumentParser(description=desc)  
    parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch')
    
    task_subparser = parser.add_subparsers(dest='task', metavar="Command for Different Task", help='sub-command help')
    
    # Task: Hysteretic Prediction
    hysteresis_parser = task_subparser.add_parser('HystereticPrediction', help='Hysteretic prediction command')
    hysteresis_arch_parser = hysteresis_parser.add_subparsers(dest='architecture', metavar="Command for Different Model Architecture", help='sub-command help')
    
    LSTM_parser = hysteresis_arch_parser.add_parser('HystereticLSTM', help='LSTM command')
    LSTM_parser.add_argument('--signal_length', type=int, default=512, help='The length of signal')
    LSTM_parser.add_argument('--hidden_state_dim', type=int, default=512, help='The dimension of hidden state')
    LSTM_parser.add_argument('--lstm_layers', type=int, default=2, help='The number of lstm layers')
    LSTM_parser.add_argument('--bidirectional', action="store_true", help='The direction of lstm')
    LSTM_parser.add_argument('--encoding_type', type=str, default="one-hot", help='The encoding type of discrete label')
    

    
    
    # Task: Image Generation
    generation_parser = task_subparser.add_parser('ImageGeneration', help='Image generation command')
    generation_arch_parser = generation_parser.add_subparsers(dest='architecture', metavar="Command for Different Model Architecture", help='sub-command help')
    
    CGAN_parser = generation_arch_parser.add_parser('CGAN', help='CGAN command')
    CGAN_parser.add_argument('--csv_file', type=str, default="train.csv", help='The path of csv file that contains the label of input image')
    CGAN_parser.add_argument('--input_image_size', type=int, default=128, help='The size of input image')
    CGAN_parser.add_argument('--z_dim', type=int, default=128, help='The size of latent dimension')
    CGAN_parser.add_argument('--discrete_column', nargs="*", type=str, default=["AR", "TR", "LR", "DI"], help='The discrete label of input')
    CGAN_parser.add_argument('--continuous_column', nargs="*", type=str, default=[], help='The continuous label of input')
    CGAN_parser.add_argument('--encoding_type', type=str, default="one-hot", help='The encoding type of discrete label')
    
    CGAN_plus_parser = generation_arch_parser.add_parser('CGAN_plus', help='CGAN_plus command')
    CGAN_plus_parser.add_argument('--csv_file', type=str, default="train.csv", help='The path of csv file that contains the label of input image')
    CGAN_plus_parser.add_argument('--input_image_size', type=int, default=128, help='The size of input image')
    CGAN_plus_parser.add_argument('--z_dim', type=int, default=128, help='The size of latent dimension')
    CGAN_plus_parser.add_argument('--discrete_column', nargs="*", type=str, default=["AR", "TR", "LR", "DI"], help='The discrete label of input')
    CGAN_plus_parser.add_argument('--continuous_column', nargs="*", type=str, default=[], help='The continuous label of input')
    CGAN_plus_parser.add_argument('--encoding_type', type=str, default="one-hot", help='The encoding type of discrete label')
    
    WCGAN_GP_parser = generation_arch_parser.add_parser('WCGAN_GP', help='WCGAN-GP command')
    WCGAN_GP_parser.add_argument('--csv_file', type=str, default="train.csv", help='The path of csv file that contains the label of input image')
    WCGAN_GP_parser.add_argument('--input_image_size', type=int, default=128, help='The size of input image')
    WCGAN_GP_parser.add_argument('--z_dim', type=int, default=128, help='The size of latent dimension')
    WCGAN_GP_parser.add_argument('--discrete_column', nargs="*", type=str, default=["AR", "TR", "LR", "DI"], help='The discrete label of input')
    WCGAN_GP_parser.add_argument('--continuous_column', nargs="*", type=str, default=[], help='The continuous label of input')
    WCGAN_GP_parser.add_argument('--gp_weight', type=float, default=10, help='The weight of gradient penalty')
    WCGAN_GP_parser.add_argument('--encoding_type', type=str, default="one-hot", help='The encoding type of discrete label')


    
    # Task: Image Conditional Generation 
    mixed_parser = task_subparser.add_parser('Mixed', help='Mixed task command')
    mixed_arch_parser = mixed_parser.add_subparsers(dest='architecture', metavar="Command for Different Model Architecture", help='sub-command help')
    
    HystereticCGAN_parser = mixed_arch_parser.add_parser('HystereticCGAN', help='HystereticCGAN command')
    HystereticCGAN_parser.add_argument('--signal_length', type=int, default=512, help='The length of signal')
    HystereticCGAN_parser.add_argument('--hidden_state_dim', type=int, default=512, help='The dimension of hidden state')
    HystereticCGAN_parser.add_argument('--lstm_layers', type=int, default=2, help='The number of lstm layers')
    HystereticCGAN_parser.add_argument('--bidirectional', action="store_true", help='The direction of lstm')
    HystereticCGAN_parser.add_argument('--embedding_dim', type=int, default=128, help='The dimension of embedding')
    HystereticCGAN_parser.add_argument('--csv_file', type=str, default="train.csv", help='The path of csv file that contains the label of input image')
    HystereticCGAN_parser.add_argument('--input_image_size', type=int, default=128, help='The size of input image')
    HystereticCGAN_parser.add_argument('--z_dim', type=int, default=128, help='The size of latent dimension')
    HystereticCGAN_parser.add_argument('--discrete_column', nargs="*", type=str, default=["AR", "TR", "LR", "DI"], help='The discrete label of input')
    HystereticCGAN_parser.add_argument('--continuous_column', nargs="*", type=str, default=[], help='The continuous label of input')
    HystereticCGAN_parser.add_argument('--encoding_type', type=str, default="one-hot", help='The encoding type of discrete label')
    
    HystereticCGAN_plus_parser = mixed_arch_parser.add_parser('HystereticCGAN_plus', help='HystereticCGAN_plus command')
    HystereticCGAN_plus_parser.add_argument('--signal_length', type=int, default=512, help='The length of signal')
    HystereticCGAN_plus_parser.add_argument('--hidden_state_dim', type=int, default=512, help='The dimension of hidden state')
    HystereticCGAN_plus_parser.add_argument('--lstm_layers', type=int, default=2, help='The number of lstm layers')
    HystereticCGAN_plus_parser.add_argument('--bidirectional', action="store_true", help='The direction of lstm')
    HystereticCGAN_plus_parser.add_argument('--embedding_dim', type=int, default=128, help='The dimension of embedding')
    HystereticCGAN_plus_parser.add_argument('--csv_file', type=str, default="train.csv", help='The path of csv file that contains the label of input image')
    HystereticCGAN_plus_parser.add_argument('--input_image_size', type=int, default=128, help='The size of input image')
    HystereticCGAN_plus_parser.add_argument('--z_dim', type=int, default=128, help='The size of latent dimension')
    HystereticCGAN_plus_parser.add_argument('--discrete_column', nargs="*", type=str, default=["AR", "TR", "LR", "DI"], help='The discrete label of input')
    HystereticCGAN_plus_parser.add_argument('--continuous_column', nargs="*", type=str, default=[], help='The continuous label of input')
    HystereticCGAN_plus_parser.add_argument('--encoding_type', type=str, default="one-hot", help='The encoding type of discrete label')
    
    
    HystereticWCGAN_GP_parser = mixed_arch_parser.add_parser('HystereticWCGAN_GP', help='HystereticWCGAN_GP command')
    HystereticWCGAN_GP_parser.add_argument('--signal_length', type=int, default=512, help='The length of signal')
    HystereticWCGAN_GP_parser.add_argument('--hidden_state_dim', type=int, default=512, help='The dimension of hidden state')
    HystereticWCGAN_GP_parser.add_argument('--lstm_layers', type=int, default=2, help='The number of lstm layers')
    HystereticWCGAN_GP_parser.add_argument('--bidirectional', action="store_true", help='The direction of lstm')
    HystereticWCGAN_GP_parser.add_argument('--embedding_dim', type=int, default=128, help='The dimension of embedding')
    HystereticWCGAN_GP_parser.add_argument('--csv_file', type=str, default="train.csv", help='The path of csv file that contains the label of input image')
    HystereticWCGAN_GP_parser.add_argument('--input_image_size', type=int, default=128, help='The size of input image')
    HystereticWCGAN_GP_parser.add_argument('--z_dim', type=int, default=128, help='The size of latent dimension')
    HystereticWCGAN_GP_parser.add_argument('--discrete_column', nargs="*", type=str, default=["AR", "TR", "LR", "DI"], help='The discrete label of input')
    HystereticWCGAN_GP_parser.add_argument('--continuous_column', nargs="*", type=str, default=[], help='The continuous label of input')
    HystereticWCGAN_GP_parser.add_argument('--gp_weight', type=float, default=10, help='The weight of gradient penalty')
    HystereticWCGAN_GP_parser.add_argument('--encoding_type', type=str, default="one-hot", help='The encoding type of discrete label')
    
    
    HCGAN_parser = mixed_arch_parser.add_parser('HCGAN', help='HCGAN command')
    HCGAN_parser.add_argument('--signal_length', type=int, default=512, help='The length of signal')
    HCGAN_parser.add_argument('--hidden_state_dim', type=int, default=512, help='The dimension of hidden state')
    HCGAN_parser.add_argument('--lstm_layers', type=int, default=2, help='The number of lstm layers')
    HCGAN_parser.add_argument('--bidirectional', action="store_true", help='The direction of lstm')
    HCGAN_parser.add_argument('--embedding_dim', type=int, default=128, help='The dimension of embedding')
    HCGAN_parser.add_argument('--csv_file', type=str, default="train.csv", help='The path of csv file that contains the label of input image')
    HCGAN_parser.add_argument('--input_image_size', type=int, default=128, help='The size of input image')
    HCGAN_parser.add_argument('--z_dim', type=int, default=128, help='The size of latent dimension')
    HCGAN_parser.add_argument('--discrete_column', nargs="*", type=str, default=["AR", "TR", "LR", "DI"], help='The discrete label of input')
    HCGAN_parser.add_argument('--continuous_column', nargs="*", type=str, default=[], help='The continuous label of input')
    HCGAN_parser.add_argument('--encoding_type', type=str, default="one-hot", help='The encoding type of discrete label')
    

    return parser.parse_args()


"""checking arguments"""
def check_args(args, current_time):
    result_dir = ["./results", args.task, args.architecture]
    if args.data_type["damaged_image"]:
        set_condition_name(args)
        result_dir.append(args.condition_name)
    
    # --result_dir
    args.result_dir = os.path.join(*result_dir, current_time)
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if args.data_type["damaged_image"]:
        if not os.path.exists(os.path.join(args.result_dir, "Process")):
            os.makedirs(os.path.join(args.result_dir, "Process"))
    if args.data_type["hysteretic_loop"]:
        if not os.path.exists(os.path.join(args.result_dir, "HystereticLoop")):
            os.makedirs(os.path.join(args.result_dir, "HystereticLoop"))
    return args

def save_args(args):
    dict_args = vars(args)
    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(dict_args, f, indent=2)
        
def set_condition_name(args):
    discrete = ""
    for i, dc in enumerate(args.discrete_column):
        if i == 0:
            discrete += dc
        else:
            discrete += f"_{dc}"

    continuous = ""
    for i, cc in enumerate(args.continuous_column):
        if i == 0:
            continuous += cc
        else:
            continuous += f"_{cc}"
    
    args.condition_name = f"d{discrete}_c{continuous}"
    
def set_dataset_args(args, train_dataloader):
    if args.data_type["hysteretic_loop"] and not args.data_type["damaged_image"]:
        simulated_loop, test_loop, length = next(iter(train_dataloader))
        args.signal_feat = simulated_loop.shape[-1]
        args.padding_signal_length = simulated_loop.shape[1]
    elif args.data_type["damaged_image"] and not args.data_type["hysteretic_loop"]:
        img, dis_label, con_label = next(iter(train_dataloader))
        args.condition_dim = dis_label.shape[-1] + con_label.shape[-1]
    elif args.data_type["hysteretic_loop"] and args.data_type["damaged_image"]:
        img, dis_label, con_label, simulated_loop, test_loop, length = next(iter(train_dataloader))
        args.signal_feat = simulated_loop.shape[-1]
        args.condition_dim = dis_label.shape[-1] + con_label.shape[-1]
        args.padding_signal_length = simulated_loop.shape[1]