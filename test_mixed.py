import torch, os, json, glob, importlib, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.dataset import InteriorLoopDataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == "__main__":
    result_path = "./results/DamagePatternGeneration/RSGAN/2024-06-15_01-21-11"
    exp = "CTR1"
    save_path = "generation"
    
    label = pd.read_csv("./Data/DamagePattern/label.csv")
    
    args = json.load(open(glob.glob(os.path.join(result_path, "**/args.json"), recursive=True)[0], "r"))
    args = argparse.Namespace(**args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
        
    # Load extractor
    extractor_path = args.extractor_path
    extractor_args = json.load(open(glob.glob(os.path.join(extractor_path, "**/args.json"), recursive=True)[0], "r"))
    extractor_args = argparse.Namespace(**extractor_args)
    extractor = importlib.import_module(f"src.model.HystereticPrediction").__dict__[extractor_args.architecture](**vars(extractor_args))
    extractor = extractor.to(device)
    state_dict = torch.load(glob.glob(os.path.join(extractor_path, "**/best.ckpt"), recursive=True)[0])["state_dict"]
    status = extractor.load_state_dict(state_dict)
    print(status)
    
    # Dataset and DataLoader
    test_ds = InteriorLoopDataset(root="./Data", exp=exp, extractor_args=extractor_args, extractor=extractor, **vars(args))
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)
    
    
    # Load model
    model = importlib.import_module(f"src.model.ImageGeneration").__dict__[args.architecture](**vars(args))    
    state_dict = torch.load(glob.glob(os.path.join(result_path, "**/*02999.ckpt"), recursive=True)[0])["state_dict"]
    status = model.load_state_dict(state_dict)
    model = model.to(device)
    print(status)
    
    
    noise = torch.randn(1, args.z_dim, 1, 1).to(device)
    
    # Get the fake images
    with torch.no_grad():
        for j, batch in enumerate(tqdm(test_dataloader)):
            hidden_state, direction, timestep, mode, info= batch
            mode = mode[0]
            decoded = torch.argmax(direction)
            if decoded == 0:
                dir_ = "Push"
            else:
                dir_ = "Pull"
            ts_ = int(timestep[0].item())

            if args.latent_conditioning:
                DI = torch.zeros(hidden_state.size(0), 1)
            else:
                try:
                    di = label[(label["exp"] == exp) & (label["timestep"] == ts_) & (label["direction"] == dir_)]["DI"].values[0]
                    DI = torch.ones(hidden_state.size(0), 1) * di
                except:
                    print(f"DI not found for {exp} at timestep {ts_} and direction {dir_}")
                    continue
            condition = model.process_condition(info, hidden_state, direction, DI)
            condition = condition.view(-1, condition.size(1), 1, 1).to(device)
            fake_image = model.G(noise, condition)
            
            fake_image = fake_image.cpu().squeeze().numpy()
            fake_image = np.transpose(fake_image, (1, 2, 0))
            fake_image = (fake_image + 1) / 2
            fake_image = (fake_image * 255).astype(np.uint8)
            fake_image = Image.fromarray(fake_image)
            condition_type = "latent" if args.latent_conditioning else "DI"
            if not os.path.exists(os.path.join(save_path, f"{args.loss_type}_{args.image_size}_{condition_type}", exp)):
                os.makedirs(os.path.join(save_path, f"{args.loss_type}_{args.image_size}_{condition_type}", exp))
            try:
                filename = label[(label["exp"] == exp) & (label["timestep"] == ts_) & (label["direction"] == dir_)]["filename"].values[0]
                fake_image.save(os.path.join(save_path, f"{args.loss_type}_{args.image_size}_{condition_type}", exp, filename.replace(".jpg", f"_{mode}.jpg")))
            except:
                fake_image.save(os.path.join(save_path, f"{args.loss_type}_{args.image_size}_{condition_type}", exp, f"{exp}_t{ts_}_{dir_}.jpg"))
