import torch, os, json, glob, importlib, argparse, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.dataset import PatternLoopDataset
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.transforms import v2

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from sklearn.metrics import r2_score

from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Baseline
    result_path_list = [
        # "./results/DamagePatternGeneration/RSGAN/2024-06-25_21-06-46"
        # "./results/DamagePatternGeneration/RSGAN/2024-06-24_19-04-20",
        # "./results/DamagePatternGeneration/RSGAN/2024-06-24_17-52-10"
        
        # "./results/DamagePatternGeneration/Baseline/2024-06-15_20-03-23", 
        # "./results/DamagePatternGeneration/Baseline/2024-06-15_20-07-08",
        # "./results/DamagePatternGeneration/Baseline/2024-06-17_12-14-26",
        # "./results/DamagePatternGeneration/Baseline/2024-06-17_12-17-37",
        # "./results/DamagePatternGeneration/SGAN/2024-06-15_19-36-55",
        # "./results/DamagePatternGeneration/SGAN/2024-06-15_01-25-09",
        # "./results/DamagePatternGeneration/SGAN/2024-06-15_01-23-07",
        # "./results/DamagePatternGeneration/SGAN/2024-06-15_01-21-08",
        # "./results/DamagePatternGeneration/RSGAN/2024-06-15_19-36-55",
        # "./results/DamagePatternGeneration/RSGAN/2024-06-15_01-25-10",
        # "./results/DamagePatternGeneration/RSGAN/2024-06-15_01-23-09",
        # "./results/DamagePatternGeneration/RSGAN/2024-06-15_01-21-11",
        # "./results/DamagePatternGeneration/RaSGAN/2024-06-15_19-36-55",
        # "./results/DamagePatternGeneration/RaSGAN/2024-06-15_01-25-13",
        # "./results/DamagePatternGeneration/RaSGAN/2024-06-15_01-23-11",
        "./results/DamagePatternGeneration/RaSGAN/2024-06-15_01-21-13"
    ]

    mode = "test"

    for result_path in result_path_list:
        args = json.load(open(glob.glob(os.path.join(result_path, "**/args.json"), recursive=True)[0], "r"))
        args = argparse.Namespace(**args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.latent_conditioning:
            l = "Latent"
        else:
            l = "DI"
            
        save_path = os.path.join("./evaluation", f"{args.loss_type}_{args.image_size}_{l}")
        # save_path = result_path
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        if args.loss_type == "Baseline":
            n = 1
        else:
            n = 1
            
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
        status = extractor.load_state_dict(torch.load(glob.glob(os.path.join(extractor_path, "**/best.ckpt"), recursive=True)[0])["state_dict"])
        print("Extractor Loading Status: ", status)
        
        # val_ds = PatternLoopDataset(root="./Data", mode="val", transform=transform, extractor=extractor, extractor_args=extractor_args, **vars(args))
        # train_ds = PatternLoopDataset(root="./Data", mode="train", transform=transform, extractor=extractor, extractor_args=extractor_args, **vars(args))
        # concat_ds = ConcatDataset([train_ds, val_ds])
        # test_dataloader = DataLoader(concat_ds, batch_size=8, shuffle=False, num_workers=os.cpu_count())
        
        test_ds = PatternLoopDataset(root="./Data", mode=mode, transform=transform, extractor=extractor, extractor_args=extractor_args, **vars(args))
        test_dataloader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=os.cpu_count())
        
        # Load regressor
        regressor_path = "./results/DamageQuantification/vit_b_16/2024-06-12_14-01-03"
        regressor_args = json.load(open(glob.glob(os.path.join(regressor_path, "**/args.json"), recursive=True)[0], "r"))
        regressor_args = argparse.Namespace(**regressor_args)
        regressor = importlib.import_module(f"src.model.DamageQuantification").__dict__["DamageQuantification"](**vars(regressor_args))
        status = regressor.load_state_dict(torch.load(glob.glob(os.path.join(regressor_path, "**/best_*.ckpt"), recursive=True)[0])["state_dict"])
        regressor = regressor.to(device)
        print("Regressor Loading Status: ", status)
        
        # Load model
        model = importlib.import_module(f"src.model.ImageGeneration").__dict__[args.architecture](**vars(args))    
        status = model.load_state_dict(torch.load(glob.glob(os.path.join(result_path, "**/*02999.ckpt"), recursive=True)[0])["state_dict"])
        model = model.to(device)
        print("Generator Loading Status: ", status)
        
        custom_transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # history = {"psnr": [], "ssim": [], "fid": [], "custom_fid": [], "r2": []}
        history = {"psnr": [], "ssim": [], "fid": [], "r2": []}
        label = pd.read_csv("./Data/DamagePattern/label.csv")
        for i in range(n):       
            fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True).to(device)
            # custom_fid = FrechetInceptionDistance(feature=regressor.backbone.to("cpu"), reset_real_features=False, 
            #                                     input_img_size=(3, 224, 224), normalize=True).to(device)
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            psnr = PeakSignalNoiseRatio().to(device)
            
            gt_list = []
            pred_list = []
            # Get the fake images
            with torch.no_grad():
                for j, batch in enumerate(tqdm(test_dataloader)):
                    real_data, hidden_state, direction, exp, timestep, DI, info= batch
                    bs_ = real_data.size(0)
                    if args.loss_type == "Baseline":
                        noise = torch.randn(bs_, 0, 1, 1).to(device)
                    else:
                        noise = torch.randn(bs_, args.z_dim, 1, 1).to(device)
                    real_data = real_data.to(device)
                    condition = model.process_condition(info, hidden_state, direction, DI)
                    condition = condition.view(-1, condition.size(1), 1, 1).to(device)
                    fake_image = model.G(noise, condition)
                    
                        
                    fake_image = (fake_image + 1) / 2.
                    real_data = (real_data + 1) / 2.
                    
                    psnr.update(fake_image, real_data)
                    ssim.update(fake_image, real_data)
                    fid.update(fake_image, real=False)
                    fid.update(real_data, real=True)
                    # custom_fid.update(custom_transform(fake_image), real=False)
                    # custom_fid.update(custom_transform(real_data), real=True)
                    
                    # Get the prediction
                    info = info.to(device)
                    pred_DI = regressor.forward(custom_transform(fake_image), info)
                    pred_list.append(pred_DI)
                    gt_list.append(DI)
                    
                    fake_image = fake_image.squeeze().cpu().numpy()
                    
                    for bs in range(bs_):
                        temp_ = np.transpose(fake_image[bs, :, :, :], (1, 2, 0))
                        temp_ = (temp_ * 255).astype(np.uint8)
                        temp_ = Image.fromarray(temp_)
                        ts_ = int(timestep[bs].item())
                        exp_ = exp[bs]
                        direction_ = direction[bs, :]
                        decoded = torch.argmax(direction_)
                        if decoded == 0:
                            dir_ = "Push"
                        else:
                            dir_ = "Pull"
                        
                        if not os.path.exists(os.path.join(save_path, exp_)):
                            os.makedirs(os.path.join(save_path, exp_))

                        try:
                            filename = label[(label["exp"] == exp_) & (label["timestep"] == ts_) & (label["direction"] == dir_)]["filename"].values[0]
                            temp_.save(os.path.join(save_path, exp_, filename.replace(".jpg", f"_{mode}.jpg")))
                        except:
                            temp_.save(os.path.join(save_path, exp_, f"{exp_}_t{ts_}_{dir_}.jpg"))
            
            fid_score = fid.compute().cpu().numpy()
            # custom_fid_score = custom_fid.compute().cpu().numpy()
            ssim_score = ssim.compute().cpu().numpy()
            psnr_score = psnr.compute().cpu().numpy()
            history["psnr"].append(psnr_score)
            history["ssim"].append(ssim_score)
            history["fid"].append(fid_score)
            # history["custom_fid"].append(custom_fid_score)
            
            
            gt_list = torch.cat(gt_list).cpu().numpy()
            pred_list = torch.cat(pred_list).cpu().numpy()
            r2 = r2_score(gt_list, pred_list)
            history["r2"].append(r2)
            print("======================================================================")
            print("")
            print(f"Image size: {args.image_size}; Loss Type: {args.loss_type}; Latent Conditioning: {args.latent_conditioning}")
            print("PSNR Score: ", psnr_score)
            print("SSIM Score: ", ssim_score)
            print("FID Score: ", fid_score)
            # print("Custom FID Score: ", custom_fid_score)
            print("R2 Score: ", r2)
            print("")
            
            
            if not os.path.exists(os.path.join(save_path, "scatter")):
                os.makedirs(os.path.join(save_path, "scatter"))
            
            fig, ax = plt.subplots()
            ax.axline((0, 0), slope=1, color="black")
            ax.scatter(gt_list, pred_list, s=5)
            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Prediction")
            ax.set_xlim(0, 1.4)
            ax.set_ylim(0, 1.4)
            ax.set_title(r"$R^2$ Score: {:.4f}".format(r2))
            plt.savefig(os.path.join(save_path, "scatter", f"{mode}_scatter_{i:02d}.png"))
            plt.close("all")
            
            fid.reset()
            # custom_fid.reset()
            ssim.reset()
            psnr.reset()
            
        pd.DataFrame(history).to_csv(os.path.join(save_path, f"history_{mode}.csv"), index=False)