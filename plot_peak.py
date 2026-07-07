import os, glob
import pandas as pd
import imageio.v2 as imageio
import torch, os, json, glob, importlib, argparse
import numpy as np
from PIL import Image
from src.dataset import HystereticDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from tqdm import tqdm
import hysteresis as hys

def write2mp4(result_path, exp):
    png_images = sorted(glob.glob(os.path.join("./temp", '*.png')))
    images = [imageio.imread(png) for png in png_images]
    if not os.path.exists(f"{result_path}/video/"):
        os.makedirs(f"{result_path}/video/")
    imageio.mimsave(f'{result_path}/video/{exp}_peak.mp4', images, fps=2)

def get_real_image(real_dir, mode, direction, timestep, label, fake_img_size):
    if mode != "inter":
            temp_df = label[(label["direction"] == direction) & (label["timestep"] == timestep)]
            real_filename = temp_df["filename"].values[0]
            real_img = Image.open(os.path.join(real_dir, real_filename))
            # Resize to the same shape with fake img
            real_img = real_img.resize((fake_img_size[0], fake_img_size[1]))
            real_img = np.array(real_img)
            
    else:
        # Set real image to gray images
        real_img = np.zeros((fake_img_size[0], fake_img_size[1], 3))
        
    return real_img

def plot(loop, exp):
    real_dir = os.path.join("./Data/DamagePattern/images")
    
    hysLoop = hys.Hysteresis(np.column_stack([loop["drift"], loop["test"]]))
    reversalIndexes = hysLoop.reversalIndexes
    
    for i, timestep in enumerate(tqdm(reversalIndexes)):
        max_force = loop["max_simulated_force"]
        
        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
        gs = gridspec.GridSpec(2, 4, figure=fig)
        ax0 = plt.subplot(gs[:, :2])
        ax1 = plt.subplot(gs[0, -2])
        ax2 = plt.subplot(gs[1, -2])
        ax3 = plt.subplot(gs[0, -1])
        ax4 = plt.subplot(gs[1, -1])

        line2 = ax0.plot(loop['drift'], loop['test']*max_force, linestyle="-", color="tab:blue", label="Experiment", alpha=0.5)
        line3 = ax0.plot(loop['drift'], loop['predicted']*max_force, linestyle="--", color="tab:red", label="Prediction", alpha=0.7)
        ax0.plot(loop['drift'][timestep], loop['predicted'][timestep]*max_force, marker="*", markeredgecolor="k", markerfacecolor="tab:red", label="Current Prediction", markersize=10)
        line4 =  Line2D([0], [0], marker='*', label='Current Point', linestyle='None', markeredgecolor='k', markerfacecolor="k", markersize=10)
        ax0.legend(handles=[line2[0], line3[0], line4])
        ax0.set_title("Hysteresis Loop")
        ax0.set_xlabel("Drift (%)")
        ax0.set_ylabel("Force (kN)")
        
        ax1.imshow(real_push_img)
        ax1.set_title("Real Image (+)")
        ax1.axis("off")
        ax2.imshow(fake_push_img)
        if push_mode != "inter":
            ax2.set_title(f"Generated Image ({push_mode})")
        else:
            ax2.set_title("Generated Image")
        ax2.axis("off")
        
        ax3.imshow(real_pull_img)
        ax3.set_title("Real Image (-)")
        ax3.axis("off")
        ax4.imshow(fake_pull_img)
        if pull_mode != "inter":
            ax4.set_title(f"Generated Image ({pull_mode})")
        else:
            ax4.set_title("Generated Image")
        ax4.axis("off")
        
        
        fig.suptitle(f"Experiment {exp} - Timestep {timestep}", fontsize=16)
        plt.savefig(f"./temp/{timestep:05d}.png")
        plt.close("all")    
        
        
    
    


if __name__ == "__main__":
    # # Clean the temp folder
    temp_dir = "./temp"
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    
    exp = "R615"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load extractor
    extractor_path = "./results/HystereticPrediction/HystereticGRU/2024-06-12_18-13-47"
    extractor_args = json.load(open(glob.glob(os.path.join(extractor_path, "**/args.json"), recursive=True)[0], "r"))
    extractor_args = argparse.Namespace(**extractor_args)
    extractor = importlib.import_module(f"src.model.HystereticPrediction").__dict__[extractor_args.architecture](**vars(extractor_args))
    extractor = extractor.to(device)
    state_dict = torch.load(glob.glob(os.path.join(extractor_path, "**/best.ckpt"), recursive=True)[0])["state_dict"]
    status = extractor.load_state_dict(state_dict)
    print(status)
    

    # Load hys
    test_ds = HystereticDataset(root="./Data/Hysteresis", specified=[exp], **vars(extractor_args))
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=os.cpu_count(), collate_fn=test_ds.collate_fn)
    
    # Predict
    with torch.no_grad():
        for j, batch in enumerate(tqdm(test_dataloader)):
            loop, test_force, original_length, maximun_simulated_force, exp = batch
            loop = loop.to(device)
            predicted_force = extractor.forward(loop)
            loop = loop.cpu().squeeze().numpy()
            predicted_force = predicted_force.cpu().squeeze().numpy()
            test_force = test_force.cpu().squeeze().numpy()
            loop = {"drift": loop[:, -2], "test": test_force, "predicted": predicted_force, "max_simulated_force": maximun_simulated_force, "exp": exp}

    # Plot the result and save to temp
    plot(loop, exp)
    
    write2mp4(result_path, exp)
    print("Done")