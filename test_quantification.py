import torch, os, json, glob, importlib, argparse
import numpy as np
import matplotlib.pyplot as plt


from src.dataset import DamageIndexDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from PIL import Image
from sklearn.metrics import r2_score

def draw_attn_map(attn_list, image, result_dir, mode, name, prediction, gt):
    patch_size = int(np.sqrt(attn_list[0].shape[0]))
    row = 3
    col = len(attn_list) // row
    fig, ax = plt.subplots(row, col)
    alpha = 0.3
    for i, attn in enumerate(attn_list):
        # Making the attention map to grid and resize
        attn = attn.reshape(patch_size, patch_size)
        attn = np.array(Image.fromarray(attn).resize((224, 224)))
        # Normalize the attention map
        attn = (attn - attn.min()) / (attn.max() - attn.min())
        # Expand the attention map to 3 channels
        attn = plt.cm.jet(attn)[:, :, :3]
        img_feature = ((attn * 255) * alpha + image * 255 * (1 - alpha)).astype(np.uint8)
        ax[i//col][i%col].imshow(img_feature)
        ax[i//col][i%col].axis("off")
        
    plt.suptitle(f"Prediction: {prediction[0, 0]:.4f}, GT: {gt[0, 0]:.4f}")
    plt.savefig(os.path.join(result_dir, "prediction", mode, f"{exp}_DI_{gt[0, 0]:.4f}_attn.svg"), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    result_path = "./results/DamageQuantification/resnet101/2024-10-28_10-05-52"
    mode = "train"
    args = json.load(open(glob.glob(os.path.join(result_path, "**/args.json"), recursive=True)[0], "r"))
    args = argparse.Namespace(**args)
    
    if not os.path.exists(os.path.join(result_path, "prediction", mode)):
        os.makedirs(os.path.join(result_path, "prediction", mode))
    
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    invTrans = v2.Compose([ 
        v2.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        v2.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
    ])
    
    model = importlib.import_module(f"src.model.DamageQuantification").__dict__["DamageQuantification"](**vars(args))
    print(model)
    if "vit" in args.backbone:
        attn_list = []
        def hook(module, input, output):
            output, attn = module.forward(input[0], input[1], input[2], need_weights=True)
            attn_list.append(attn[0, 0, 1:].clone().detach().cpu().numpy())
            
        for i, layer in enumerate(model.backbone.encoder.layers.children()):
            layer.self_attention.register_forward_hook(hook)
            
    elif "swin" in args.backbone:
        pass
    
    test_ds = DamageIndexDataset(root="./Data", mode=mode, transform=transform, **vars(args))
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=os.cpu_count())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    state_dict = torch.load(glob.glob(os.path.join(result_path, "**/r2*.ckpt"), recursive=True)[0])["state_dict"]
    status = model.load_state_dict(state_dict)
    print(status)
    
    pred_list = []
    gt_list = []
    exp_list = []
    for i, sample in enumerate(tqdm(test_dataloader)):
        attn_list = []
        image, features, exp, DI = sample
        image = image.to(device)
        features = features.to(device)
        
        with torch.no_grad():
            prediction = model.forward(image, features)
        
        pred_list.append(prediction)
        gt_list.append(DI)
        
        prediction = prediction.cpu().numpy()
        gt = DI.cpu().numpy()
        image = invTrans(image).squeeze().cpu().permute(1, 2, 0).numpy()
        # Save image
        exp = exp[0]
        exp_list.append(exp)
        plt.imsave(os.path.join(result_path, "prediction", mode, f"{exp}_DI_{gt[0, 0]:.4f}.png"), image)
        # draw_attn_map(attn_list, image, result_path, mode, exp, prediction, gt)
        
    
    # Draw the scatter plot
    prediction = torch.cat(pred_list).cpu().numpy()
    gt = torch.cat(gt_list).cpu().numpy()
    exp = np.array(exp_list)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axline((0, 0), slope=1, color="black", alpha=0.5, linestyle="--")
    for g in np.unique(exp):
        i = np.where(exp == g)[0]
        ax.scatter(gt[i], prediction[i], label=g, s=5)
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1.4)
    ax.legend()
    r2 = r2_score(gt, prediction)

    if mode == "val":
        ax.set_title(f"Validation"+ r"($R^2$" + f"={r2:.4f})")
    elif mode == "test":
        ax.set_title(f"Test"+ r"($R^2$" + f"={r2:.4f})")
    elif mode == "train":
        ax.set_title(f"Train"+ r"($R^2$" + f"={r2:.4f})")    
    print(f"R2 Score: {r2}")
    plt.savefig(os.path.join(result_path, "prediction", mode, "scatter.svg"))