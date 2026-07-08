import torch, os, json, glob, importlib, argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.dataset import HystereticDataset
from tqdm import tqdm
from torch.utils.data import DataLoader

import time

import hysteresis as hys

def draw_results(drift, simulated_force, pred_force, gt_force, result_dir, exp, mode):
    # Dark mode
    plt.style.use("dark_background")
    fig = plt.figure()
    if simulated_force is not None:
        plt.plot(drift, simulated_force, label="Simulated", linestyle="dotted", color="tab:gray", alpha=0.7)
    plt.plot(drift, gt_force, label="Experiment", color="tab:blue", alpha=0.7)
    plt.plot(drift, pred_force, label="Predicted", linestyle="--", color="tab:red", alpha=0.7)
    
    plt.xlabel("Drift (%)")
    plt.ylabel("Force (kN)")
    plt.legend()
    plt.title(f"Specimen {exp}")
    plt.savefig(os.path.join(result_dir, "prediction(black)", mode, f"{exp}.svg"), transparent=True)
    plt.close()


if __name__ == "__main__":
    # Update this path to your trained run directory before testing.
    result_path = "./results/HystereticPrediction/HystereticGRU/<run_id>"
    mode = "val"
    args = json.load(open(glob.glob(os.path.join(result_path, "**/args.json"), recursive=True)[0], "r"))
    args = argparse.Namespace(**args)
    
    if not os.path.exists(os.path.join(result_path, "prediction(black)", mode)):
        os.makedirs(os.path.join(result_path, "prediction(black)", mode))
    
    # if hasattr(args, "fold"):
    #     train_id = []
    #     val_id = []
    #     fold_list = json.load(open("./Data/Hysteresis/fold.json"))
    #     for i in range(1, 6):
    #         if i == args.fold:
    #             val_id.extend(fold_list[f"fold{i}"])
    #         else:
    #             train_id.extend(fold_list[f"fold{i}"])
    #     print(f"Fold {args.fold}")
    #     if mode == "train":
    #         args.specified = train_id
    #     else:
    #         args.specified = val_id
    
    args.specified = ["CTR1"]
    
    model = importlib.import_module(f"src.model.HystereticPrediction").__dict__[args.architecture](**vars(args))
    test_ds = HystereticDataset(root="./Data/Hysteresis", mode=mode, **vars(args))
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=os.cpu_count(), collate_fn=test_ds.collate_fn)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    state_dict = torch.load(glob.glob(os.path.join(result_path, "**/*best.ckpt"), recursive=True)[0])["state_dict"]
    status = model.load_state_dict(state_dict)
    print(status)
    
    
    history = {"mse_pred": [], "mse_sim": [], "rmse_pred": [], "rmse_sim": [], "nrmse_pred": [], "nrmse_sim": [], "peak_error": [], "peak_error_sim": []}
    time_count = 0
    with torch.no_grad():
        for j, batch in enumerate(tqdm(test_dataloader)):
            simulated_loop, test_force, original_length, maximun_simulated_force, exp = batch
            simulated_loop = simulated_loop.to(device)
            current_time = time.time()
            predicted_force = model.forward(simulated_loop)
            if j == 0:
                free, total = torch.cuda.mem_get_info(device)
                mem_used_MB = (total - free) / 1024 ** 2
                print(f"Memory used: {mem_used_MB} MB")
            time_count += time.time() - current_time
            if args.simulate:
                drift = simulated_loop[:, :, -2].cpu().squeeze().numpy()
                simulated_force = (simulated_loop[:, :, -1].cpu() * maximun_simulated_force).squeeze().numpy()
            else:
                drift = simulated_loop[:, :, -1].cpu().squeeze().numpy()
                simulated_force = None
            predicted_force = (predicted_force.cpu() * maximun_simulated_force).squeeze().numpy()
            test_force = (test_force.cpu() * maximun_simulated_force).squeeze().numpy()
            mse_pred = np.mean((test_force - predicted_force) ** 2)
            rmse_pred = np.sqrt(mse_pred)
            nrmse_pred = rmse_pred / (np.max(test_force) - np.min(test_force))
            
            # Find peak in drift
            hysLoop_test = hys.Hysteresis(np.column_stack([drift, test_force]))
            reversalIndex = hysLoop_test.reversalIndexes
            
            if args.simulate:
                mse_sim = np.mean((test_force - simulated_force) ** 2)
                rmse_sim = np.sqrt(mse_sim)
                nrmse_sim = rmse_sim / (np.max(test_force) - np.min(test_force))
                hysLoop_sim = hys.Hysteresis(np.column_stack([drift, simulated_force]))
                
                mse_peak_sim = np.mean((test_force[reversalIndex] - simulated_force[reversalIndex]) ** 2)
                rmse_peak_sim = np.sqrt(mse_peak_sim)
                history["peak_error_sim"].append(rmse_peak_sim)
                history["mse_sim"].append(mse_sim)
                history["rmse_sim"].append(rmse_sim)
                history["nrmse_sim"].append(nrmse_sim)
                
            
            history["mse_pred"].append(mse_pred)
            history["rmse_pred"].append(rmse_pred)
            history["nrmse_pred"].append(nrmse_pred)
            
            
            mse_peak = np.mean((test_force[reversalIndex] - predicted_force[reversalIndex]) ** 2)
            rmse_peak = np.sqrt(mse_peak)
            history["peak_error"].append(rmse_peak)
            
            os.makedirs(os.path.join(result_path, "csv", mode), exist_ok=True)
            df = pd.DataFrame({"Drift": drift, "Simulated": simulated_force, "Predicted": predicted_force, "Test": test_force})
            df.to_csv(os.path.join(result_path, "csv", mode, f"{exp[0]}.csv"), index=False)

            draw_results(drift, simulated_force, predicted_force, test_force, result_path, exp[0], mode)
    if args.simulate:
        print(f"MSE Pred: {np.mean(history['mse_pred']):.4f} - MSE Sim: {np.mean(history['mse_sim']):.4f} - RMSE Pred: {np.mean(history['rmse_pred']):.4f} - RMSE Sim: {np.mean(history['rmse_sim']):.4f} - NRMSE Pred: {np.mean(history['nrmse_pred']):.4f} - NRMSE Sim: {np.mean(history['nrmse_sim']):.4f} - Peak Error: {np.mean(history['peak_error']):.4f} - Peak Error Sim: {np.mean(history['peak_error_sim']):.4f}")
        
    else:
        print(f"MSE Pred: {np.mean(history['mse_pred']):.4f} - RMSE Pred: {np.mean(history['rmse_pred']):.4f} - NRMSE Pred: {np.mean(history['nrmse_pred']):.4f} - Peak Error: {np.mean(history['peak_error']):.4f}")
    
    print("Total time: ", time_count)
    
    # Take average for each
    new_history = {}
    for key in history.keys():
        if len(history[key]) == 0:
            continue
        new_history[key] = float(np.mean(history[key]))
        
    # Save to json
    with open(os.path.join(result_path, f"metrics_{mode}.json"), "w") as f:
        json.dump(new_history, f, indent=4)