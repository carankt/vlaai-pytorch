import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import EegAudioDataset
from model import VLAAI, pearson_loss, pearson_metric
import os
from tqdm import tqdm
import numpy as np
import utils
import random


def main(chkpt_path):
    local = True
    if local:
        root_dir = "/Volumes/Datasets/ICASSP_2024_EEG/split_data/"    
    else:
        root_dir = "/home/karan/sda_link/datasets/ICASSP_2024_EEG/split_data/"

    # Set a random seed for Python's random module
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # If you're using CUDA, set the seed for CUDA as well
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(random_seed)

    # Parameters
    batch_size = 64
    shuffle = True
    num_workers = 16
    learning_rate = 0.01
    num_epochs = 1000
    valid_epochs = 20
    display_loss = 2
    steps = 0

    # get the parent directory of the checkpoint path
    save_dir = os.path.dirname(chkpt_path)
   
    # DataLoader
    test_dataset = EegAudioDataset(data_path= "./evaluation_datasets/DTU/", mode="DTU")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)

    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VLAAI().to(device)
    criterion =  pearson_loss # Assuming classification problem, modify if not
    
    # Testing
    model.eval()
    test_loss = []
    test_metric = []
    test_model = utils.load_model(model, chkpt_path, device)

    test_metric_dict = {}

    with torch.no_grad():
        for batch_data in tqdm(test_dataloader):
            inputs, outputs, name = batch_data[0].to(device).squeeze(0), batch_data[1].to(device).squeeze(0), batch_data[2][0].split("/")[-1]

            # Forward pass
            predictions = test_model(inputs)

            # Compute loss
            loss = criterion(outputs.transpose(1, 2), predictions.transpose(1, 2)).mean()

            # Compute metric
            metric = pearson_metric(outputs.transpose(1, 2), predictions.transpose(1, 2))

            test_loss.append(loss.item())
            test_metric.append(metric.detach().cpu().numpy().reshape(-1, 1))
            test_metric_dict[name] = metric.detach().cpu().numpy().reshape(-1, 1)
    
    avg_test_loss = sum(test_loss)/len(test_loss)
    avg_test_metric = np.concatenate(test_metric, axis = 0).mean()

    print(f"DTU dataset | Test Loss : {avg_test_loss:.4f}, Test Metric: {avg_test_metric:.4f}")

    #save the test metric
    model_name = chkpt_path.split("/")[-1].split(".")[0]
    utils.save_pickle(test_metric_dict, f"{save_dir}/DTU_test_metric_{model_name}.pkl")

    print("Testing completed!")

    return

if __name__ == "__main__":
    main(chkpt_path = "/home/karan/sda_link/GitHub/vlaai-pytorch/checkpoints/2023-11-05_10:03:55/model_2023-11-05_10:03:55_300.pt")
    