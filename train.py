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

# Assuming your model and dataset classes are imported
# from model import VLAAI
# from dataset import EegAudioDataset

def main():
    local = False
    if local:
        root_dir = "/Volumes/Datasets/ICASSP_2024_EEG/split_data/"    
    else:
        root_dir = "/home/karan/sda_link/datasets/ICASSP_2024_EEG/split_data/"

    # generate a ID based on date and time
    import datetime
    now = datetime.datetime.now()
    ID = now.strftime("%Y-%m-%d_%H:%M:%S")

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
    num_workers = 32
    learning_rate = 0.01
    num_epochs = 1000
    valid_epochs = 20
    display_loss = 2
    steps = 0
    save_dir = f"./checkpoints/{ID}/"
    comment = "VLAAI with 64 windows chosen randomly"
    EarlyStopping = utils.EarlyStopping(patience=5)
    logging_file = f"{save_dir}/training_log_{ID}.txt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Logging the training parameters
    with open(logging_file, "a") as f:
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"shuffle: {shuffle}\n")
        f.write(f"num_workers: {num_workers}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"valid_epochs: {valid_epochs}\n")
        f.write(f"display_loss: {display_loss}\n")
        f.write(f"save_dir: {save_dir}\n")
        f.write(f"logging_file: {logging_file}\n")
        f.write(f"comment: {comment}\n")

    # DataLoader
    train_dataset = EegAudioDataset(data_path= root_dir, mode="train_and_val")
    val_dataset = EegAudioDataset(data_path= root_dir, mode="test")
    test_dataset = EegAudioDataset(data_path= "./evaluation_datasets/DTU/", mode="DTU")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)

    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VLAAI().to(device)
    criterion =  pearson_loss # Assuming classification problem, modify if not
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    train_loss = []

    for epoch in tqdm(range(1, num_epochs)):
        if EarlyStopping.early_stop == True:
            print("Early stopping")
            break
        else:
            for batch_data in train_dataloader:
                inputs, outputs = batch_data[0].to(device).squeeze(0), batch_data[1].to(device).squeeze(0)

                # Forward pass
                predictions = model(inputs)
                
                # Compute loss
                loss = criterion(outputs.transpose(1, 2), predictions.transpose(1, 2)).mean()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

                steps += 1

            if epoch % valid_epochs == 0:
                # Compute validation loss and metric
                model.eval()
                val_loss = []
                val_metric = []
                with torch.no_grad():
                    for batch_data in val_dataloader:
                        inputs, outputs = batch_data[0].to(device).squeeze(0), batch_data[1].to(device).squeeze(0)

                        # Forward pass
                        predictions = model(inputs)

                        # Compute loss
                        loss = criterion(outputs.transpose(1, 2), predictions.transpose(1, 2)).mean()

                        # Compute metric
                        metric = pearson_metric(outputs.transpose(1, 2), predictions.transpose(1, 2))

                        val_loss.append(loss.item())
                        val_metric.append(metric.detach().cpu().numpy().reshape(-1, 1))
                
                avg_val_loss = sum(val_loss)/len(val_loss)
                avg_val_metric = np.concatenate(val_metric, axis = 0).mean()

                print(f"Epoch [{epoch + 1}/{num_epochs}] | Step [{steps}/{num_epochs * len(train_dataloader)}] | Valid Loss: {avg_val_loss:.4f}, Metric: {avg_val_metric:.4f}")

                torch.save(model.state_dict(), save_dir + f"model_{ID}_{epoch}.pt")
                print("Model saved!")

                ## logging
                with open(logging_file, "a") as f:
                    f.write(f"Epoch [{epoch + 1}/{num_epochs}] | Step [{steps}/{num_epochs * len(train_dataloader)}] | Valid Loss: {avg_val_loss:.4f}, Metric: {avg_val_metric:.4f}\n")

                # Early stopping
                EarlyStopping.step(avg_val_metric)
                model.train()
            
            if epoch % display_loss == 0:
                avg_train_loss = sum(train_loss)/len(train_loss)
                # Print loss every epoch
                print(f"Epoch [{epoch + 1}/{num_epochs}] | Step [{steps}/{num_epochs * len(train_dataloader)}] | Train Loss: {avg_train_loss:.4f}, Metric: ")
                ## logging
                with open(logging_file, "a") as f:
                    f.write(f"Epoch [{epoch + 1}/{num_epochs}] | Step [{steps}/{num_epochs * len(train_dataloader)}] | Train Loss: {avg_train_loss:.4f}, Metric: \n")
                train_loss = []

    print("Training completed!")
    # Testing
    model.eval()
    test_loss = []
    test_metric = []
    test_model = utils.load_model(model, f"model_{ID}_{epoch}.pt", device)

    test_metric_dict = {}

    with torch.no_grad():
        for batch_data in test_dataloader:
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
    utils.save_pickle(test_metric_dict, f"{save_dir}/DTU_test_metric_{ID}_{epoch}.pkl")

    # logging
    with open(logging_file, "a") as f:
        f.write(f"DTU dataset | Test Loss : {avg_test_loss:.4f}, Test Metric: {avg_test_metric:.4f}\n")

    print("Testing completed!")

    return

if __name__ == "__main__":
    main()
    
