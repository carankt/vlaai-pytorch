import torch
from torch.utils.data import Dataset
import utils
import os
import numpy as np

class EegAudioDataset(Dataset):
    def __init__(self, data_path, mode = "train", transform="normalize", window_size=320, hop_size=64):
        """
        Args:
            data (list or similar data structure): Your data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        if mode not in ["train", "val", "test", "train_and_val", "DTU"]:
            raise ValueError(f"Mode {mode} not supported")
        
        if mode == "train_and_val":
            self.eeg_paths = utils.find_files_with_prefix_suffix(data_path, prefix="train", suffix="_eeg.npy")
            self.eeg_paths.extend(utils.find_files_with_prefix_suffix(data_path, prefix="val", suffix="_eeg.npy"))
            self.subjects = list(set(["".join(os.path.basename(x).split("_")[2]) for x in self.eeg_paths]))

        elif mode == "DTU":
            self.eeg_paths = utils.find_files_with_prefix_suffix(data_path, prefix=mode, suffix=".npz")
            self.subjects = list(set(["".join(os.path.basename(x).split("_")[1]) for x in self.eeg_paths]))
        
        else:
            self.eeg_paths = utils.find_files_with_prefix_suffix(data_path, prefix=mode, suffix="_eeg.npy")
            self.subjects = list(set(["".join(os.path.basename(x).split("_")[2]) for x in self.eeg_paths]))
        
        print(f"Found {len(self.subjects)} Subject for {mode}")
        print(f"Found {len(self.eeg_paths)} paths for {mode}")


        self.transform = transform
        self.window_size = window_size
        self.hop_size = hop_size
        self.mode = mode

    def __len__(self):
        return len(self.eeg_paths)

    def __getitem__(self, idx):
        if self.mode == "DTU":
            eeg = np.load(self.eeg_paths[idx])["eeg"]
            audio = np.load(self.eeg_paths[idx])["envelope"]
        
        else:
            eeg = np.load(self.eeg_paths[idx])
            try:
                audio = np.load(self.eeg_paths[idx].replace("eeg.npy", "envelope.npy"))
            except:
                print(f"Could not find envelope for {self.eeg_paths[idx]}")
                return None

        if self.transform == "normalize":
            # Standardize EEG and envelope
            eeg = (eeg - eeg.mean(axis=0, keepdims=True)) / eeg.std(
                axis=0, keepdims=True
            )
            audio = (
                audio - audio.mean(axis=0, keepdims=True)
            ) / audio.std(axis=0, keepdims=True)

        windowed_eeg = utils.window_data(eeg, self.window_size, self.hop_size)
        windowed_audio = utils.window_data(audio, self.window_size, self.hop_size)

        # pick a random window
        #random_index = np.random.randint(0, windowed_eeg.shape[0])
        
        # select 64 random windows
        if self.mode == "train" or self.mode == "train_and_val":
            random_index = np.random.randint(0, windowed_eeg.shape[0], 1)
            windowed_eeg = windowed_eeg[random_index]
            windowed_audio = windowed_audio[random_index]

            windowed_eeg = torch.from_numpy(windowed_eeg).float().transpose(1,2).squeeze(0)
            windowed_audio = torch.from_numpy(windowed_audio).float().transpose(1,2).squeeze(0)
        else:
            windowed_eeg = torch.from_numpy(windowed_eeg).float().transpose(1,2).squeeze(0)
            windowed_audio = torch.from_numpy(windowed_audio).float().transpose(1,2).squeeze(0)
        #print(windowed_eeg.shape)
        #print(windowed_audio.shape)

        if self.mode == "DTU":
            return windowed_eeg, windowed_audio, self.eeg_paths[idx]
        else:
            return windowed_eeg, windowed_audio

if __name__ == "__main__":
    local = False
    if local == True:
        dataset = EegAudioDataset(data_path="/Volumes/Datasets/ICASSP_2024_EEG/split_data/", mode = "test")
    else:
        dataset = EegAudioDataset(data_path="/home/karan/sda_link/datasets/ICASSP_2024_EEG/split_data/", mode ="val")

    dataset = EegAudioDataset(data_path="./evaluation_datasets/DTU/", mode = "DTU")

    from torch.utils.data import DataLoader

    # Create DataLoader
    batch_size = 1
    shuffle = True
    num_workers = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Iterating over batches
    for batch_data in dataloader:
        # Your training or processing code here
        print(batch_data[0].shape)
        print(batch_data[1].shape)
        print(batch_data[2][0])
        break
        
