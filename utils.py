import os
import numpy as np
import pickle
import numpy as np
import torch

def find_npy_files(root_dir):
    """
    Find all .npy files under a directory and its subdirectories.
    
    Parameters:
    - root_dir: The root directory to start the search from.
    
    Returns:
    - A list of full paths to .npy files.
    """
    npy_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.npy'):
                full_path = os.path.join(dirpath, filename)
                npy_files.append(full_path)

    return npy_files

import os

def parse_bids_filename(filepath):
    """
    Parse a BIDS-format filepath to extract its components.
    
    Parameters:
    - filepath: Full path to the file following BIDS convention.
    
    Returns:
    - Dictionary with the identified BIDS components.
    """
    bids_components = {}

    # Extract the filename and its components
    directory, filename = os.path.split(filepath)
    base, ext = os.path.splitext(filename)

    # Use split to separate components
    components = base.split("_")

    for component in components:
        key, value = component.split("-")[0], "-".join(component.split("-")[1:])
        if key in ['sub', 'ses', 'task', 'run']:
            bids_components[key] = value
        elif key == 'desc':
            # This will ensure that we capture all the parts of the descriptor after 'desc-' till the end of the filename (excluding the extension).
            # Special handling for desc since it might contain additional underscores
            desc_index = base.index("desc-")
            eeg_index = base.index("_eeg")
            bids_components['desc'] = base[desc_index + len("desc-"):eeg_index][14:] # just get the audio descriptor
            break  # Since desc is the last identifiable component before the modality suffix (eeg), we break out.

    return bids_components

def find_files_with_prefix_suffix(search_dir, prefix, suffix):
    """
    Find files in a directory (and its subdirectories) that start with a specific prefix and end with a specific suffix.
    
    Parameters:
    - search_dir: The directory in which to start the search.
    - prefix: The prefix string that files should start with.
    - suffix: The suffix string that files should end with.
    
    Returns:
    - A list of full paths to files matching the criteria.
    """
    matching_files = []

    for dirpath, dirnames, filenames in os.walk(search_dir):
        for filename in filenames:
            if filename.startswith(prefix) and filename.endswith(suffix):
                full_path = os.path.join(dirpath, filename)
                matching_files.append(full_path)

    return matching_files

def save_pickle(data, filename):
    """
    Save data to a pickle file.

    Parameters:
    - data: The data to be saved.
    - filename: The filename of the pickle file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
def load_pickle(filename):
    """
    Load data from a pickle file.

    Parameters:
    - filename: The filename of the pickle file.

    Returns:
    - The data loaded from the pickle file.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def window_data(data, window_length, hop):
    """Window data into overlapping windows.

    Parameters
    ----------
    data: np.ndarray
        Data to window. Shape (n_samples, n_channels)
    window_length: int
        Length of the window in samples.
    hop: int
        Hop size in samples.

    Returns
    -------
    np.ndarray
        Windowed data. Shape (n_windows, window_length, n_channels)
    """
    new_data = np.empty(
        ((data.shape[0] - window_length) // hop, window_length, data.shape[1])
    )
    for i in range(new_data.shape[0]):
        new_data[i, :, :] = data[
            i * hop : i * hop + window_length, :  # noqa: E203 E501
        ]
    return new_data

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, current_score):
        """
        Check if early stopping criteria is met.

        Args:
            current_score (float): Current validation score.

        Returns:
            bool: Whether training should be stopped.
        """
        if self.best_score is None:
            self.best_score = current_score
        
        elif current_score >= self.best_score:  # Assuming we want to minimize the score (e.g., loss). For accuracy, you should change this condition.
            self.best_score = current_score
            self.counter = 0
        
        elif current_score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

def load_model(model, model_path, device):
    """
    Load a model from a file.

    Parameters:
    - model: The model to load the parameters into.
    - model_path: The path to the model file.
    - device: The device to load the model on.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model