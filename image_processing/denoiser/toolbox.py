import h5py
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len

def load_dataset_rcmdd(dataset_path):
    
    if '.mat' in dataset_path:
        training_data = sio.loadmat(dataset_path)

        training_input = training_data['X_train']
        X = np.array(training_input)

        training_labels = training_data['y_train']
        y = np.array(training_labels)


    elif '.hdf5' in dataset_path:
        with h5py.File(dataset_path, 'r') as training_data:
            training_input = training_data['X_train']
            X = np.array(training_input)

            training_labels = training_data['y_train']
            y = np.array(training_labels)
    else:
        print('Dataset file must be in .mat or .hdf5 file format.')

    return X, y

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)