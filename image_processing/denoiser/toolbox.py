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

def load_dataset(dataset_path):
    
    if '.mat' in dataset_path:
        training_data = sio.loadmat(dataset_path)
        training_labels = training_data['labels']
        training_labels = np.array(training_labels)

    elif '.hdf5' in dataset_path:
        with h5py.File(dataset_path, 'r') as training_data:
            training_labels = training_data['labels']
            training_labels = np.array(training_labels)
    else:
        print('Dataset file must be in .mat or .hdf5 file format.')

    return training_labels