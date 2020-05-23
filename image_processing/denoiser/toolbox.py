import h5py
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len

def load_dataset(dataset_path, ):
    if '.mat' in dataset_path:
        training_data = sio.loadmat(dataset_path)
        training_labels = training_data['labels']

    elif '.hdf5' in dataset_path:
        with h5py.File(dataset_name, 'w') as f:
            training_data = f['labels']

    else:
        print('Dataset file must be in .mat or .hdf5 file format.')