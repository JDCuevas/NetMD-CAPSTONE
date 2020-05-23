import rcmdd
import torch
import glob
import h5py
import numpy as np
import scipy.io as sio
import platform
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser

parser = ArgumentParser(description='RCMDD')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=80, help='epoch number of end training')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--dataset', type=str, default='RCM', help='dataset from data directory for training')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='../../data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

args = parser.parse_args()

training_data_dir = '%s/%s' % (args.data_dir, args.dataset)
filepath = glob.glob(training_data_dir + '/*ISTA*')[0]

if '.mat' in filepath:
    training_data = sio.loadmat(filepath)
    dataset = training_data['labels']

elif '.hdf5' in filepath:
    with h5py.File(filepath, 'r') as f:
        dataset = np.array(f['labels'][:200000])

class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len

if (platform.system() =="Windows"):
    dataset_loader = DataLoader(dataset=RandomDataset(dataset, dataset.shape[0]), batch_size=args.batch_size, num_workers=0,
                             shuffle=True)
else:
    dataset_loader = DataLoader(dataset=RandomDataset(dataset, dataset.shape[0]), batch_size=args.batch_size, num_workers=4,
                             shuffle=True)

model = rcmdd.RCMDD()

if args.start_epoch > 0:
    pre_model_dir = args.model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

outputs = model.train(dataset_loader, num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=1e-4)
