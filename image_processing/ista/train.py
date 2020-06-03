import os
import platform
import glob
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader

from ista_net import ISTA_Net
from argparse import ArgumentParser
from toolbox import load_dataset, RandomDataset


parser = ArgumentParser(description='Train ISTA-Net-plus')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--batch_size', type=int, default=64, help='epoch number of end training')
parser.add_argument('--cs_ratio', type=int, required=True, help='cs ratio used to create dataset')

parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='../../data', help='data directory')
parser.add_argument('--dataset_name', type=str, default='RCM', help='dataset directory inside data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

parser.add_argument('--sampling_matrix', type=str, required=True, help='path to sampling matrix')
parser.add_argument('--initialization_matrix', type=str, help='path to initialization matrix')

args = parser.parse_args()


# Arguments
start_epoch = args.start_epoch
end_epoch = args.end_epoch
batch_size = args.batch_size
cs_ratio = args.cs_ratio
phi_path = args.sampling_matrix
qinit_path = args.initialization_matrix
data_dir = args.data_dir
dataset_name = args.dataset_name
model_dir = args.model_dir
log_dir = args.log_dir


# Load Dataset
dataset_path = glob.glob('%s/%s/*Training_Data_ISTA*' % (data_dir, dataset_name))[0]
training_labels = load_dataset(dataset_path)

# Train Model
ista_net = ISTA_Net()

## Loading Sampling Matrix
ista_net.load_phi(phi_path)

## Computing Initialization Matrix:
if qinit_path:
    qinit_data = sio.loadmat(qinit_path)
    qinit = qinit_data['Qinit']

else:
    X_data = training_labels.transpose()
    Y_data = np.dot(ista_net.Phi_input, X_data)
    Y_YT = np.dot(Y_data, Y_data.transpose())
    X_YT = np.dot(X_data, Y_data.transpose())
    Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
    del X_data, Y_data, X_YT, Y_YT

    qinit_dir = './initialization_matrix/%s' % (dataset_name)
    qinit_path = '%s/Initialization_Matrix_%d.mat' % (qinit_dir, cs_ratio)

    if not os.path.exists(qinit_dir):
        os.makedirs(qinit_dir)

    sio.savemat(qinit_path, {'Qinit': Qinit})

ista_net.load_qinit(qinit_path)

model_save_dir = '%s/%s' % (model_dir, dataset_name)
log_save_dir = '%s/%s' % (log_dir, dataset_name)

ista_net._train(training_labels, batch_size, start_epoch, end_epoch, cs_ratio, model_save_dir, log_save_dir)