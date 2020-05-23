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
parser.add_argument('--cs_ratio', type=int, default=10, help='from {1, 4, 10, 25, 40, 50}')

parser.add_argument('--matrix_path', type=str, required=True, help='path to sampling matrix')
parser.add_argument('--qinit_path', type=str, help='path to initialization matrix')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='../../data', help='data directory')
parser.add_argument('--dataset_name', type=str, default='RCM', help='dataset name')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
batch_size = args.batch_size
cs_ratio = args.cs_ratio
phi_path = args.matrix_path
qinit_path = args.qinit_path
data_dir = args.data_dir
dataset_name = args.dataset_name
model_dir = args.model_dir
log_dir = args.log_dir

dataset_path = glob.glob(data_dir + '/' + dataset_name + '/*Training_Data_ISTA*')[0]

training_labels = load_dataset(dataset_path)

ista_net = ISTA_Net()
ista_net.load_phi(phi_path)

# Computing Initialization Matrix:
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

    qinit_path = './initialization_matrix/%s/Initialization_Matrix_%d.mat' % (dataset_name, cs_ratio)
    sio.savemat(qinit_path, {'Qinit': Qinit})

ista_net.load_qinit(qinit_path)

model_save_dir = '%s/%s' % (model_dir, dataset_name)
log_save_dir = '%s/%s' % (log_dir, dataset_name)

ista_net._train(training_labels, batch_size, start_epoch, end_epoch, cs_ratio, model_save_dir, log_save_dir)