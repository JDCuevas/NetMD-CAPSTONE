import rcmdd
import torch
import glob
import h5py
import numpy as np
import scipy.io as sio
import platform
from argparse import ArgumentParser
from toolbox import load_dataset_rcmdd

parser = ArgumentParser(description='RCMDD')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=80, help='epoch number of end training')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
parser.add_argument('--cs_ratio', type=int, required=True, help='cs ratio used to create dataset')

parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='../../data', help='data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

args = parser.parse_args()


# Arguments
start_epoch = args.start_epoch
end_epoch = args.end_epoch
batch_size = args.batch_size
dataset = args.dataset
cs_ratio = args.cs_ratio

model_dir = args.model_dir
data_dir = args.data_dir
dataset_name = args.dataset_name
log_dir = args.log_dir

# Load Dataset
dataset_path = glob.glob('%s/%s/*Training_Data_RCMDD*' % (data_dir, dataset_name))[0]
X_train, y_train = load_dataset_rcmdd(dataset_path)

# Train model
model = rcmdd.RCMDD()

model_save_dir = '%s/%s' % (model_dir, dataset_name)
log_save_dir = '%s/%s' % (log_dir, dataset_name)

model.train(X_train, y_train, batch_size, start_epoch, end_epoch, cs_ratio, model_dir, log_dir)
