import rcmdd
import torch
import glob
import h5py
import numpy as np
import scipy.io as sio
import platform
from argparse import ArgumentParser
from toolbox import load_dataset

parser = ArgumentParser(description='RCMDD')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=80, help='epoch number of end training')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
parser.add_argument('--cs_ratio', type=int, required=True, help='cs ratio used to create dataset')

parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='../../data', help='training data directory')
parser.add_argument('--dataset_name', type=str, required=True, help='dataset name')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
batch_size = args.batch_size
learning_rate = args.learning_rate
dataset = args.dataset
cs_ratio = args.cs_ratio

model_dir = args.model_dir
data_dir = args.data_dir
dataset_name = args.dataset_name
log_dir = args.log_dir


training_data_dir = '%s/%s' % (data_dir, dataset)
dataset_path = glob.glob('%s/*Training_Data_RCMDD_cs_%d*' % (training_data_dir, cs_ratio))[0]

training_labels = load_dataset(dataset_path)

model = rcmdd.RCMDD()

model_save_dir = '%s/%s' % (model_dir, dataset_name)
log_save_dir = '%s/%s' % (log_dir, dataset_name)

model.train(training_labels, batch_size, start_epoch, end_epoch, cs_ratio, model_dir, log_dir)
