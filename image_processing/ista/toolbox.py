import cv2
import math
import torch
import h5py
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset

# Add padding to image for block extraction
def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]

# Extracts vectorized image blocks
def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col

# Reconstructs image from vectorized image blocks
def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col] # Removes padding
    return X_rec

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def snr(orig_img, recon_img):
    orig_img.astype(np.float32)
    recon_img.astype(np.float32)

    orig_img = orig_img.reshape((-1))
    recon_img = recon_img.reshape((-1))

    signal = orig_img
    noise = recon_img - orig_img

    per_pixel_snr = []

    for i in range(len(signal)):
        if noise[i] != 0:
            per_pixel_snr.append(signal[i] / noise[i])

    snr = np.mean(np.array(per_pixel_snr))
    
    return snr

def extract_luminance(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_y = img_yuv[:,:,0]

    return img_y

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
