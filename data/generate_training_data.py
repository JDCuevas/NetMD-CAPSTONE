import os
import cv2
import glob
import h5py
import numpy as np
import scipy.io as sio

from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser(description='ISTA-Net Training Data Generator')

parser.add_argument('--data_dir', type=str, default='data', help='training or test data directory')
parser.add_argument('--dataset_name', type=str, default='RCM', help='name of dataset')
parser.add_argument('--output_name', type=str, default='Training_Data', help='name of dataset')

args = parser.parse_args()

def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


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


def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec

# Load images
#imgs_dir = os.path.join(args.data_dir, args.dataset_name)
imgs_dir = '/' + args.dataset_name
filepaths = glob.glob(imgs_dir + '/*.tif')

num_imgs = len(filepaths)

dataset = np.zeros(shape=(874509, 1089))
last_idx = 0

# Extract vectorized cropped img blocks
for img_no in tqdm(range(num_imgs), desc='Vectorized Image Blocks'):
        imgName = filepaths[img_no]

        Img = cv2.imread(imgName, 1)

        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
        Img_rec_yuv = Img_yuv.copy()

        Iorg_y = Img_yuv[:,:,0]

        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
        vectorized_img_blocks = img2col_py(Ipad, 33).transpose()/255.0

        dataset[last_idx : last_idx + vectorized_img_blocks.shape[0]] = vectorized_img_blocks
        last_idx = last_idx + vectorized_img_blocks.shape[0]


# Shuffle 
np.random.shuffle(dataset)
print(dataset.shape)

# Save dataset

dataset_name = './%s/%s.hdf5' % (args.dataset_name, 'Training_Data')
#sio.savemat(dataset_name, {'labels': dataset[:500000]}, do_compression=True)

with h5py.File(dataset_name, 'w') as f:
        dset = f.create_dataset('labels', data=dataset)