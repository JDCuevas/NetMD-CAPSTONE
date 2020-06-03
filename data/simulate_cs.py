import os
import cv2
import ntpath
import numpy as np
import scipy.io as sio
from argparse import ArgumentParser


parser = ArgumentParser(description='Train ISTA-Net-plus')

parser.add_argument('--image', type=str, help='path to image')
parser.add_argument('--sampling_matrix', type=str, required=True, help='path to sampling_matrix')
parser.add_argument('--dataset_name', type=str, default='RCM', help='image dataset name')
parser.add_argument('--cs_ratio', type=str, required=True, help='cs ratio')


args = parser.parse_args()

image_path = args.image
phi_path = args.sampling_matrix
cs_ratio = args.cs_ratio
dataset_name = args.dataset_name


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

img_name, _ = os.path.splitext(ntpath.basename(image_path))
img = cv2.imread(image_path, 1)

phi_data = sio.loadmat(phi_path)
phi = phi_data['phi']

img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
img_rec_yuv = img_yuv.copy()

i_org_y = img_yuv[:,:,0]

[Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(i_org_y)
vectorized_img_blocks = img2col_py(Ipad, 33).transpose()/255.0

cs_measurements = np.dot(vectorized_img_blocks, np.transpose(phi))

output_dir = '../cs_test_samples/' + dataset_name + '/cs_' + cs_ratio + '/' 

if not os.path.exists(output_dir):
        os.makedirs(output_dir)

sio.savemat(output_dir + img_name + '_cs.mat', {'cs_measurements': cs_measurements})
