import sys
import cv2
import numpy as np
import torch
import scipy.io as sio
from ista import ISTA_Net
from denoiser import RCMDD
from ista.toolbox import imread_CS_py, img2col_py, col2im_CS_py
from argparse import ArgumentParser


parser = ArgumentParser(description='ISTA-Net + RCMDD')

parser.add_argument('--cs_ratio', type=int, default=25, help='from {10, 25, 50}')

args = parser.parse_args()

cs_ratio = args.cs_ratio


ista_net = ISTA_Net()
ista_net.load_model('ista/model/Natural_Images/CS_ISTA_Net_plus_ratio_%d/net_params.pkl' % (cs_ratio))
ista_net.load_phi(phi_path='ista/sampling_matrix/phi_0_%d_1089.mat' % (cs_ratio))
ista_net.load_qinit(qinit_path='ista/initialization_matrix/Natural_Images/Initialization_Matrix_%d.mat' % (cs_ratio))

rcmdd = RCMDD()
#rcmdd.load_model('denoiser/model/Natural_Images/CS_RCMDD_ratio_%d_lr_0.0001/net_params_80.pkl' % (cs_ratio))

# Simulating CS Measurements
Phi_data_Name = 'ista/sampling_matrix/phi_0_%d_1089.mat' % (cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi = Phi_data['phi']

Img = cv2.imread('../data/Natural_Images/testing_imgs/barbara.tif', 1)

Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
Img_rec_yuv = Img_yuv.copy()

Iorg_y = Img_yuv[:,:,0]

[Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
vectorized_img_blocks = img2col_py(Ipad, 33).transpose()/255.0

cs_measurements = np.dot(vectorized_img_blocks, np.transpose(Phi))

recon = ista_net.reconstruct(cs_measurements=cs_measurements, orig_width=row, orig_height=col)
denoised_recon = rcmdd.denoise(recon)

cv2.imwrite("test_ratio_%d.png" % (cs_ratio), denoised_recon)