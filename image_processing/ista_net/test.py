import cv2
import numpy as np
import torch
import scipy.io as sio
from ista_net import ISTA_Net
from toolbox import imread_CS_py, img2col_py, col2im_CS_py
from argparse import ArgumentParser

parser = ArgumentParser(description='ISTA-Net-plus')

parser.add_argument('--cs_ratio', type=int, default=25, help='from {1, 4, 10, 25, 30, 40, 50}')

args = parser.parse_args()

cs_ratio = args.cs_ratio

net = ISTA_Net(phi_path='sampling_matrix/phi_0_%d_1089.mat' % (cs_ratio), qinit_path='initialization_matrix/Initialization_Matrix_%d.mat' % (cs_ratio))
net.load_model('model/Natural_Images/CS_ISTA_Net_plus_layer_9_ratio_%d_lr_0.0001/net_params_200.pkl' % (cs_ratio))

Phi_data_Name = 'sampling_matrix/phi_0_%d_1089.mat' % (cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']
Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)

Img = cv2.imread('../../data/RCM/training_imgs/Fair1_1_RCM.tif', 1)
print(Img.shape)
cv2.imwrite("Fair1_1_RCM.png", Img)
exit()
Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
Img_rec_yuv = Img_yuv.copy()

Iorg_y = Img_yuv[:,:,0]

[Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
Icol = img2col_py(Ipad, 33).transpose()/255.0
Icol = torch.from_numpy(Icol).type(torch.FloatTensor)

cs_measurements = torch.mm(Icol, torch.transpose(Phi, 0, 1))

recon = net.reconstruct(cs_measurements=cs_measurements, orig_width=row, orig_height=col)

cv2.imwrite("test_ratio_%d.png" % (cs_ratio), recon)
