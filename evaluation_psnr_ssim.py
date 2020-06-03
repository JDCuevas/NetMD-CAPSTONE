import os
import cv2
from time import time
import numpy as np
import torch
import glob
import scipy.io as sio
from image_processing.ista.ista_net import ISTA_Net
from toolbox import imread_CS_py, img2col_py, psnr
from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser



cs_ratios = [10, 25, 50]


for cs_ratio in cs_ratios:

    net = ISTA_Net()
    net.load_phi(phi_path='image_processing/ista/sampling_matrix/phi_0_%d_1089.mat' % (cs_ratio))
    net.load_qinit(qinit_path='image_processing/ista/initialization_matrix/Natural_Images/Initialization_Matrix_%d.mat' % (cs_ratio))
    net.load_model('image_processing/ista/model/Natural_Images/CS_ISTA_Net_plus_ratio_%d/net_params.pkl' % (cs_ratio))

    Phi_data_Name = 'image_processing/ista/sampling_matrix/phi_0_%d_1089.mat' % (cs_ratio)
    Phi_data = sio.loadmat(Phi_data_Name)
    Phi_input = Phi_data['phi']
    Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)


    test_dir ='data/Natural_Images/testing_imgs'
    filepaths = glob.glob(test_dir + '/*.tif')

    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

    print('\n')
    print("CS Reconstruction Start | CS Ratio %d" % (cs_ratio))

    with torch.no_grad():
        for img_no in range(ImgNum):

            imgName = filepaths[img_no]

            Img = cv2.imread(imgName, 1)

            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            Img_rec_yuv = Img_yuv.copy()

            Iorg_y = Img_yuv[:,:,0]

            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
            Icol = img2col_py(Ipad, 33).transpose()/255.0

            cs_measurements = np.dot(Icol, np.transpose(Phi))

            start = time()

            recon = net.reconstruct(cs_measurements=cs_measurements, orig_width=row, orig_height=col)

            end = time()

            rec_PSNR = psnr(recon, Iorg.astype(np.float64))
            rec_SSIM = ssim(recon, Iorg.astype(np.float64), data_range=255)

            print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))

            Img_rec_yuv[:,:,0] = recon

            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM

    print('\n')
    output_data = "CS ratio is %d, Avg PSNR/SSIM for %s is %.2f/%.4f.\n" % (cs_ratio, 'Natural Images Dataset', np.mean(PSNR_All), np.mean(SSIM_All))
    print(output_data)