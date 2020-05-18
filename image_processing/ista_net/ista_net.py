import cv2
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
from architecture import ISTANetModel
from toolbox import imread_CS_py, img2col_py, col2im_CS_py

class ISTA_Net():
    def __init__(self, phi_path, qinit_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = ISTANetModel(9)
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        self.Phi_data_Name = phi_path
        self.Phi_data = sio.loadmat(self.Phi_data_Name)
        self.Phi_input = self.Phi_data['phi']
        self.Phi = torch.from_numpy(self.Phi_input).type(torch.FloatTensor)
        self.Phi = self.Phi.to(self.device)

        self.Qinit_Name = qinit_path
        self.Qinit_data = sio.loadmat(self.Qinit_Name)
        self.Qinit = self.Qinit_data['Qinit']
        self.Qinit = torch.from_numpy(self.Qinit).type(torch.FloatTensor)
        self.Qinit = self.Qinit.to(self.device)

        self.block_size = 33

        self.last_rec = None

    def reconstruct(self, cs_measurements, orig_width, orig_height):
        Phix = cs_measurements

        [X_output, _] = self.model(Phix, self.Phi, self.Qinit)

        Prediction_value = X_output.cpu().data.numpy()

        row_pad = self.block_size-np.mod(orig_width, self.block_size)
        col_pad = self.block_size-np.mod(orig_height, self.block_size)
        pad_width = orig_height + row_pad
        pad_height = orig_height + col_pad

        X_rec = np.clip(col2im_CS_py(Prediction_value.transpose(), 256, 256, 264, 264), 0, 1)

        img_rec = X_rec * 255
        self.last_rec = img_rec

        del X_output

        return img_rec

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))