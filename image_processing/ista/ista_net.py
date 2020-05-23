import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import platform
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
from toolbox import imread_CS_py, img2col_py, col2im_CS_py, RandomDataset

class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, 33, 33)

        x_D = F.conv2d(x_input, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_G

        x_pred = x_pred.view(-1, 1089)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]


# Define ISTA-Net-plus
class ISTANetModel(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANetModel, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit):

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]


class ISTA_Net():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.layer_num = 9 
        self.block_size = 33

        self.model = ISTANetModel(self.layer_num)
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        self.last_rec = None

    def reconstruct(self, cs_measurements, orig_width, orig_height):
        Phix = cs_measurements

        [X_output, _] = self.model(Phix, self.Phi, self.Qinit)

        Prediction_value = X_output.cpu().data.numpy()

        row_pad = self.block_size-np.mod(orig_width, self.block_size)
        col_pad = self.block_size-np.mod(orig_height, self.block_size)
        pad_width = orig_height + row_pad
        pad_height = orig_height + col_pad

        X_rec = np.clip(col2im_CS_py(Prediction_value.transpose(), orig_width, orig_height, pad_width, pad_height), 0, 1)

        img_rec = X_rec * 255
        self.last_rec = img_rec

        del X_output

        return img_rec

    def load_phi(self, phi_path):
        self.Phi_data_Name = phi_path
        self.Phi_data = sio.loadmat(self.Phi_data_Name)
        self.Phi_input = self.Phi_data['phi']
        self.Phi = torch.from_numpy(self.Phi_input).type(torch.FloatTensor)
        self.Phi = self.Phi.to(self.device)
    
    def load_qinit(self, qinit_path):
        self.Qinit_Name = qinit_path
        self.Qinit_data = sio.loadmat(self.Qinit_Name)
        self.Qinit = self.Qinit_data['Qinit']
        self.Qinit = torch.from_numpy(self.Qinit).type(torch.FloatTensor)
        self.Qinit = self.Qinit.to(self.device)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

    def _train(self, dataset, batch_size, start_epoch, end_epoch, cs_ratio, model_dir, log_dir):
        train_size = dataset.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

 
        model_dir = "./%s/CS_ISTA_Net_plus_layer_%d_ratio_%d_lr_%.4f" % (model_dir, self.layer_num, cs_ratio, 0.0001)
        log_dir = "./%s/Log_CS_ISTA_Net_plus_layer_%d_ratio_%d_lr_%.4f" % (log_dir, self.layer_num, cs_ratio, 0.0001)
        

        if (platform.system() =="Windows"):
            rand_loader = DataLoader(dataset=RandomDataset(dataset, train_size), batch_size=batch_size, num_workers=0,
                                    shuffle=True)
        else:
            rand_loader = DataLoader(dataset=RandomDataset(dataset, train_size), batch_size=batch_size, num_workers=4,
                                    shuffle=True)


        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)


        if start_epoch > 0:
            pre_model_dir = model_dir
            self.model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

        for epoch_i in range(start_epoch + 1, end_epoch + 1):
            for data in rand_loader:

                batch_x = data
                batch_x = batch_x.to(self.device)

                Phix = torch.mm(batch_x, torch.transpose(self.Phi, 0, 1))

                [x_output, loss_layers_sym] = self.model(Phix, self.Phi, self.Qinit)

                # Compute and print loss
                loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

                loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
                for k in range(self.layer_num-1):
                    loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))

                gamma = torch.Tensor([0.01]).to(self.device)

                loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss_all.backward()
                optimizer.step()

                output_data = "[%02d/%02d] Total Loss: %.4f, Discrepancy Loss: %.4f,  Constraint Loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item(), loss_constraint)
                print(output_data)
            
                output_file = open(log_dir + '/log.txt', 'a')
                output_file.write(output_data)
                output_file.close()

                if epoch_i % 5 == 0:
                    torch.save(self.model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters

