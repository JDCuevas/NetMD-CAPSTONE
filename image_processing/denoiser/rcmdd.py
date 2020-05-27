import os
import cv2
import torch
import platform
import torch.nn as nn
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from toolbox import RandomDataset

class RCMDDModel(nn.Module):
    def __init__(self):
        super(RCMDDModel, self).__init__()

        #encode layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(11, 11), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(7, 7), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1), stride=2, padding=1),
        )
       
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=(1, 1), stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(7, 7), stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(11, 11), stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class RCMDD():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = RCMDDModel()
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    def denoise(self, noisy_img):
        width, height = noisy_img.shape[0], noisy_img.shape[1]
        noisy_img = noisy_img.reshape((1, 1, width, height))
        noisy_img = torch.from_numpy(noisy_img)
        noisy_img = noisy_img.type(torch.FloatTensor)
        noisy_img /= 255

        denoised_img = self.model(noisy_img)
        denoised_img = denoised_img.cpu().data.numpy()
        denoised_img *= 255
        denoised_img = denoised_img.reshape((width, height))
        return denoised_img

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))    

    def _train(self, dataset, batch_size, start_epoch, end_epoch, cs_ratio, model_dir, log_dir):
        train_size = dataset.shape[0]
        mse = nn.MSELoss() 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1-4)

        model_dir = "./%s/CS_RCMDD_ratio_%d" % (model_dir, cs_ratio)
        log_dir = "./%s/Log_CS_RCMDD_ratio_%d" % (log_dir, cs_ratio)
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)


        if (platform.system() =="Windows"):
            rand_loader = DataLoader(dataset=RandomDataset(dataset, train_size), batch_size=batch_size, num_workers=0,
                                    shuffle=True)
        else:
            rand_loader = DataLoader(dataset=RandomDataset(dataset, train_size), batch_size=batch_size, num_workers=4,
                                    shuffle=True)


        if start_epoch > 0:
            pre_model_dir = model_dir
            self.model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

        outputs = []

        for epoch_i in range(start_epoch + 1, end_epoch + 1):
            for data in rand_loader:
                imgs = data

                recon = self.model(imgs)

                loss = mse(recon, imgs)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                output_data = "[%02d/%02d] Total Loss: %.4f\n" % (epoch_i, end_epoch, loss.item())
                print(output_data)

                output_file = open(log_dir + '/log.txt', 'a')
                output_file.write(output_data)
                output_file.close()

            if epoch_i % 5 == 0 and epoch_i != end_epoch:
                torch.save(self.model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
            elif epoch_i == end_epoch:
                torch.save(self.model.state_dict(), "./%s/net_params.pkl" % (model_dir))