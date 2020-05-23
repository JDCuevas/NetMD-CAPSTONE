import cv2
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np

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
        noisy_img = torch.from_numpy(noisy_img)
        noisy_img = noisy_img.type(torch.FloatTensor)
        noisy_img /= 255

        denoised_img = self.model(noisy_img)
        denoised_img = denoised_img.cpu().data.numpy()
        denoised_img *= 255

        return denoised_img

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))    

    def _train(self, dataset_loader, num_epochs=80, batch_size=128, learning_rate=1e-4):
        mse = nn.MSELoss() 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        outputs = []

        for epoch in range(num_epochs):
            for data in dataset_loader:
                imgs = data
                recon = self.model(imgs)
                loss = mse(recon, imgs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
            outputs.append((epoch, imgs, recon),)
            
        return outputs