import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init

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