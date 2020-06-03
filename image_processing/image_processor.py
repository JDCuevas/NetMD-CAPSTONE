import os
from ista.ista_net import ISTA_Net
from denoiser.rcmdd import RCMDD

class Image_Processor():
    def __init__(self, denoiser_status):
        self.ista_net = ISTA_Net()
        self.rcmdd = RCMDD()

        self.denoiser_status = denoiser_status


    def process(self, cs_measurements, width, height, phi_path, qinit_path):
        self.ista_net.load_phi(phi_path)
        self.ista_net.load_qinit(qinit_path)
        
        # Reconstruct and denoise CS image
        img_recon = self.ista_net.reconstruct(cs_measurements, width, height)

        if self.denoiser_status == 'on':
            img_recon = self.rcmdd.denoise(img_recon)

        return img_recon

    def load_models(self, ista_models_dir, rcmdd_models_dir, cs_ratio):
        self.ista_net.load_model(os.path.join(ista_models_dir, 'CS_ISTA_Net_plus_ratio_%d/net_params.pkl' % (cs_ratio)))
        
        if self.denoiser_status == 'on':
            self.rcmdd.load_model(os.path.join(rcmdd_models_dir, 'RCMDD_ratio_%d/net_params.pkl' % (cs_ratio)))