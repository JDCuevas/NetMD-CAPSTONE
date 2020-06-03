import os
import cv2
import ntpath
import numpy as np
import scipy.io as sio
from PySide2 import QtWidgets, QtCore, QtGui
from ui import instructions, home, settings
from image_processing import Image_Processor, snr, extract_luminance
from argparse import ArgumentParser



class NetMDInstructions(instructions.Ui_InstructionsWindow, QtWidgets.QMainWindow):

    def __init__(self, args):
        super(NetMDInstructions, self).__init__()
        self.setupUi(self)
        self.setMinimumHeight(750)
        self.setMinimumWidth(1250)
        self.startButton.clicked.connect(self.get_started)
        
        self.args = args

        self.home = NetMDHome(self)
        self.settings = NetMDSettings(self)

    def get_started(self):
        self.hide()
        self.home.show()

    def settings_back(self):
        self.hide()
        self.settings.show()


class NetMDHome(home.Ui_HomeWindow, QtWidgets.QMainWindow):
    
    def __init__(self, parent=None):
        super(NetMDHome, self).__init__(parent)
        self.setupUi(self)
        self.setMinimumHeight(750)
        self.setMinimumWidth(1250)
        self.actionSettings.triggered.connect(self.settings)
        self.actionInstructions.triggered.connect(self.instructions)

        # Buttons
        self.csMeasurementsButton.clicked.connect(self.select_cs)
        self.samplingMatrixButton.clicked.connect(self.select_phi)
        self.initializationMatrixButton.clicked.connect(self.select_qinit)
        self.origImageButton.clicked.connect(self.select_orig_image)

        self.reconstructButton.clicked.connect(self.reconstruct)
        self.snrButton.clicked.connect(self.calculate_snr)
        self.saveButton.clicked.connect(self.save)

        self.csRatiosComboBox.setCurrentIndex(1)

        # ISTA-Net+RCMDD Variables
        self.cs_measurements_path = None
        self.phi_path = None
        self.qinit_path = None

        self.ista_models_dir = os.path.join(os.getcwd(), 'image_processing/rcmdd/model/RCM/')
        self.rcmdd_models_dir = os.path.join(os.getcwd(), 'image_processing/rcmdd/model/RCM/')
    
        self.img_recon = None
        self.img_recon_flag = False

        self.denoiser_status = self.parent().args.denoiser
        self.image_processor = Image_Processor(self.denoiser_status)
        
    # Go to Instructions screen
    def instructions(self):
        self.hide()
        self.parent().show()

    # Go to settings screen
    def settings(self):
        self.hide()
        self.parent().settings_back()

    # Reconstruct uploaded image
    def reconstruct(self):
        cs_measurements_path = self.csMeasurementsPathLine.text()
        phi_path = self.samplingMatrixPathLine.text()
        qinit_path = self.initializationMatrixPathLine.text()

        if not cs_measurements_path:
            QtWidgets.QMessageBox.about(self, "CS Measurements Required", "Please upload cs measurements.")
            return
        elif not phi_path: 
            QtWidgets.QMessageBox.about(self, "Sampling Matrix Required", "Please upload sampling matrix.")
            return
        elif not qinit_path: 
            QtWidgets.QMessageBox.about(self, "Initialization Matrix required", "Please upload initialization matrix.")
            return

        try: 
            width = int(self.widthLineEdit.text())
            height = int(self.heightLineEdit.text())

        except ValueError:
            QtWidgets.QMessageBox.about(self, "Width and Height type error", "Width and Height must be integers.")
            return

        _, cs_file_ext = os.path.splitext(cs_measurements_path)
        _, phi_file_ext = os.path.splitext(phi_path)
        _, qinit_file_ext = os.path.splitext(qinit_path)

        if cs_file_ext != '.mat' or phi_file_ext != '.mat' or qinit_file_ext != '.mat':
            QtWidgets.QMessageBox.about(self, "File Extension Error", "Files must be in .mat format.")

        cs_measurements = np.array(sio.loadmat(cs_measurements_path)['cs_measurements'])
        cs_ratio = int(self.csRatiosComboBox.currentText())

        # Reconstruct image
        self.image_processor.load_models(self.ista_models_dir, self.rcmdd_models_dir, cs_ratio)
        self.img_recon = self.image_processor.process(cs_measurements, width, height, phi_path, qinit_path)
        self.img_recon_flag = True
            
        # Preview reconstruction
        cv2.imwrite('tmp.png', self.img_recon)
        self.img_recon = cv2.imread('tmp.png')
        self.recoveredImageLabel.setPixmap(QtGui.QPixmap('tmp.png'))
        os.remove('tmp.png')

    def calculate_snr(self):
        orig_img_path = self.origImagePathLine.text()
        
        if not orig_img_path:
            QtWidgets.QMessageBox.about(self, "Original Image Required", "Please upload original image.")
            return

        if not self.img_recon_flag:
            QtWidgets.QMessageBox.about(self, "Reconstructed Image Required", "Please reconstruct cs image first.")
            return

        orig_img = cv2.imread(orig_img_path)
        snr_val = snr(orig_img, self.img_recon)

        self.snrLCDNumber.display(float(snr_val))

    def save(self):
        if not self.img_recon_flag:
            QtWidgets.QMessageBox.about(self, "Reconstructed Image Required", "No reconstruction to save.")
            return

        default = os.path.join(os.getcwd(), 'reconstructed_samples/')
        save_dir = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", default))
        cs_measurements_path = self.csMeasurementsPathLine.text()
        cs_ratio = int(self.csRatiosComboBox.currentText())
        orig_name, _ = os.path.splitext(ntpath.basename(cs_measurements_path))

        filepath = '%s/%s_recon_%s.png' % (save_dir, orig_name, cs_ratio)
        
        cv2.imwrite(filepath, self.img_recon)

    def select_cs(self):
        default = os.path.join(os.getcwd(), 'cs_test_samples/')
        self.cs_measurements_path, ext = QtWidgets.QFileDialog.getOpenFileName(self, 'Select CS Image measurements.', default)
        self.csMeasurementsPathLine.setText(self.cs_measurements_path)
    
    def select_phi(self):
        default = os.path.join(os.getcwd(), 'image_processing/ista/sampling_matrix/')
        self.phi_path, ext = QtWidgets.QFileDialog.getOpenFileName(self, 'Select sampling matrix.', default)
        self.samplingMatrixPathLine.setText(self.phi_path)

    def select_qinit(self):
        default = os.path.join(os.getcwd(), 'image_processing/ista/initialization_matrix/')
        self.qinit_path, ext = QtWidgets.QFileDialog.getOpenFileName(self, 'Select initialization matrix.', default)
        self.initializationMatrixPathLine.setText(self.qinit_path)

    def select_orig_image(self):
        default = os.path.join(os.getcwd(), 'data/')
        self.orig_img_path, ext = QtWidgets.QFileDialog.getOpenFileName(self, 'Select original image.', default)
        self.origImagePathLine.setText(self.orig_img_path)



class NetMDSettings(settings.Ui_SettingsWindow, QtWidgets.QMainWindow):
    
    def __init__(self, parent=None):

        super(NetMDSettings, self).__init__(parent)
        self.setupUi(self)
        self.setMinimumHeight(750)
        self.setMinimumWidth(1250)
        self.actionHome.triggered.connect(self.home)
        self.actionInstructions.triggered.connect(self.instructions)

        # Buttons
        self.istaToolButton.clicked.connect(self.select_ista_model_dir)
        self.rcmddToolButton.clicked.connect(self.select_rcmdd_model_dir)

        self.saveSettingsButton.clicked.connect(self.saveSettings)

        # Defaults
        self.ista_models_dir = os.path.join(os.getcwd(), 'image_processing/ista/model/RCM/')
        self.istaPathLine.setText(self.ista_models_dir)

        self.rcmdd_models_dir = os.path.join(os.getcwd(), 'image_processing/rcmdd/model/RCM/')
        self.rcmddPathLine.setText(os.path.join(os.getcwd(), 'image_processing/rcmdd/model/RCM/'))

    def select_ista_model_dir(self):
        default = os.path.join(os.getcwd(), 'image_processing/ista/model/')
        self.ista_models_dir = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select ISTA-Net Models Directory", default))
        self.istaPathLine.setText(self.ista_models_dir)

    def select_rcmdd_model_dir(self):
        default = os.path.join(os.getcwd(), 'image_processing/rcmdd/model/')
        self.rcmdd_models_dir = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select RCMDD Models Directory", default))
        self.rcmddPathLine.setText(self.rcmdd_models_dir)

    def saveSettings(self):
        self.parent().home.ista_models_dir = self.ista_models_dir
        self.parent().home.rcmdd_models_dir = self.rcmdd_models_dir

    def instructions(self):
        self.hide()
        self.parent().show()

    def home(self):
        self.hide()
        self.parent().get_started()
        

if __name__ == '__main__':
    parser = ArgumentParser(description='Train ISTA-Net-plus')
    parser.add_argument('--denoiser', type=str, default='on', help='set denoiser on/off')
    args = parser.parse_args()

    app = QtWidgets.QApplication()
    qt_app = NetMDInstructions(args)
    qt_app.show()
    app.exec_()
