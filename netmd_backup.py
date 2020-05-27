import os
import cv2
import numpy as np
import scipy.io as sio
from PySide2 import QtWidgets, QtCore, QtGui
from ui import instructions, home, settings
from image_processing.ista.ista_net import ISTA_Net



class NetMDInstructions(instructions.Ui_InstructionsWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super(NetMDInstructions, self).__init__()
        self.setupUi(self)
        self.setMinimumHeight(750)
        self.setMinimumWidth(1250)
        self.startButton.clicked.connect(self.get_started)

    def get_started(self):
        self.hide()
        app = NetMDHome(self)
        app.show()

    def settings_back(self):
        self.hide()
        settings = NetMDSettings(self)
        settings.show()


class NetMDHome(home.Ui_HomeWindow, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(NetMDHome, self).__init__(parent)
        self.setupUi(self)
        self.setMinimumHeight(750)
        self.setMinimumWidth(1250)
        self.actionSettings.triggered.connect(self.settings)
        self.actionIntructions.triggered.connect(self.instructions)

        # Upload buttons
        self.csMeasurementsButton.clicked.connect(self.select_cs)
        self.samplingMatrixButton.clicked.connect(self.select_phi)
        self.initializationMatrixButton.clicked.connect(self.select_qinit)

        self.reconstructButton.clicked.connect(self.reconstruct)

        # Defaults
        self.csMeasurementsPathLine.setText(os.path.join(os.getcwd(), 'cs_test_samples/Natural_Images/cs_25/barbara_cs.mat'))
        self.samplingMatrixPathLine.setText(os.path.join(os.getcwd(), 'image_processing/ista/sampling_matrix/phi_0_25_1089.mat'))
        self.initializationMatrixPathLine.setText(os.path.join(os.getcwd(), 'image_processing/ista/initialization_matrix/Natural_Images/Initialization_Matrix_25.mat'))

        self.csRatiosComboBox.setCurrentIndex(1)

        # ISTA-Net+RCMDD Variables
        self.cs_measurements_path = None
        self.phi_path = None
        self.qinit_path = None
        self.model_path = 
        
        self.ista_net = ISTA_Net()
        

    def instructions(self):
        self.hide()
        self.parent().show()
        self.close()

    def settings(self):
        self.hide()
        self.parent().settings_back()

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

        _, cs_file_ext = os.path.splitext(cs_measurements_path)
        _, phi_file_ext = os.path.splitext(phi_path)
        _, qinit_file_ext = os.path.splitext(qinit_path)

        if cs_file_ext != '.mat' or phi_file_ext != '.mat' or qinit_file_ext != '.mat':
            QtWidgets.QMessageBox.about(self, "File Extension Error", "Files must be in .mat format.")

        print(cs_measurements_path)
        print(phi_path)
        print(qinit_path)

        cs_measurements = np.array(sio.loadmat(cs_measurements_path)['cs_measurements'])
        cs_ratio = int(self.csRatiosComboBox.currentText())

        self.ista_net.load_phi(phi_path)
        self.ista_net.load_qinit(qinit_path)
        self.ista_net.load_model(os.path.join(os.getcwd(), 'image_processing/ista/model/Natural_Images/CS_ISTA_Net_plus_layer_9_ratio_%d_lr_0.0001/net_params_200.pkl' % (cs_ratio)))
        self.img_recon = self.ista_net.reconstruct(cs_measurements, 256, 256)

        cv2.imwrite('tmp.png', self.img_recon)
        #self.image = QtGui.QImage(self.img_recon, self.img_recon.shape[0], self.img_recon.shape[1], QtGui.QImage.Format_Grayscale8)
        self.recoveredImageLabel.setPixmap(QtGui.QPixmap('tmp.png')) 

    def select_cs(self):
        self.cs_measurements_path, ext = QtWidgets.QFileDialog.getOpenFileName(self, 'Select CS Image measurements.')
        self.csMeasurementsPathLine.setText(self.cs_measurements_path)
    
    def select_phi(self):
        self.phi_path, ext = QtWidgets.QFileDialog.getOpenFileName(self, 'Select sampling matrix.')
        self.samplingMatrixPathLine.setText(self.phi_path)

    def select_qinit(self):
        self.qinit_path, ext = QtWidgets.QFileDialog.getOpenFileName(self, 'Select initialization matrix.')
        self.initializationMatrixPathLine.setText(self.qinit_path)

    def preview_img_recon(self):
        self.recoveredImageLabel.setPixmap(QtGui.QtQPixmap("python.jpg"))



class NetMDSettings(settings.Ui_SettingsWindow, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(NetMDSettings, self).__init__(parent)
        self.setupUi(self)
        self.setMinimumHeight(750)
        self.setMinimumWidth(1250)
        self.actionHome.triggered.connect(self.home)
        self.actionInstructions.triggered.connect(self.instructions)


    def instructions(self):
        self.hide()
        self.parent().show()
        self.close()

    def home(self):
        self.hide()
        self.parent().get_started()
        



if __name__ == '__main__':
    app = QtWidgets.QApplication()
    qt_app = NetMDInstructions()
    qt_app.show()
    app.exec_()
