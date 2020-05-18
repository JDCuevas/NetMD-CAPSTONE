from PySide2 import QtWidgets, QtCore
from ui import instructions, home, settings



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

        # ISTA-Net+RCMDD Variables
        self.cs_measurements_path = None
        self.phi_path = None
        self.qinit_path = None
        

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

    def select_cs(self):
        self.cs_measurements_path, ext = QtWidgets.QFileDialog.getOpenFileName(self, 'Select CS Image measurements.')
        self.csMeasurementsPathLine.setText(self.cs_measurements_path)
    
    def select_phi(self):
        self.phi_path, ext = QtWidgets.QFileDialog.getOpenFileName(self, 'Select sampling matrix.')
        self.samplingMatrixPathLine.setText(self.phi_path)

    def select_qinit(self):
        self.qinit_path, ext = QtWidgets.QFileDialog.getOpenFileName(self, 'Select initialization matrix.')
        self.initializationMatrixPathLine.setText(self.qinit_path)



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
