# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/instructions.ui',
# licensing of 'ui/instructions.ui' applies.
#
# Created: Wed Jun  3 01:15:41 2020
#      by: pyside2-uic  running on PySide2 5.9.0~a1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_InstructionsWindow(object):
    def setupUi(self, InstructionsWindow):
        InstructionsWindow.setObjectName("InstructionsWindow")
        InstructionsWindow.resize(1076, 1191)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(InstructionsWindow.sizePolicy().hasHeightForWidth())
        InstructionsWindow.setSizePolicy(sizePolicy)
        InstructionsWindow.setMaximumSize(QtCore.QSize(2000000, 1999998))
        InstructionsWindow.setBaseSize(QtCore.QSize(0, 800))
        self.centralwidget = QtWidgets.QWidget(InstructionsWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(7)
        self.verticalLayout.setObjectName("verticalLayout")
        self.instructions = QtWidgets.QPlainTextEdit(self.frame)
        self.instructions.setMinimumSize(QtCore.QSize(100, 100))
        self.instructions.setReadOnly(True)
        self.instructions.setObjectName("instructions")
        self.verticalLayout.addWidget(self.instructions)
        self.startButton = QtWidgets.QPushButton(self.frame)
        self.startButton.setMinimumSize(QtCore.QSize(20, 20))
        self.startButton.setStyleSheet("color: rgb(85, 170, 255);\n"
"")
        self.startButton.setObjectName("startButton")
        self.verticalLayout.addWidget(self.startButton)
        self.gridLayout_2.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.logoLabel = QtWidgets.QLabel(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.logoLabel.sizePolicy().hasHeightForWidth())
        self.logoLabel.setSizePolicy(sizePolicy)
        self.logoLabel.setMinimumSize(QtCore.QSize(100, 100))
        self.logoLabel.setMaximumSize(QtCore.QSize(200, 200))
        self.logoLabel.setBaseSize(QtCore.QSize(50, 0))
        self.logoLabel.setAccessibleDescription("")
        self.logoLabel.setText("")
        self.logoLabel.setPixmap(QtGui.QPixmap(":/images/images/NetMD_logo.png"))
        self.logoLabel.setScaledContents(True)
        self.logoLabel.setObjectName("logoLabel")
        self.gridLayout_2.addWidget(self.logoLabel, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        InstructionsWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(InstructionsWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1076, 39))
        self.menubar.setObjectName("menubar")
        InstructionsWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(InstructionsWindow)
        self.statusbar.setObjectName("statusbar")
        InstructionsWindow.setStatusBar(self.statusbar)

        self.retranslateUi(InstructionsWindow)
        QtCore.QMetaObject.connectSlotsByName(InstructionsWindow)

    def retranslateUi(self, InstructionsWindow):
        InstructionsWindow.setWindowTitle(QtWidgets.QApplication.translate("InstructionsWindow", "Instructions", None, -1))
        self.instructions.setPlainText(QtWidgets.QApplication.translate("InstructionsWindow", "Welcome!\n"
"\n"
"    NetMD is an easy-to-use, easy-to-learn tool for recovery of compressively sampled (CS) Reflectance Confocal Microscopy (RCM) images of skin. \n"
"\n"
"\n"
"Instructions:\n"
"\n"
"1- Upload your compessed sample.\n"
"\n"
"2- Upload the sampling matrix used to collect the images. \n"
"\n"
"    - Expected to be a .mat file with an array accessed through key         \n"
"    \'cs_measurements\'\n"
"\n"
"    - Array must have shape:\n"
"         [num_img_blocks, 1089 * cs sampling ratio]\n"
"    where cs sampling ratio is 10, 25 or 50 for 10%, 25% or 50%.\n"
"\n"
"3 - Upload the initialization matrix for the initial guess, used during network training. (Some provided in \'image_processing/ista/initialization_matrix/\')\n"
"\n"
"4- Indicate the expected width and height of the output image.\n"
"\n"
"5- Reconstruct the image using the neural network backend!\n"
"\n"
"\n"
"Optional:\n"
"\n"
"- Calculate Singal-to-Noise Ratio (SNR) -\n"
"\n"
"1- Reconstruct the image from the cs measurements.\n"
"\n"
"2- Upload the original full-sized image if available.\n"
"\n"
"3- Calculate the Signal-to-Noise Ratio (SNR).\n"
"\n"
"\n"
"- Change Neural Network Models - \n"
"\n"
"    If you\'ve trained alternate models using the train.py files located inside the program\'s ISTA-Net and RCMDD respective module folders, you can change the models the network is using in the settings page by selecting the folder where the models where saved. For example, the default ISTA-Net and RCMDD models for RCM image recovery are saved in the \'image_processing/ista/model/RCM/\'  and \'image_processing/rcmdd/model/RCM/\' directories respectively. NOTE: The models must be named the same as the ones provided with the program, where the only thing that changes is the CS ratio in the name.\n"
"\n"
"For more information on model training visit:\n"
"https://github.com/JDCuevas/NetMD-CAPSTONE", None, -1))
        self.startButton.setText(QtWidgets.QApplication.translate("InstructionsWindow", "Get Started", None, -1))

import ui.icons_rc
