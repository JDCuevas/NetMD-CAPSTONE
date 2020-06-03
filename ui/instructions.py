# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'instructions.ui'
##
## Created by: Qt User Interface Compiler version 5.14.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *

import icons_rc

class Ui_InstructionsWindow(object):
    def setupUi(self, InstructionsWindow):
        if not InstructionsWindow.objectName():
            InstructionsWindow.setObjectName(u"InstructionsWindow")
        InstructionsWindow.resize(1076, 1191)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(InstructionsWindow.sizePolicy().hasHeightForWidth())
        InstructionsWindow.setSizePolicy(sizePolicy)
        InstructionsWindow.setMaximumSize(QSize(2000000, 1999998))
        InstructionsWindow.setBaseSize(QSize(0, 800))
        self.centralwidget = QWidget(InstructionsWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.gridLayout_2 = QGridLayout(self.frame)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setSpacing(7)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.instructions = QPlainTextEdit(self.frame)
        self.instructions.setObjectName(u"instructions")
        self.instructions.setMinimumSize(QSize(100, 100))
        self.instructions.setReadOnly(True)

        self.verticalLayout.addWidget(self.instructions)

        self.startButton = QPushButton(self.frame)
        self.startButton.setObjectName(u"startButton")
        self.startButton.setMinimumSize(QSize(20, 20))
        self.startButton.setStyleSheet(u"color: rgb(85, 170, 255);\n"
"")

        self.verticalLayout.addWidget(self.startButton)


        self.gridLayout_2.addLayout(self.verticalLayout, 1, 0, 1, 1)

        self.logoLabel = QLabel(self.frame)
        self.logoLabel.setObjectName(u"logoLabel")
        sizePolicy1 = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.logoLabel.sizePolicy().hasHeightForWidth())
        self.logoLabel.setSizePolicy(sizePolicy1)
        self.logoLabel.setMinimumSize(QSize(100, 100))
        self.logoLabel.setMaximumSize(QSize(200, 200))
        self.logoLabel.setBaseSize(QSize(50, 0))
#if QT_CONFIG(accessibility)
        self.logoLabel.setAccessibleDescription(u"")
#endif // QT_CONFIG(accessibility)
        self.logoLabel.setPixmap(QPixmap(u":/images/images/NetMD_logo.png"))
        self.logoLabel.setScaledContents(True)

        self.gridLayout_2.addWidget(self.logoLabel, 0, 0, 1, 1)


        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        InstructionsWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(InstructionsWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1076, 22))
        InstructionsWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(InstructionsWindow)
        self.statusbar.setObjectName(u"statusbar")
        InstructionsWindow.setStatusBar(self.statusbar)

        self.retranslateUi(InstructionsWindow)

        QMetaObject.connectSlotsByName(InstructionsWindow)
    # setupUi

    def retranslateUi(self, InstructionsWindow):
        InstructionsWindow.setWindowTitle(QCoreApplication.translate("InstructionsWindow", u"Instructions", None))
        self.instructions.setPlainText(QCoreApplication.translate("InstructionsWindow", u"Welcome!\n"
"\n"
"	NetMD is an easy-to-use, easy-to-learn tool for recovery of compressively sampled (CS) Reflectance Confocal Microscopy (RCM) images of skin. \n"
"\n"
"\n"
"Instructions:\n"
"\n"
"1- Upload to NetMD your compessed samples, alongside the sampling matrix used to collect the images.\n"
"\n"
"2- Specify the cs ratio used to collect the images.\n"
"\n"
"3- Indicate the expected width and height of the output image.\n"
"\n"
"4- Reconstruct the image using the neural network backend!\n"
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
"	If you've trained alternate models using the train.py files located inside the program's ISTA-Net and RCMDD respective module folders, you can change the models the network is using in the settings page by selectin"
                        "g the folder where the models where saved. For example, the default ISTA-Net and RCMDD models for RCM image recovery are saved in the 'image_processing/ista/model/RCM/'  and 'image_processing/rcmdd/model/RCM/' directories respectively. NOTE: The models must be named the same as the ones provided with the program, where the only thing that changes is the CS ratio in the name.\n"
"\n"
"For more information on model training visit:\n"
"https://github.com/JDCuevas/NetMD-CAPSTONE", None))
        self.startButton.setText(QCoreApplication.translate("InstructionsWindow", u"Get Started", None))
        self.logoLabel.setText("")
    # retranslateUi

