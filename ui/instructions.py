# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/instructions.ui',
# licensing of 'ui/instructions.ui' applies.
#
# Created: Mon May 18 13:53:11 2020
#      by: pyside2-uic  running on PySide2 5.9.0~a1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets
import icons_rc

class Ui_InstructionsWindow(object):
    def setupUi(self, InstructionsWindow):
        InstructionsWindow.setObjectName("InstructionsWindow")
        InstructionsWindow.resize(727, 854)
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
        self.menubar.setGeometry(QtCore.QRect(0, 0, 727, 39))
        self.menubar.setObjectName("menubar")
        InstructionsWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(InstructionsWindow)
        self.statusbar.setObjectName("statusbar")
        InstructionsWindow.setStatusBar(self.statusbar)

        self.retranslateUi(InstructionsWindow)
        QtCore.QMetaObject.connectSlotsByName(InstructionsWindow)

    def retranslateUi(self, InstructionsWindow):
        InstructionsWindow.setWindowTitle(QtWidgets.QApplication.translate("InstructionsWindow", "Instructions", None, -1))
        self.instructions.setPlainText(QtWidgets.QApplication.translate("InstructionsWindow", "Instructions:\n"
"", None, -1))
        self.startButton.setText(QtWidgets.QApplication.translate("InstructionsWindow", "Get Started", None, -1))
