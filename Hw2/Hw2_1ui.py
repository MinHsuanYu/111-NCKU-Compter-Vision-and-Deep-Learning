import cv2
import numpy as np
import os

import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets

def hw2_2Init(input_path):
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)

    input_imgs = []
    output_imgs = []
    objpoints = []
    imgpoints = []
    for i in range(1, 16):
        path = os.path.join(input_path, str(i) + '.bmp')
        input_imgs.append(cv2.imread(path))
    for img in input_imgs:
        flag, cns = cv2.findChessboardCorners(img, (8, 11))
        out_img = img.copy()

        if flag:
            objpoints.append(objp)
            imgpoints.append(cns)

            cv2.drawChessboardCorners(out_img, (8, 11), cns, flag)
            output_imgs.append(out_img)
        else:
            print("Error!")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
    return input_imgs, output_imgs, mtx, rvecs, tvecs, dist

class Ui_Dialog(object):
    def __init__(self):
        super().__init__()
        input_path = "Dataset_OpenCvDl_Hw2/Q2_Image"
        self.inputs, self.outputs, self.intrinsic, self.rvecs, self.tvecs, self.dist = hw2_2Init(input_path) 

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(250, 300)
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(20, 30, 210, 240))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(30, 20, 150, 25))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 50, 150, 25))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 170, 150, 25))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_5.setGeometry(QtCore.QRect(30, 200, 150, 25))
        self.pushButton_5.setObjectName("pushButton_5")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_3.setGeometry(QtCore.QRect(30, 80, 150, 80))
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 45, 100, 25))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setGeometry(QtCore.QRect(20, 20, 65, 25))
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit.setGeometry(QtCore.QRect(80, 20, 60, 20))
        self.lineEdit.setObjectName("lineEdit")

        self.pushButton.clicked.connect(self.ShowCorner)
        self.pushButton_2.clicked.connect(self.ShowIntrinsic)
        self.pushButton_3.clicked.connect(self.ShowExtrinsic)
        self.pushButton_4.clicked.connect(self.ShowDistortion)
        self.pushButton_5.clicked.connect(self.ShowUndistortion)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.show()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox.setTitle(_translate("Dialog", "2. Calibration"))
        self.pushButton.setText(_translate("Dialog", "2.1 Find Corners"))
        self.pushButton_2.setText(_translate("Dialog", "2.2 Find intrinsic"))
        self.pushButton_4.setText(_translate("Dialog", "2.4 Dinf Distortion"))
        self.pushButton_5.setText(_translate("Dialog", "2.5 Show result"))
        self.groupBox_3.setTitle(_translate("Dialog", "2.3 Find Extrinsic"))
        self.pushButton_3.setText(_translate("Dialog", "2.3 Find Extrinsic"))
        self.label.setText(_translate("Dialog", "Selct image:"))

    def ShowCorner(self):
        cv2.namedWindow("Corner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Corner", (1024, 1024))
        for img in self.outputs:
            cv2.imshow("Corner", img)
            cv2.waitKey(500)
    
    def ShowIntrinsic(self):
        print("Intrinsic:")
        print(self.intrinsic)

    def ShowExtrinsic(self):
        choice = self.lineEdit.text()
        choice = int(choice)
        rmatrix, _ = cv2.Rodrigues(self.rvecs[choice])
        extrinsic = np.c_[rmatrix, self.tvecs[choice]]
        print("Extrinsic:")
        print(extrinsic)
    
    def ShowDistortion(self):
        print("Distortion:")
        print(self.dist)

    def ShowUndistortion(self):
        cv2.namedWindow("Undistortion", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Undistortion", (1024, 512))
        undis_img = []
        for img in self.inputs:
            tmp = img.copy()
            stack_tmp = np.hstack([tmp, cv2.undistort(tmp, self.intrinsic, self.dist)])
            undis_img.append(stack_tmp)
        cv2.imshow("Undistortion", undis_img[0])