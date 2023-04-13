import cv2
import numpy as np
import tkinter as tk
import math

from numpy.lib.shape_base import column_stack
import sklearn.preprocessing
from tkinter import Tk, filedialog
from PyQt5 import QtCore, QtGui, QtWidgets
from numpy.lib.type_check import imag

class Ui_Dialog(object):
    def __init__(self):
        super().__init__()
        self.img = None
        self.img2 = None

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(840, 360)
        self.groupBox_2 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_2.setGeometry(QtCore.QRect(220, 20, 180, 320))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_6.setGeometry(QtCore.QRect(20, 160, 140, 25))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_5.setGeometry(QtCore.QRect(20, 80, 140, 25))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_7.setGeometry(QtCore.QRect(20, 240, 140, 25))
        self.pushButton_7.setObjectName("pushButton_7")
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 180, 320))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(20, 40, 140, 25))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 120, 140, 25))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 200, 140, 25))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(20, 280, 140, 25))
        self.pushButton_4.setObjectName("pushButton_4")
        self.groupBox_4 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_4.setGeometry(QtCore.QRect(620, 20, 180, 320))
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_12 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_12.setGeometry(QtCore.QRect(20, 40, 140, 25))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_13 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_13.setGeometry(QtCore.QRect(20, 120, 140, 25))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_14 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_14.setGeometry(QtCore.QRect(20, 200, 140, 25))
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_15 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_15.setGeometry(QtCore.QRect(20, 280, 140, 25))
        self.pushButton_15.setObjectName("pushButton_15")
        self.groupBox_3 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_3.setGeometry(QtCore.QRect(420, 20, 180, 320))
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_10 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_10.setGeometry(QtCore.QRect(20, 200, 140, 25))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_9.setGeometry(QtCore.QRect(20, 120, 140, 25))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_11.setGeometry(QtCore.QRect(20, 280, 140, 25))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_8.setGeometry(QtCore.QRect(20, 40, 140, 25))
        self.pushButton_8.setObjectName("pushButton_8")

        self.pushButton.clicked.connect(self.LoadImg)
        self.pushButton_2.clicked.connect(self.ColorSep)
        self.pushButton_3.clicked.connect(self.ColorTrans)
        self.pushButton_4.clicked.connect(self.Blending)
        self.pushButton_5.clicked.connect(self.GaussianBlur_Smooth)
        self.pushButton_6.clicked.connect(self.BilateralFilter)
        self.pushButton_7.clicked.connect(self.MedianFilter)
        self.pushButton_8.clicked.connect(self.GaussianBlur_Edge)
        self.pushButton_9.clicked.connect(self.SobelX)
        self.pushButton_10.clicked.connect(self.SobelY)
        self.pushButton_11.clicked.connect(self.Magnitude)
        self.pushButton_12.clicked.connect(self.Resize)
        self.pushButton_13.clicked.connect(self.Translation)
        self.pushButton_14.clicked.connect(self.Rotation_Scaling)
        self.pushButton_15.clicked.connect(self.Shearing)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.show()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox_2.setTitle(_translate("Dialog", "2. Image Smoothing"))
        self.pushButton_6.setText(_translate("Dialog", "2.2 Bilateral Filter"))
        self.pushButton_5.setText(_translate("Dialog", "2.1 Gaussian Blur"))
        self.pushButton_7.setText(_translate("Dialog", "2.3 Median Filter"))
        self.groupBox.setTitle(_translate("Dialog", "1. Image Processing"))
        self.pushButton.setText(_translate("Dialog", "1.1 Load Image"))
        self.pushButton_2.setText(_translate("Dialog", "1.2 Color Seperation"))
        self.pushButton_3.setText(_translate("Dialog", "1.3 Color Transformations"))
        self.pushButton_4.setText(_translate("Dialog", "1.4 Blending"))
        self.groupBox_4.setTitle(_translate("Dialog", "4. Transformation"))
        self.pushButton_12.setText(_translate("Dialog", "4.1 Resize"))
        self.pushButton_13.setText(_translate("Dialog", "4.2 Translation"))
        self.pushButton_14.setText(_translate("Dialog", "4.3 Rotation, Scaling"))
        self.pushButton_15.setText(_translate("Dialog", "4.4 Shearing"))
        self.groupBox_3.setTitle(_translate("Dialog", "3. Edge Detection"))
        self.pushButton_10.setText(_translate("Dialog", "3.3 Sobel Y"))
        self.pushButton_9.setText(_translate("Dialog", "3.2 Sobel X"))
        self.pushButton_11.setText(_translate("Dialog", "3.4 Magnitude"))
        self.pushButton_8.setText(_translate("Dialog", "3.1 Gaussian Blur"))
    
    def imread(self):
        root = tk.Tk()
        root.withdraw()
        root.update()
        file_path = filedialog.askopenfilename()
        img = cv2.imread(file_path)
        root.destroy()
        return img

    def LoadImg(self):
        self.img = self.imread()
        print("Height:", self.img.shape[0])
        print("Width:", self.img.shape[1])

    def ColorSep(self):
        zero = np.zeros(self.img.shape[:2], dtype=np.uint8)
        B, G, R = cv2.split(self.img)
        cv2.imshow("R", cv2.merge([zero, zero, R]))
        cv2.imshow("G", cv2.merge([zero, G, zero]))
        cv2.imshow("B", cv2.merge([B, zero, zero]))

    def ColorTrans(self):
        zero = np.zeros(self.img.shape[:2], dtype=np.uint16)
        B, G, R = np.array(cv2.split(self.img), dtype= np.uint16)
        zero = np.array((B+G+R)/3, dtype=np.uint8)
        tmp = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("cv_function", tmp)
        cv2.imshow("average", zero)

    def Blending(self):
        self.img2 = self.imread()
        cv2.namedWindow("Blending", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Blend", "Blending", 0, 255, self.BlendUpdate)
    def BlendUpdate(self, x):
        weight = cv2.getTrackbarPos("Blend", "Blending") / 255
        tmp = cv2.addWeighted(self.img, 1 - weight, self.img2, weight, 0)
        cv2.imshow("Blending", tmp)

    def GaussianBlur_Smooth(self):
        tmp = cv2.GaussianBlur(self.img, (5, 5), 0)
        cv2.imshow("GaussianBlur", tmp)

    def BilateralFilter(self):
        tmp = cv2.bilateralFilter(self.img, 9, 90 ,90)
        cv2.imshow("BilateralFilter", tmp)

    def MedianFilter(self):
        tmp = cv2.medianBlur(self.img, 3)
        cv2.imshow("MedianFilter_3*3", tmp)
        tmp = cv2.medianBlur(self.img, 5)
        cv2.imshow("MedianFilter_5*5", tmp)
    
    def GaussianBlur_Edge(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        Gaussian = np.zeros((3, 3))
        for i in range(0, 3):
            for j in range(0, 3):
                x2 = (i-1) * (i-1)
                y2 = (j-1) * (j-1)
                Gaussian[i, j] = math.exp(-(x2+y2))
        Gaussian = Gaussian / np.sum(np.sum(Gaussian))
        self.img2 = np.zeros(self.img.shape[:2], dtype=np.uint8)
        for x in range(1, self.img2.shape[0]-1):
            for y in range(1, self.img2.shape[1]-1):
                sum = 0 
                for i in range(0, 3):
                    for j in range(0, 3):
                        sum = sum + self.img[x+i-1][y+j-1] * Gaussian[i][j]
                sum = abs(sum)
                if sum >= 255:
                    sum = 255
                self.img2[x][y] = sum
        cv2.imshow("GaussianBlur", self.img2)

    def SobelX(self):
        sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        tmp = np.zeros(self.img.shape[:2], dtype=np.uint8)
        for x in range(1, tmp.shape[0]-1):
            for y in range(1, tmp.shape[1]-1):
                sum = 0 
                for i in range(0, 3):
                    for j in range(0, 3):
                        sum = sum + self.img2[x+i-1][y+j-1] * sobelX[i][j]
                sum = abs(sum)
                if sum >= 255:
                    sum = 255
                tmp[x][y] = sum
        cv2.imshow("SobelX", tmp)

    def SobelY(self):
        sobelY = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        tmp = np.zeros(self.img.shape[:2], dtype=np.uint8)
        for x in range(1, tmp.shape[0]-1):
            for y in range(1, tmp.shape[1]-1):
                sum = 0 
                for i in range(0, 3):
                    for j in range(0, 3):
                        sum = sum + self.img2[x+i-1][y+j-1] * sobelY[i][j]
                sum = abs(sum)
                if sum >= 255:
                    sum = 255
                tmp[x][y] = sum
        cv2.imshow("SobelY", tmp)

    def Magnitude(self):
        sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobelY = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        
        tmp = np.zeros(self.img.shape[:2], dtype=np.uint8) #doesn't normalize
        #tmp = np.zeros(self.img.shape[:2], dtype=np.float16)
        for x in range(1, tmp.shape[0]-1):
            for y in range(1, tmp.shape[1]-1):
                fx = 0
                fy = 0 
                for i in range(0, 3):
                    for j in range(0, 3):
                        fx = fx + self.img2[x+i-1][y+j-1] * sobelX[i][j]
                        fy = fy + self.img2[x+i-1][y+j-1] * sobelY[i][j]
                sum = math.sqrt(fx*fx + fy*fy)
                sum = min(255, sum)
                tmp[x][y] = sum
        #newImg = sklearn.preprocessing.normalize(tmp)
        #newImg = np.array(newImg*255, dtype=np.uint8)
        #cv2.imshow("Magnitude", newImg)
        cv2.imshow("Magnitude", tmp)
    
    def Resize(self):
        self.img = cv2.resize(self.img, (256, 256))
        cv2.imshow("Resize", self.img)

    def Translation(self):
        m = np.float32([[1, 0, 0], [0, 1, 60]])
        self.img = cv2.warpAffine(self.img, m, (400, 300))
        cv2.imshow("Translation", self.img)

    def Rotation_Scaling(self):
        m = cv2.getRotationMatrix2D((128, 188), 10, 0.5)
        self.img = cv2.warpAffine(self.img, m, (400, 300))
        cv2.imshow("Rotation_Scaling", self.img)
    
    def Shearing(self):
        before = np.float32([[50, 50], [200, 50], [50, 200]])
        after = np.float32([[10, 100], [200, 50], [100, 250]])
        m = cv2.getAffineTransform(before, after)
        self.img = cv2.warpAffine(self.img, m, (400, 300))
        cv2.imshow("Shearing", self.img)
