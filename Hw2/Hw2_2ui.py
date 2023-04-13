from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from torch.nn.modules import module
from torchsummary import summary


class Ui_Dialog(object):
    def __init__(self):
        super().__init__()
        self.model = torch.load("trained_No_pre.pt")

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(250, 300)
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(20, 30, 210, 240))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(30, 30, 150, 25))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 70, 150, 25))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 190, 150, 25))
        self.pushButton_4.setObjectName("pushButton_4")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(30, 150, 150, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 110, 150, 25))
        self.pushButton_3.setObjectName("pushButton_3")

        self.pushButton.clicked.connect(self.ShowModel)
        self.pushButton_2.clicked.connect(self.ShowTensorBoard)
        self.pushButton_3.clicked.connect(self.Inference)
        self.pushButton_4.clicked.connect(self.ShowDataAugumentation)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.show()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox.setTitle(_translate("Dialog", "5. Classification"))
        self.pushButton.setText(_translate("Dialog", "1. Show Model Structure"))
        self.pushButton_2.setText(_translate("Dialog", "2. Show TensorBoard"))
        self.pushButton_4.setText(_translate("Dialog", "4.Data Augmentation"))
        self.pushButton_3.setText(_translate("Dialog", "3. Test"))

    def ShowModel(self):
        summary(self.model, (3, 224, 224))

    def ShowTensorBoard(self):
        img = cv2.imread("TensorBoard.png")
        cv2.imshow("TensorBoard", img)

    def Inference(self):
        img_name = self.lineEdit.text()
        path = "PetImages/" + img_name + ".jpg"
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        tensor = []
        tensor.append(img)
        tensor = np.array(tensor, dtype=np.float32)
        tensor = torch.from_numpy(tensor).cuda()
        tensor = tensor.permute(0, 3, 1, 2)
        output = self.model(tensor)
        output = torch.sigmoid(output)
        _, predicted = torch.max(output.data, 1)
        title = ""
        if predicted == 0:
            title = "Dog"
        else:
            title = "Cat"
        plt.figure()
        plt.title(title)
        plt.imshow(img)
        plt.show()

    def ShowDataAugumentation(self):
        img_name = self.lineEdit.text()
        path = "PetImages/" + img_name + ".jpg"
        img1 = cv2.imread(path)
        img2 = img1.copy()
        img2 = cv2.flip(img2, 1)
        img = np.concatenate([img1, img2], axis=1)
        cv2.imshow("DataAugumentation", img)
        cmp = cv2.imread("compare.png")
        cv2.imshow("Compare", cmp)
