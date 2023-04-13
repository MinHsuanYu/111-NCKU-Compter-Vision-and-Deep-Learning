from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
#Machine Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models
from torchsummary import summary
import torchvision.transforms as transforms
#visualize
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(230, 200)
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 210, 180))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(30, 20, 150, 25))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 50, 150, 25))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 80, 150, 25))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 110, 150, 25))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_5.setGeometry(QtCore.QRect(30, 140, 150, 25))
        self.pushButton_5.setObjectName("pushButton_5")

        self.pushButton.clicked.connect(self.ShowTrainImg)
        self.pushButton_2.clicked.connect(self.ShowParameter)
        self.pushButton_3.clicked.connect(self.ShowModel)
        self.pushButton_4.clicked.connect(self.ShowAccuracy)
        self.pushButton_5.clicked.connect(self.TestImg)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.show()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox.setTitle(_translate("Dialog", "GroupBox"))
        self.pushButton.setText(_translate("Dialog", "1.Show Train Images"))
        self.pushButton_2.setText(_translate("Dialog", "2. Show HyperParameter"))
        self.pushButton_3.setText(_translate("Dialog", "3. Show Model Shortcut"))
        self.pushButton_4.setText(_translate("Dialog", "4. Show Accuracy"))
        self.pushButton_5.setText(_translate("Dialog", "5. Test"))

    def ShowTrainImg(self):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        plt.figure(figsize=(9, 9))
        for i in range(9):
            index = random.randint(0, 50000)
            plt.subplot(3, 3, i+1)
            plt.title(trainset.classes[trainset.targets[index]])
            plt.imshow(trainset.data[index])
            plt.axis('off')
        plt.show()
    
    def ShowParameter(self):
        print("hyperparameters:")
        print("batch size:", 32)
        print("learning rate:", 0.001)
        print("optimizer:", "SGD")
        
    def ShowModel(self):
        tmp = torch.load('model.pt')
        tmp = tmp.cuda()
        print(summary(tmp, (3, 32, 32)))

    def ShowAccuracy(self):
        acc = cv2.imread("Accuracy.png", cv2.IMREAD_COLOR)
        cv2.imshow("Accuracy", acc)
        loss = cv2.imread("Loss.png", cv2.IMREAD_COLOR)
        cv2.imshow("Loss", loss)

    def TestImg(self):
        choice = 23

        model = torch.load("trained.pt")
        model = model.cuda()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        input = np.array(testset.data[choice])
        plt.figure()
        plt.imshow(input)
        plt.show()
        input = input/255
        input_tensor = torch.from_numpy(input)
        input_tensor = input_tensor.permute(2, 0, 1)
        input_tensor = input_tensor.unsqueeze_(0).to('cuda')
        input_tensor = input_tensor.float()
        output = model(input_tensor)
        tmp = torch.softmax(output[0], 0)
        plt.figure()
        plt.bar(testset.classes[:], tmp.cpu().detach().numpy())
        plt.show()