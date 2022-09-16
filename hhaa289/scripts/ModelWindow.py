import sys

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton

import numpy as np
import matplotlib.pyplot as plt

import torchvision.models as models

class ModelWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Select Model")
        self.setGeometry(300, 300, 300, 200)

        self.Model1 = QPushButton(self)
        self.Model1.setText('AlexNet Model')
        self.Model1.clicked.connect(self.modelSelected)

        self.Model2 = QPushButton(self)
        self.Model2.setText('VGG-16 Model')
        self.Model2.clicked.connect(self.modelSelected)

        self.Model3 = QPushButton(self)
        self.Model3.setText('Inception-v3 Model')
        self.Model3.clicked.connect(self.modelSelected)

        self.Model4 = QPushButton(self)
        self.Model4.setText('ResNet Model')
        self.Model4.clicked.connect(self.modelSelected)

        self.Model5 = QPushButton(self)
        self.Model5.setText('ResNeXt Model')
        self.Model5.clicked.connect(self.modelSelected)

        self.Model6 = QPushButton(self)
        self.Model6.setText('GoogleNet Model')
        self.Model6.clicked.connect(self.modelSelected)

        vbox = QVBoxLayout()
        vbox.addWidget(self.Model1)
        vbox.addWidget(self.Model2)
        vbox.addWidget(self.Model3)
        vbox.addWidget(self.Model4)
        vbox.addWidget(self.Model5)
        vbox.addWidget(self.Model6)
        self.setLayout(vbox)

    def modelSelected(self):
        if self.Model1:
            alexnet = models.alexnet()
            self.hide() #This closes the ModelSelect window
        
        if self.Model2:
            vgg16 = models.vgg16()
            self.hide() #This closes the ModelSelect window

        if self.Model3:
            inception = models.inception_v3()
            self.hide() #This closes the ModelSelect window
        
        if self.Model4:
            resnet18 = models.resnet18()
            self.hide() #This closes the ModelSelect window
        
        if self.Model5:
            resnext50_32x4d = models.resnext50_32x4d()
            self.hide() #This closes the ModelSelect window
        
        if self.Model6:
            googlenet = models.googlenet()
            self.hide() #This closes the ModelSelect window