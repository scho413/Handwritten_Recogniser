import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, qApp, QWidget, QPushButton, 
QHBoxLayout, QVBoxLayout, QMessageBox, QLabel, QTextBrowser, QProgressBar)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn, optim

import matplotlib.pyplot as plt
import numpy as np
import os

from time import time

train_dataset = ""
test_dataset = ""

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.setWindowTitle('Handwritten Digit Recognizer')
        self.setGeometry(1000, 500, 800, 500)  

        train_model_action = QAction('Train Model', self)
        train_model_action.setShortcut('Ctrl+M')
        train_model_action.setStatusTip('Train model')
        train_model_action.triggered.connect(self.openTrainModel)
        exit_action = QAction('Quit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(qApp.quit)

        view_train_images = QAction('View Train Images', self)
        view_train_images.setShortcut('Ctrl+R')
        view_train_images.setStatusTip('View Train Images')
        # view_train_images.triggered.connect(self.function)

        view_testing_images = QAction('View Testing Images', self)
        view_testing_images.setShortcut('Ctrl+E')
        view_testing_images.setStatusTip('View Testing Images')
        # view_testing_images.triggered.connect(self.function)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(train_model_action)
        file_menu.addSeparator
        file_menu.addAction(exit_action)
        view_menu = menu_bar.addMenu('&View')
        view_menu.addAction(view_train_images)
        view_menu.addAction(view_testing_images)

        self.show()
    
    def openTrainModel(self):
        self.open_train_model = trainModel()
        self.open_train_model.show()

class trainModel(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.setWindowTitle('Train Model')
        self.setGeometry(1050, 550, 700, 400)

        title = QLabel('Train Model Status', self)
        title.setAlignment(Qt.AlignCenter)

        self.status = QTextBrowser()

        self.progress = QProgressBar(self)
        self.progress.setMaximum(100)
        self.progress.setGeometry(200, 80, 500, 20)
        
        download = QPushButton('Download')
        download.setToolTip('Download MNIST datasets')
        download.clicked.connect(self.downloadClicked)
        train = QPushButton('Train')
        train.setToolTip('Train Model')
        train.clicked.connect(self.trainClicked)
        clear = QPushButton('Clear')
        clear.setToolTip('Clear the status message')
        clear.clicked.connect(self.clearClicked)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(download)
        hbox.addWidget(train)
        hbox.addWidget(clear)
        hbox.addStretch(1)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addWidget(title)
        vbox.addStretch(1)
        vbox.addWidget(self.status)
        vbox.addStretch(1)
        vbox.addWidget(self.progress)
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addStretch(1)

        self.setLayout(vbox)
    
    def downloadClicked(self):
        global train_dataset, test_dataset, progress_count

        progress_count = 0

        train_dataset = datasets.MNIST(root='C:/Users/jdfm0/Desktop/Part 3/COMPSYS 302/Part 1 python/project/mnist_data_train/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)

        if (train_dataset != None):
            self.status.append('Downloading train dataset...')
            while progress_count < 50:
                progress_count += 0.00005
                self.progress.setValue(progress_count)
            self.status.append('Finished downloading test dataset!')
                
        test_dataset = datasets.MNIST(root='C:/Users/jdfm0/Desktop/Part 3/COMPSYS 302/Part 1 python/project/mnist_data_test/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=False)

        if (test_dataset != None):
            self.status.append('Downloading test dataset...')
            while progress_count < 100:
                progress_count += 0.00005
                self.progress.setValue(progress_count)
            self.status.append('Finished downloading test dataset!')
        
    def trainClicked(self):

        if (train_dataset != "" and test_dataset != ""):
            progress_count = 0
            self.progress.setValue(progress_count)

            self.status.append('Training model...')

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=False)

            pixel_size = 784
            processing_layers = [128, 64]
            output_size = 10
            learning_rate = 0.001

            model = nn.Sequential(nn.Linear(pixel_size, processing_layers[0]),
                                nn.ReLU(),
                                nn.Linear(processing_layers[0], processing_layers[1]),
                                nn.ReLU(),
                                nn.Linear(processing_layers[1], output_size),
                                nn.LogSoftmax(dim=1))

            criterion = nn.NLLLoss()
            image, label = next(iter(train_loader))
            image = image.view(image.shape[0], -1)

            log_probability = model(image)
            loss = criterion(log_probability, label)

            optimiser = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

            epochs = 5

            for i in range(epochs):
                losses = 0
                for image, label in train_loader:

                    image_size = image.shape[0]
                    image = image.view(image_size, -1)

                    optimiser.zero_grad()
                    logits = model(image)
                    loss = criterion(logits, label)
                    loss.backward()
                    optimiser.step()

                    losses += loss.item()

                else:
                    while progress_count < ((i*10) + 10):
                        progress_count += 0.00005
                        self.progress.setValue(progress_count)
                    self.status.append("Epoch {} - Training loss: {}".format(i, losses/len(train_loader)))
            self.status.append('Finished training datasets!')

            correct_count = 0
            total_test_datasets = 0
            self.status.append('Calculating accuracy...')
            for image,label in test_loader:
                for i in range(len(label)):
                    img = image[i].view(1, 784)
                    with torch.no_grad():
                        log_probability = model(img)
                    
                    prediction = torch.exp(log_probability)
                    probability = list(prediction.numpy()[0])
                    predicted_label = probability.index(max(probability))
                    original_label = label.numpy()[i]
                    if(original_label == predicted_label):
                        correct_count += 1
                    total_test_datasets += 1
            while progress_count < 100:
                progress_count += 0.00005
                self.progress.setValue(progress_count)
                    
            self.status.append('Number Of Images Tested = {}'.format(total_test_datasets))
            self.status.append('Model Accuracy = {}%'.format((correct_count/total_test_datasets)*100))

            torch.save(model, 'C:/Users/jdfm0/Desktop/Part 3/COMPSYS 302/Part 1 python/project/train_model.pt')
        else:
            self.status.append("Please download the MNIST datasets before training")


    def clearClicked(self):
        self.status.clear()
        progress_count = 0
        self.progress.setValue(progress_count)

if __name__ == '__main__':
   app = QApplication(sys.argv)
   window = MyApp()
   sys.exit(app.exec_())