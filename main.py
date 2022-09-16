# sources:
# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, qApp, QWidget, QPushButton, QHBoxLayout, 
QVBoxLayout, QMessageBox, QLabel, QTextBrowser, QProgressBar, QGroupBox, QGridLayout, QComboBox, QLineEdit)
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt

import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn, optim

# import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL.ImageOps

from time import time

train_dataset = ""
test_dataset = ""

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.setWindowTitle('Handwritten Digit Recognizer')
        self.setGeometry(400, 150, 800, 500)  

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
        view_train_images.triggered.connect(self.openTrainingImages)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(train_model_action)
        file_menu.addSeparator
        file_menu.addAction(exit_action)
        view_menu = menu_bar.addMenu('&View')
        view_menu.addAction(view_train_images)

        # Change the layout of the App window
        # Allows use to add widgets onto QMainWindow even though it is already set
        widget = QWidget()
        grid = QGridLayout()
        grid.addWidget(self.ButtonsGroup(), 0, 1)
        grid.addWidget(self.CanvasGroup(), 0, 0)
        # grid.addWidget(self.predictDigit(), 1, 1)

        widget.setLayout(grid)
        self.setCentralWidget(widget)

        self.show()

    def CanvasGroup(self):
        groupboxCanvas = QGroupBox()

        x = None
        y = None

        self.setMouseTracking(True) #use to track mouse movement

        #create canvas to draw on
        self.canvas = QPixmap(600,400)
        self.canvas.fill(Qt.white)
        self.Display = QLabel()
        self.Display.setPixmap(self.canvas)

        #Layout
        vbox = QHBoxLayout() 
        vbox.addWidget(self.Display)
        groupboxCanvas.setLayout(vbox) 

        return groupboxCanvas

    def mouseMoveEvent(self, e):
        global img
        x = e.x()
        y = e.y()

        qp = QPainter(self.Display.pixmap())#draw on canvas
        qp.setPen(QPen(Qt.black, 10, Qt.SolidLine))#set pen colour
        qp.drawPoint(x,y)
        qp.end()
        self.update()

        # return img

    def ButtonsGroup(self):
        groupboxButtons = QGroupBox()

        self.ClearAction = QPushButton(self)
        self.ClearAction.setText('Clear')
        self.ClearAction.clicked.connect(self.clearClicked)

        self.Model = QComboBox(self)
        self.Model.addItem('Model')
        self.Model.addItem('Trained Model')
        self.Model.activated.connect(self.ModelClicked)
        
        self.Recognise = QPushButton(self)
        self.Recognise.setText('Recognise')
        self.Recognise.clicked.connect(self.RecogniseClicked)
        # self.Recognise.clicked.connect(self.predictDigit)

        self.text = QLabel()
        self.text.setText('Recognised Digit:')

        vbox = QVBoxLayout()
        vbox.addWidget(self.ClearAction)
        #vbox.addWidget(Random)
        vbox.addWidget(self.Model)
        vbox.addWidget(self.Recognise)
        vbox.addWidget(self.text)
        groupboxButtons.setLayout(vbox) 

        return groupboxButtons

    def clearClicked(self):
        # if self.ClearAction:
        self.Display.clear()
        self.Display.setPixmap(self.canvas)
        self.update()
    
    def ModelClicked(self):
        if self.Model.currentIndex() == 'Train':
            model = torch.load("train_model.pt")

    def RecogniseClicked(self):

        # save the updated canvas as an image
        self.img = self.Display.pixmap()
        self.img.save("canvas.png","PNG")

        self.image = Image.open("canvas.png")
        
        # convert to greyscale
        invert = PIL.ImageOps.invert(self.image)

        grey = invert.convert("L")

        # convert to 28x28 pixels - 784pixels
        resized = grey.resize((28,28))

        resized.save("pic.png", "PNG") 
        
        # resized.save("pic.png","PNG") #check after image is manipulated
        array = np.asarray(resized)
        # print(array.shape)

        # image = array/225
        image = array.reshape((1, 784))
        image = image/225

        # Convert to tensor to support our model input
        image_tensor = torch.FloatTensor(image)
        
        # load model
        model = torch.load("train_model.pt")
        
        log_probability = model(image_tensor)
                    
        prediction = torch.exp(log_probability)
        probability = list(prediction.detach().numpy()[0])
        predicted = probability.index(max(probability))

        print ("Prediction digit:", predicted)
        self.text.setText('Recognised Digit: ' + str(predicted))

        self.graph = self.view_classify(prediction)
    
    def view_classify(self, prediction):

        prediction = prediction.data.numpy().squeeze()
        graph = plt.barh(np.arange(10), prediction)
        plt.xticks(np.arange(1))
        plt.yticks(np.arange(10))
        plt.title('Class Probability')
        plt.show()

    def openTrainModel(self):
        self.open_train_model = trainModel()
        self.open_train_model.show()
    
    def openTrainingImages(self):
        self.open_training_images = viewTrainingImages()
        self.open_training_images.show()


class trainModel(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.setWindowTitle('Train Model')
        self.setGeometry(350, 200, 700, 400)

        title = QLabel('Train Model Status', self)
        title.setAlignment(Qt.AlignCenter)

        self.status = QTextBrowser()

        self.progress = QProgressBar(self)
        self.progress.setMaximum(100)
        self.progress.setGeometry(200, 80, 500, 20)
        
        # initialise buttons 
        download = QPushButton('Download')
        download.setToolTip('Download MNIST datasets')
        download.clicked.connect(self.downloadClicked)
        train = QPushButton('Train')
        train.setToolTip('Train Model')
        train.clicked.connect(self.trainClicked)
        clear = QPushButton('Clear')
        clear.setToolTip('Clear the status message')
        clear.clicked.connect(self.clearClicked)

        # settings for box layout
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

        # downlaod MNIST train datasets
        train_dataset = datasets.MNIST(root='Documents/GitHub/project-1-50/Final/mnist_data_train/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)

        # progress bar to max 50
        if (train_dataset != None):
            self.status.append('Downloading train dataset...')
            while progress_count < 50:
                progress_count += 0.00005
                self.progress.setValue(progress_count)
            self.status.append('Finished downloading test dataset!')
                
        # downlaod MNIST test datasets
        test_dataset = datasets.MNIST(root='Documents/GitHub/project-1-50/Final/mnist_data_test/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=False)

        # progress bar to max 100
        if (test_dataset != None):
            self.status.append('Downloading test dataset...')
            while progress_count < 100:
                progress_count += 0.00005
                self.progress.setValue(progress_count)
            self.status.append('Finished downloading test dataset!')
        
    def trainClicked(self):

        # if downloading is finished
        if (train_dataset != "" and test_dataset != ""):
            progress_count = 0
            self.progress.setValue(progress_count)

            self.status.append('Training model...')

            # load the datasets 
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=False)

            # initialisations for the model
            pixel_size = 784
            processing_layers = [128, 64]
            output_size = 10
            learning_rate = 0.001

            # model structure 
            model = nn.Sequential(nn.Linear(pixel_size, processing_layers[0]),
                                nn.ReLU(),
                                nn.Linear(processing_layers[0], processing_layers[1]),
                                nn.ReLU(),
                                nn.Linear(processing_layers[1], output_size),
                                nn.LogSoftmax(dim=1))

            # Define loss
            criterion = nn.NLLLoss()
            image, label = next(iter(train_loader))
            image = image.view(image.shape[0], -1)

            #log probability of an image
            log_probability = model(image)
            #calculate the NLL loss
            loss = criterion(log_probability, label)

            # Define optimiser
            optimiser = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

            # Training process
            epochs = 5 # trained 5 times

            for i in range(epochs):
                losses = 0
                for image, label in train_loader:

                    image_size = image.shape[0]
                    image = image.view(image_size, -1)

                    # training pass
                    optimiser.zero_grad()
                    # forward
                    logits = model(image)
                    # compute the objective function
                    loss = criterion(logits, label)
                    #compute the partial derivatives of obj with respect to parameters
                    loss.backward()
                    #step in the opposite direction of the gradient 
                    optimiser.step()

                    # add up losses of an image
                    losses += loss.item()

                else:
                    # increase 10% every epoch
                    while progress_count < ((i*10) + 10):
                        progress_count += 0.00005
                        self.progress.setValue(progress_count)
                    self.status.append("Epoch {} - Training loss: {}".format(i, losses/len(train_loader)))
            self.status.append('Finished training datasets!')

            # calculate accuracy
            correct_count = 0
            total_test_datasets = 0
            self.status.append('Calculating accuracy...')
            # loop through the whole test sets
            for image,label in test_loader:
                for i in range(len(label)):
                    img = image[i].view(1, 784)
                    with torch.no_grad():
                        # forward
                        log_probability = model(img)
                    
                    # predict the digit 
                    prediction = torch.exp(log_probability)
                    # calculate probability of the digit
                    probability = list(prediction.numpy()[0])
                    # fetch the predicted digit's label with highest probability
                    predicted_label = probability.index(max(probability))
                    # fetch original label
                    original_label = label.numpy()[i]
                    # check if the label is same as predicted label
                    if(original_label == predicted_label):
                        correct_count += 1
                    total_test_datasets += 1
            while progress_count < 100:
                progress_count += 0.00005
                self.progress.setValue(progress_count)
                    
            # display number of tests and the accuracy
            self.status.append('[{} / {}] Model Accuracy = {}%'.format(correct_count, total_test_datasets, (correct_count/total_test_datasets)*100))

            # save model
            torch.save(model, 'train_model.pt')
        #if MNIST datasets are not downloaded
        else:
            self.status.append("Please download the MNIST datasets before training")

    def clearClicked(self):
        #clear status text box
        self.status.clear()
        progress_count = 0
        self.progress.setValue(progress_count)
        
class viewTrainingImages(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.setWindowTitle('Train Images')
        self.setGeometry(450, 150, 700, 400)

        # load train datasets
        train_dataset = datasets.MNIST(root='Documents/GitHub/project-1-50/Final/mnist_data_train/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=False)

        dataiter = iter(train_loader)
        images, labels = dataiter.next()

        figure = plt.figure()
        num_of_images = 60
        for index in range(1, num_of_images + 1):
            plt.subplot(6, 10, index)
            plt.axis('off')
            plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
        plt.show()

class viewTestingImages(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Testing Images')
        self.setGeometry(450, 150, 700, 400)

        train_dataset = datasets.MNIST(root='Documents/GitHub/project-1-50/Final/mnist_data_test/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=False)

        dataiter = iter(train_loader)
        images, labels = dataiter.next()

        figure = plt.figure()
        num_of_images = 60
        for index in range(1, num_of_images + 1):
            plt.subplot(6, 10, index)
            plt.axis('off')
            plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    sys.exit(app.exec_())