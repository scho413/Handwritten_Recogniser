Handwritten Digit Recognizer
==============================

This is a handwritten digit recognizer where I designed and developed the image training part (scho413) and my partner has developed the GUI (hhaa289)

Our project uses pyqt5 GUI to perform deep neural network on the MNIST database. The user will be able visualise the process of downloading and training the MNIST datasets. 

Once the process is finished, user will be able to view trained images.

The user can use the mouse to draw a number on the draw panel and test out whether the programm identifies the number correctly.

Description of the files
===========================

main: main source file which contains all the code to run the programm including the GUI

train_model.pt: It's the output file of the trained deep neural network. This can be updated if new trains are performed otherwise this can be loaded to perform recognition tests

requirement: explains about the requirements (libraries needed) to run the programm

README: explains about our project and the files 

Steps:
=======

1. download and train the model by opening the model train window from the menu bar or with a shirt key Ctrl+M. 

2. Once downloading and training is done, exit the train model

3. To view which images are used to train the model, click on the view trained images to view the digit images.

4. Draw a digit on the drawing panel. 

5. Select a model and press recognise button to recognise the digit you have written 


