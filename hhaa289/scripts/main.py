import sys
from ModelWindow import ModelWindow
from PyQt5.QtWidgets import (QApplication, QDesktopWidget, QWidget, QLabel, QComboBox, 
QVBoxLayout, QPushButton, QHBoxLayout, QGroupBox, QGridLayout)
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.setWindowTitle('MyApp')
        self.setGeometry(400, 150, 500, 400)

        self.newWindow = ModelWindow() #window for model selection
        self.m = ModelWindow().modelSelected()

        #App Layout
        grid = QGridLayout()
        grid.addWidget(self.ButtonsGroup(), 0, 1)
        grid.addWidget(self.CanvasGroup(), 0, 0)
        self.setLayout(grid)

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
        x = e.x()
        y = e.y()

        qp = QPainter(self.Display.pixmap())#draw on canvas
        qp.setPen(QPen(Qt.black, 10, Qt.SolidLine))#set pen colour
        qp.drawPoint(x,y)
        qp.end()
        self.update()

    def ButtonsGroup(self):
        groupboxButtons = QGroupBox()

        self.ClearAction = QPushButton(self)
        self.ClearAction.setText('Clear')
        self.ClearAction.clicked.connect(self.clearClicked)

        Random = QPushButton(self)
        Random.setText('Random')
        #Random.clicked.connect(self.RandomClicked)

        self.Model = QPushButton(self)
        self.Model.setText('Model')
        self.Model.clicked.connect(self.ModelClicked)

        Recognise = QPushButton(self)
        Recognise.setText('Recognise')
        #Recognise.clicked.connect(self.RecogniseClicked)

        vbox = QVBoxLayout()
        vbox.addWidget(self.ClearAction)
        vbox.addWidget(Random)
        vbox.addWidget(self.Model)
        vbox.addWidget(Recognise)
        groupboxButtons.setLayout(vbox) 

        return groupboxButtons

    def clearClicked(self):
        if self.ClearAction:
            self.Display.clear()
            self.Display.setPixmap(self.canvas)
            self.update()
    
    def ModelClicked(self):
        if self.newWindow.isVisible():
            self.newWindow.hide()
        else:
            self.newWindow.show()
            

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())

