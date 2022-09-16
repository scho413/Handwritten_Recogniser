import sys
from PyQt5.QtWidgets import (QApplication, QWidget, 
QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout)

class Buttons(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        #self.setLayout(vbox)
        self.setWindowTitle('QPushButton')
        self.setGeometry(300, 300, 300, 200)

        grid = QGridLayout()
        grid.addWidget(self.ButtonsGroup(), 0, 0)
        self.setLayout(grid)

        self.show()

    def ButtonsGroup(self):
        groupboxButtons = QGroupBox()

        ClearAction = QPushButton(self)
        ClearAction.setText('Clear')
        ClearAction.move(0,0)
        #Clear.clicked.connect(self.clearClicked)

        Random = QPushButton(self)
        Random.setText('Random')
        Random.move(1,0)

        Model = QPushButton(self)
        Model.setText('Model')
        Model.move(2,0)

        Recognise = QPushButton(self)
        Recognise.setText('Recognise')
        Recognise.move(3,0)
        #Recognise.clicked.connect(self.RecogniseClicked)

        vbox = QVBoxLayout()
        vbox.addWidget(ClearAction)
        vbox.addWidget(Random)
        vbox.addWidget(Model)
        vbox.addWidget(Recognise)
        groupboxButtons.setLayout(vbox) 

        return groupboxButtons

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Buttons()
    sys.exit(app.exec_())