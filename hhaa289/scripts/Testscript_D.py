import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('MyApp')
        self.setGeometry(400, 150, 500, 400)

        x = None
        y = None

        self.setMouseTracking(True)

        self.canvas = QPixmap(500,500)
        self.canvas.fill(Qt.white)
        self.Display = QLabel()
        self.Display.setPixmap(self.canvas)

        vbox = QVBoxLayout
        vbox = QHBoxLayout() 
        vbox.addWidget(self.Display)
        self.setLayout(vbox) 

    def mouseMoveEvent(self, e):
        x = e.x()
        y = e.y()

        qp = QPainter(self.Display.pixmap())#draw on canvas
        qp.setPen(QPen(Qt.black, 10, Qt.SolidLine))#set pen colour
        qp.drawPoint(x,y)
        qp.end()
        self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
        