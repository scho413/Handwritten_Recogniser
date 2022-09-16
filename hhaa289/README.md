class ModelWindow(QWidget): - Class Description:
    
   This is a window contains a selection of model types. This window is opened when the model button is pressed (In MyApp class)

class MyApp(QWidget): - Function Descriptions:

  def CanvasGroup(self):
    
   This function creates a blank canvas for the user to draw on and puts it in a group box using the QGroupBox() pyqt5 widget. This is to ensure we can lay the app out properly. It also sets the setMouseTracking() to true so that we can track the mouses movements
  
  def mouseMoveEvent(self, e):
    
   This function gets the mouses current position and draws on the canvas created in the CanvasGroup. We draw on the canvas using the QPainter widget and drawpoint. 
   
  def ButtonsGroups(self):
    
   This function initialises the push buttons used in the app using the QPushButton and puts them in the group box using the QGropuBox widget, this is so we can lay the app out using the QGridLayout widget. The buttons are also connceted to their related fuctions so that when they are clicked the proper outcome is made
    
   def clearClicked(self):
    
   This function clears the canvas when the clear button is pushed.
    
   def RandomClicked(self):

   def ModelClicked(self):
    This opens the ModelWindow class that contains the model types

   def RecogniseClicked(self):
