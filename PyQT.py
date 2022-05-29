import cv2
import numpy as np
import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras




model = tf.keras.models.load_model('save_model/64model7.h5')

img_height = 180
img_width =  180

k = 0
img = 'NULL'
predictions = 'NULL'
img_name = "img.png"

camera_port = 0
camera = cv2.VideoCapture(camera_port)

class ShowVideo(QtCore.QObject):
    # initiating the built in camera
    camera_port = 0
    camera = cv2.VideoCapture(camera_port)
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)

    @QtCore.pyqtSlot()
    def startVideo(self):

        run_video = True
        while run_video:
            
            ret, image = self.camera.read()

            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width, _ = color_swapped_image.shape

            #width = camera.set(CAP_PROP_FRAME_WIDTH, 1600)
            #height = camera.set(CAP_PROP_FRAME_HEIGHT, 1080)
            #camera.set(CAP_PROP_FPS, 15)

            qt_image = QtGui.QImage(color_swapped_image.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)

            self.VideoSignal.emit(qt_image)
            
            img_name = "img.png"
            cv2.imwrite(img_name, image)
            
            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(25, loop.quit) #25 ms
            loop.exec_()
            
            img = keras.preprocessing.image.load_img(
            img_name, target_size = (img_height, img_width)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
                
            predictions = model.predict(img_array)
            predictions = np.argmax(predictions)
            if(predictions == 0):
                os.system('cls')
                print('생분해성 폐기물이에요 재활용이 필요 없어요!')
            elif(predictions == 1):
                os.system('cls')
                print('재활용이 필요한 폐기물이네요 재활용으로 지구를 구합시다!') 
            #k = cv2.waitKey(1)
            
                # SPACE pressed
            
            
                        
            

    
    def Classification(self):
        
        ret, image = self.camera.read()

        color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = color_swapped_image.shape

            #width = camera.set(CAP_PROP_FRAME_WIDTH, 1600)
            #height = camera.set(CAP_PROP_FRAME_HEIGHT, 1080)
            #camera.set(CAP_PROP_FPS, 15)

        qt_image = QtGui.QImage(color_swapped_image.data,
                                 width,
                                height,
                                color_swapped_image.strides[0],
                                QtGui.QImage.Format_RGB888)

        self.VideoSignal.emit(qt_image)
        
        

        img_name = "img.png"
        cv2.imwrite(img_name, image)
        '''        
        img = keras.preprocessing.image.load_img(
            img_name, target_size = (img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
            
        predictions = model.predict(img_array)
        predictions = np.argmax(predictions)
        if(predictions == 0):
            print('생분해성 폐기물이에요 재활용이 필요 없어요!')
        elif(predictions == 1):
            print('재활용이 필요한 폐기물이네요 재활용으로 지구를 구합시다!')                
        '''        
class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
    
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()
    
    def initUI(self):
        self.setWindowTitle('이미지 분류 프로그램')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    thread = QtCore.QThread()
    thread.start()
    vid = ShowVideo()
    vid.moveToThread(thread)
    image_viewer = ImageViewer()
    
    vid.VideoSignal.connect(image_viewer.setImage)

    # Button to start the videocapture:

    push_button1 = QtWidgets.QPushButton('Start') # 
    push_button2 = QtWidgets.QPushButton('Auto Capturing')
    label = QtWidgets.QLabel('Test')
    
    
    push_button1.clicked.connect(vid.startVideo)
    push_button2.clicked.connect(vid.Classification)
    vertical_layout = QtWidgets.QVBoxLayout() # 수직 박스 레이아둣. 세로 방향으로 위젯들을 배치

    vertical_layout.addWidget(image_viewer)
    vertical_layout.addWidget(push_button1)
    vertical_layout.addWidget(push_button2)
    vertical_layout.addWidget(label)


    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(vertical_layout)

    
    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    sys.exit(app.exec_())