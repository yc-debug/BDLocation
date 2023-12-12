import sys
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pandas as pd
import numpy as np
import pyproj
import generatePointSet
import geometricCalculate
import KalmanFilter
import matplotlib.pyplot as plt
import math


class Example(QWidget):
    def __init__(self):
        super().__init__()
        # self.draw_test()
        self.x = [[0, 0], [0, 0]]

    def draw_test(self):
        layout = QVBoxLayout(self)
        label = QLabel()
        pixmap = QPixmap("./data/1.jpg")
        self.setGeometry(100, 100, pixmap.width(), pixmap.height())
        label.setPixmap(pixmap)
        layout.addWidget(label)
        painter = QPainter(label)
        painter.translate(1416, 198)
        painter.scale(5, -5)
        painter.drawLine(0, 0, 100, 100)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.HighQualityAntialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        pixmap = QPixmap("./data/1.jpg")
        self.setGeometry(100, 100, pixmap.width(), pixmap.height())
        painter.drawPixmap(self.rect(), pixmap)
        painter.translate(1416, 198)
        painter.scale(1.69, -1.72)
        pen = QPen(Qt.red, 1)
        painter.setPen(pen)
        for item in self.x:
            s = item[0]
            e = item[1]
            painter.drawLine(s[0], s[1], e[0], e[1])

    def drawLine(self, start, end):
        self.x = [start, end]
        self.repaint()

    # def mouseMoveEvent(self, event):
    #     s = event.windowPos()
    #     self.setMouseTracking(True)
    #     print('X:' + str(s.x()))
    #     print('Y:' + str(s.y()))
    def drawLines(self, x):
        self.x = x
        self.repaint()


app = QApplication(sys.argv)
window = QWidget()
ex = Example()
true_data = generatePointSet.generateEdgeSet()
# 判断逻辑
# 创建一个全为 0 的 77x77 数组
array = np.zeros((len(true_data), len(true_data)))
col = 1
row = 0


def change(t):
    global col
    global row
    global array
    if t == 1 and col != row:
        array[row][col] = 1
        col = col + 1
    else:
        array[row][col] = 0
        col = col + 1
    if col > len(true_data) - 1:
        col = 0
        row = row + 1
    if row > len(true_data) - 1:
        np.savetxt('Adjacency.txt', array)
        return
    ex.drawLine(start=true_data[row], end=true_data[col])


def yes():
    change(1)


def no():
    change(0)


layout = QVBoxLayout()
btn = QPushButton("是")
btn1 = QPushButton("否")
btn.clicked.connect(yes)
btn1.clicked.connect(no)
# 画线
ex.drawLines([true_data[row], true_data[col]])
layout.addWidget(ex)
layout.addWidget(btn)
layout.addWidget(btn1)
window.setLayout(layout)
window.showMaximized()
# ex.show()


sys.exit(app.exec_())
