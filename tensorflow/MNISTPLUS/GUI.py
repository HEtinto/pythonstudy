from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox
from PyQt5.QtWidgets import (QHBoxLayout, QVBoxLayout, QPushButton, QSplitter,
    QComboBox, QLabel, QSpinBox, QFileDialog,
    QTextEdit, QGridLayout, QTextBrowser, QDesktopWidget)
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QFont
from PaintBoard import PaintBoard
from MainWidget import MainWidget
from PyQt5.QtWidgets import QApplication

import sys
import tensorflow as tf
import numpy as np
"""
出现错误：
Function call stack:
distributed_function

解决方案为下面的代码：
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras


class MainWidget(QWidget):
    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)

        self.__InitData()  # 先初始化数据，再初始化界面
        self.__InitView()
        self.center()

    def __InitData(self):
        '''
                  初始化成员变量
        '''
        self.__paintBoard = PaintBoard(self)
        # 获取颜色列表(字符串类型)
        self.__colorList = QColor.colorNames()

    def __InitView(self):
        '''
                  初始化界面
        '''

        '''
        self.setFixedSize(800, 600)
        设置布局大小，如果采用setFixedSize(0,0,a,b)的方法设置，那么窗口将是不可伸缩的
        下面的方法设置的窗口是可以伸缩的，前面两个参数(0,0)表示从屏幕上的(0,0)位置开始，
        后面两个参数分别表示宽和高,它显示一个a*b的界面
        '''
        self.setGeometry(300, 300, 1000, 900)
        self.setWindowTitle("手写数字识别")
        self.setWindowIcon(QIcon('C:/Users/Y2469/Desktop/'
                                 'pythonstudy/tensorflow/MNISTPLUS/logo.ico'))
        """
        使用QLabel的setFont属性改变字体大小
        需要PyQt5中的QGui库的QFont属性设置
        我们通过QFont的setPointSize属性设置字体大小
        并通过向QLabel的setFont属性中传入一个已经设置好字体大小的QFont对象来完成设置
        """
        self.label_name = QLabel('桂林电子科技大学', self)
        ft1 = QFont()
        ft1.setPointSize(30)
        self.label_name.setFont(ft1)
        self.label_name.setGeometry(500, 5, 300, 35)

        self.label_name = QLabel('数学与计算科学学院', self)
        self.label_name.setFont(ft1)
        self.label_name.setGeometry(500, 35, 300, 35)

        self.label_name = QLabel('班级:101', self)
        self.label_name.setFont(ft1)
        self.label_name.setGeometry(500, 65, 200, 35)

        self.label_name = QLabel('姓名:喻建明', self)
        self.label_name.setFont(ft1)
        self.label_name.setGeometry(500, 95, 200, 35)
        # 接下来，我将创建一个QTextBrowser的对象用于输出结果
        self.text_browser = QTextBrowser(self)
        # 这里给出了这个输出板中的初始内容
        ft1 = QFont()
        ft1.setPointSize(20)
        self.text_browser.setFont(ft1)
        self.text_browser.setText("输出运行结果")
        # 设置输出板的位置
        self.text_browser.setGeometry(10, 10, 480, 400)

        layout = QVBoxLayout()
        layout.addWidget(self.text_browser)

        # 新建一个水平布局作为本窗体的主布局
        main_layout = QHBoxLayout(self)
        # 设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10)

        # 在主界面左侧放置画板
        # 后面两个参数表示放置在左下位置——————QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom
        main_layout.addWidget(self.__paintBoard, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)

        # 新建垂直子布局用于放置按键
        sub_layout = QVBoxLayout()
        # 设置此子布局和内部控件的间距为5px
        sub_layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(self)  # 占位符
        sub_layout.addWidget(splitter)

        self.__btn_Recognize = QPushButton("开始识别")
        self.__btn_Recognize.setFixedSize(200, 100)
        ft1 = QFont()
        ft1.setPointSize(40)
        self.__btn_Recognize.setFont(ft1)
        self.__btn_Recognize.setParent(self)
        self.__btn_Recognize.clicked.connect(self.on_btn_Recognize_Clicked)
        sub_layout.addWidget(self.__btn_Recognize)

        self.__btn_Clear = QPushButton("清空画板")
        self.__btn_Clear.setFixedSize(200, 100)
        self.__btn_Clear.setFont(ft1)
        self.__btn_Clear.setParent(self)  # 设置父对象为本界面

        # 将按键按下信号与画板清空函数相关联
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear)
        sub_layout.addWidget(self.__btn_Clear)

        self.__btn_Quit = QPushButton("退出")
        self.__btn_Quit.setFixedSize(200, 100)
        self.__btn_Quit.setFont(ft1)
        self.__btn_Quit.setParent(self)  # 设置父对象为本界面
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)

        self.__btn_Save = QPushButton("保存作品")
        self.__btn_Save.setFixedSize(200, 100)
        self.__btn_Save.setFont(ft1)
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        sub_layout.addWidget(self.__btn_Save)

        """
        在这里将设置此按钮的大小，我们先通过QFont属性设置字体大小，创建QFont对象后，通过setPointSize()方法设置字体大小
        然后通过样式表设置复选框大小:{your QCheckbox}.setStyleSheet("QCheckBox::indicator { width: npx; height: mpx;}")
        我们直接调用QCheckBox属性的setStyleSheet方法即可以调整复选框的大小
        """
        self.__cbtn_Eraser = QCheckBox("  使用橡皮擦")
        ft1 = QFont()
        ft1.setPointSize(26)
        self.__cbtn_Eraser.setFont(ft1)
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        sub_layout.addWidget(self.__cbtn_Eraser)
        self.__cbtn_Eraser.setStyleSheet("QCheckBox::indicator { width: 30px; height: 30px;}")

        """
        我们开始设置画笔粗细控件的大小
        可以直接使用QFont方法设置，但是这样设置后发现字体被遮挡
        解决方案是将QLabel的setFixedHeight方法设置为40，即为小于设置字体的大小(此处设置为26)
        """
        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("画笔粗细")
        ft1 = QFont()
        ft1.setPointSize(26)
        self.__label_penThickness.setFont(ft1)
        self.__label_penThickness.setFixedHeight(40)
        sub_layout.addWidget(self.__label_penThickness)

        """
        QSpinBox控件:
        setMinimum()	设置计数器的下界
        setMaximum()	设置计数器的上界
        setRange()	    设置计数器的最大值，最小值，步长值
        setValue()	    设置计数器的当前值
        Value()	        返回计数器的当前值
        singleStep()	设置计数器的步长值
        我们同样可以通过setFont()方法设置QSpinBox对象的字体大小
        通过样式表设置计数器的大小，即调用 setStyleSheet("QCheckBox::indicator { width: 30px; height: 30px;}")
        同样的，我们通过setFixedHeight()方法设置大小上限，如果不设置，那么对于字体大小和控件大小的设置将会失效
        画笔粗细在PaintBoard.py文件中修改
        """
        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(30)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(18)  # 默认粗细为10
        self.__spinBox_penThickness.setSingleStep(2)  # 最小变化值为2
        self.__spinBox_penThickness.setFont(ft1)
        self.__spinBox_penThickness.setFixedHeight(40)
        self.__spinBox_penThickness.setStyleSheet("QCheckBox::indicator { width: 30px; height: 30px;}")
        # 绑定槽函数
        self.__spinBox_penThickness.valueChanged.connect(
            self.on_PenThicknessChange)  # 关联spinBox值变化信号和函数on_PenThicknessChange
        sub_layout.addWidget(self.__spinBox_penThickness)

        """
        同样的，我们通过QFont设置字体大小，并通过setFixedHeight设置大小上限
        """
        self.__label_penColor = QLabel(self)
        self.__label_penColor.setText("画笔颜色")
        self.__label_penColor.setFont(ft1)
        self.__label_penColor.setFixedHeight(40)
        sub_layout.addWidget(self.__label_penColor)

        self.__comboBox_penColor = QComboBox(self)

        self.__fillColorList(self.__comboBox_penColor)  # 用各种颜色填充下拉列表
        self.__comboBox_penColor.currentIndexChanged.connect(
            self.on_PenColorChange)  # 关联下拉列表的当前索引变更信号与函数on_PenColorChange
        self.__comboBox_penColor.setStyleSheet("QCheckBox::indicator { width: 50px; height: 50px;}")
        sub_layout.addWidget(self.__comboBox_penColor)

        main_layout.addLayout(sub_layout)  # 将子布局加入主布局

    def __fillColorList(self, comboBox):

        index_black = 0
        index = 0
        for color in self.__colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)

    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        self.__paintBoard.ChangePenColor(color_str)

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)

    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath[0])
        print(savePath[0])

    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True  # 进入橡皮擦模式
        else:
            self.__paintBoard.EraserMode = False  # 退出橡皮擦模式

    def on_btn_Recognize_Clicked(self):
        savePath = "./text.png"
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath)
        print(savePath)
        # 加载图像
        img = keras.preprocessing.image.load_img(savePath, target_size=(28, 28))
        img = img.convert('L')
        x = keras.preprocessing.image.img_to_array(img)
        x = abs(255 - x)
        # x = x.reshape(28,28)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        new_model = keras.models.load_model('C:/Users/Y2469/Desktop/'
                                            'pythonstudy/tensorflow/'
                                            'MNISTPLUS/model/my_model.h5')
        prediction = new_model.predict(x)
        output = np.argmax(prediction, axis=1)
        print("手写数字识别为：" + str(output[0]))
        # 此处将控制台内容重定向输出到text_browser输出面板中
        self.printf("手写数字识别为：" + str(output[0]))

    def printf(self, mes):
        # 这个函数将帮助我们在text_browser控件中输出我们需要输出的信息
        self.text_browser.append(mes)  # 在指定的区域显示提示信息
        self.cursot = self.text_browser.textCursor()
        self.text_browser.moveCursor(self.cursot.End)
    # 接收信号str的信号槽

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)

    def outputWritten(self, text):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textEdit.setTextCursor(cursor)
        self.textEdit.ensureCursorVisible()

    def Quit(self):
        self.close()


# 用来发射标准输出作为信号
class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)  # 定义一个发送str的信号

    def write(self, text):
        self.textWritten.emit(str(text))


def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)

    mainWidget = MainWidget()  # 新建一个主界面
    mainWidget.show()  # 显示主界面

    exit(app.exec_())  # 进入消息循环


if __name__ == '__main__':
    main()