from PySide2.QtWidgets import QApplication, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile
from PySide2 import QtWidgets, QtCore
from PaintBoard import PaintBoard
import os
import sys

envpath = r'C:\Anaconda\envs\tensorflow_gpu\Lib\site-packages\PySide2\plugins\platforms'
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = envpath
"""
出现错误：
Function call stack:
distributed_function

解决方案为下面的代码：
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""


class GUI():
    def __init__(self):
        super().__init__()
        # 从文件加载UI定义
        qfile_gui = QFile('gui.ui')
        qfile_gui.open(QFile.ReadOnly)
        qfile_gui.close()
        # 从UI定义中动态创建一个相应的窗口对象
        # 注意：里面的空间对象也成为窗口对象的属性了
        self.ui = QUiLoader().load(qfile_gui)
        # 设置按钮指向


class myStdout():
    def __init__(self):
        self.stdoutbak = sys.stdout
        self.stderrbak = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, info):
        # info信息即标准输出sys.stdout和sys.stderr接收到的输出信息
        str = info.rstrip("\r\n")
        if len(str): self.processInfo(str)  # 对输出信息进行处理的方法

    def processInfo(self, info):
        self.stdoutbak.write("标准输出接收到消息：" + info + "\n")  # 可以将信息再输出到原有标准输出，在定位问题时比较有用

    def restoreStd(self):
        print("准备恢复标准输出")
        sys.stdout = self.stdoutbak
        sys.stderr = self.stderrbak
        print("恢复标准输出完成")

    def __del__(self):
        self.restoreStd()


if __name__ == '__main__':
    # QtDesigner 预览结果与实际运行不一致时添加下面一行代码
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    print("主程序开始运行,创建标准输出替代对象....")
    mystd = myStdout()
    print("标准输出替代对象创建完成,准备销毁该替代对象")
    # mystd.restoreStd()
    app = QApplication([])
    gui = GUI()
    gui.ui.show()
    app.exec_()
    del mystd
    print("主程序结束")


