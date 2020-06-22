from matplotlib import pyplot as plt
import numpy as np


def plot_images_labels_prediction(images, labels, prediction, index, num=10):
    # 图像列表，标签列表，预测列表，从第index个开始显示，缺省一次显示10幅
    fig = plt.gcf()  # 获取当前图表
    fig.set_size_inches(10, 12)
    if num > 25:
        num = 25  # 最多显示25幅
    for i in range(0, num):
        ax = plt.subplot(5, 5, i+1)  # 获取当前要处理的子图
        ax.imshow(np.reshape(images[index], (28, 28)), cmap='binary')
        title = "label=" + str(np.argmax(labels[index]))  # 构建需要显示的图像
        if len(prediction) > 0:
            title += ",predict" + str(prediction[index])
        ax.set_title(title, fontsize=10)  # 显示图上的title信息
        ax.set_xticks([])  # 不显示坐标轴
        ax.set_yticks([])
        index += 1
    plt.show()




