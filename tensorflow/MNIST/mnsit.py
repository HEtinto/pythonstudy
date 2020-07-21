import tensorflow_core.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print('训练集train数量:', mnist.train.num_examples, '验证集 validation数量:',
       mnist.validation.num_examples, ',测试集test数量:', mnist.test.num_examples)

# mnist中每张图片有28*28=784个像素点
tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32, [None, 784], name="X")
# 0-9一共10个数字>=10个类别
y = tf.compat.v1.placeholder(tf.float32, [None, 10], name="Y")
# 定义模型变量
# 用正态分布随机数初始化权重w，以常数0初始化偏置b
w = tf.Variable(tf.random.normal([784, 10]), name="w")
b = tf.Variable(tf.zeros([10]), name="b")
# 定义当前向计算和结果分类
forward = tf.matmul(x, w) + b
pred = tf.nn.softmax(forward) # Softmax分类

# 设置训练参数
train_epochs = 50  # 训练次数
batch_size = 100   # 单次训练样本数
total_batch = int(mnist.train.num_examples/batch_size)  # 一轮训练有多少批次
display_step = 1  # 显示粒度
learning_rate = 0.1  # 学习率
# 定义损失函数，选择优化器
# 定义交叉熵损失函数
loss_function = tf.reduce_mean(-tf.compat.v1.reduce_sum(y*tf.math.log(pred), reduction_indices=1))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# 定义准确率
# 检查预测类别tf.argmax(pred,1)与实际类别tf.argmax(y.1)的匹配情况，argmax能把最大值下标取出来
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) # 相等返回True
# 准确率，将布尔值转化为浮点数，并计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.compat.v1.Session() # 声明会话
init = tf.compat.v1.global_variables_initializer()  # 变量初始化
sess.run(init)

# 数据可视化，此处使用python的matplotlib库中提供的方法
def plot_image(image):
    plt.imshow(image.reshape(28, 28), cmap='binary')
    plt.show()


# plot_image(mnist.train.images[1])