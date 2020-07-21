#### 1.遇到了pycharm中同级目录不能相互导入的问题 
```python
# 采用新的引入方法
from folder import modular
```
#### 2.在使用tensorflow时发现其所包含的文件并不满足需求，缺少了需要的tutorials文件夹，使用git方法去获取只能获取整个库，这里采用SVN来获取github上的tutorials文件夹 
```python
# 在github中获取到的链接
url = 'https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials'
# 实际输入到SVN checkout中链接
url = 'https://github.com/tensorflow/tensorflow/trunk/tensorflow/examples/tutorials'
# 这里做了一个替换:将 "/branches/branchname/" 替换为 "/trunk/"
```
![avatar](SVN.png) 
#### 3.使用tensorflow的placeholder模块时发现该模块已经被移除 
```python
# 原始代码
x = tf.placeholder(tf.float32,[None,784],name="X")
# 更改后的代码
x = tf.compat.v1.placeholder(tf.float32, [None, 784], name="X")
# 依旧报错
"RuntimeError: tf.placeholder() is not compatible with eager execution."
# 在前面加上代码
tf.compat.v1.disable_eager_execution()
```
#### 4.初始化偏置b的原因
```
'''
* 对于分类器而言，如果不加上偏置项b，那么我们的分类器只能过原点
* 通过偏置项，可以让模型在训练的过程中，动态地调整分类器以画出最佳的决策面
'''
```
#### 5.调整学习率提高准确率
```python
# 在测试中将学习率从0.01上调到0.1,测试集的正确率达到了0.9
# 相比之前的正确率0.86有所上升
```

#### 6.将.gz数据集转化为csv格式的数据集
```python
# 注意MNIST数据集的路径，在此代码下，MNSIT数据集应该位于同一级目录
# 注意要将当前目录下得MNIST数据集解压，然后再执行下述代码
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "mnist_train.csv", 60000)
convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
        "mnist_test.csv", 10000)
``` 
### 在使用save等方式保存模型时，遇到了模型不能保存的问题
```python
# 参考官方文档
# 需要注意的是，我们可能需要把路径改成如下精确的绝对路径
checkpoint_path = "C:/Users/Y2469/Desktop/pythonstudy/tensorflow/MNISTPLUS/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个保存模型权重的回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# 使用新的回调训练模型
model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # 通过回调训练

# 这可能会生成与保存优化程序状态相关的警告。
# 这些警告（以及整个笔记本中的类似警告）是防止过时使用，可以忽略。

# 创建一个新的模型实例
model = create_model()

# 训练模型
model.fit(train_images, train_labels, epochs=5)

# 将整个模型保存为HDF5文件
model.save('C:/Users/Y2469/Desktop/pythonstudy/tensorflow/MNISTPLUS/model/my_model.h5')
```

### 参考文献
```python
# 参考代码
url = 'https://www.cnblogs.com/HuangYJ/p/11642475.html'
# 数据集
url = 'http://yann.lecun.com/exdb/mnist/'
# tensorflow官网官方文档
url = 'https://www.tensorflow.org/'
# tensorflow非官方文档中文版
url ='https://www.w3cschool.cn/tensorflow_python/tensorflow_python-bm7y28si.html'
``` 