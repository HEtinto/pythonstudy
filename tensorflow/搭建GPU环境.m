## 自动匹配并且搭建相应的tensorflow GPU版本
### 1.创建一个虚拟环境
```python
conda create -n tensorflow_gpu=3.6
```
### 2.进入创建的虚拟环境
```python
conda activate tensorflow_gpu
```
### 3.安装常用库
```python
conda install anaconda
```
### 4.安装tensorflow
```python
# 输入以下指令后将自动匹配适配版本
# 按照提示输入y继续安装即可
conda install tensorflow-gpu
```
### 5.测试安装
```python
# 运行下列代码即可
python
import tensorflow as tf
sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
```


## 搭建GPU(1.14.0)
### 1.创建一个虚拟环境
```python
conda create -n python36tfgpu python=3.6.5
# conda 会自动解析依赖，判断需要安装的包，并提示是否继续
# 输入y，同意并继续安装
```
### 2.切换到创建的虚拟环境
```python
conda activate python36tfgpu
```
### 3.安装tensorflow
```python
conda install tensorflow-gpu=1.14.0
# 按照提示输入y继续安装即可
```
### 4.测试安装
```python
# 运行下列代码即可
python
import tensorflow as tf
sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
```