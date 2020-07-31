# HiNN
#### 参考某教程完成的神经网络框架

#### 网络结构和超参数通过json文件进行定义

#### 使用jsoncpp解析json文件，Armadillo完成矩阵运算，protobuf存储模型

#### 支持以下特性：

- 全连接层，池化层和卷积层的前向/反向传播

- 模型的存储和载入，以进行fine-tune

- ReLU和tanh激活函数，交叉熵和Hinge损失函数

- 随机梯度下降 / Momentum / RMSprop等优化方法

- L2正则 / Dropout / BatchNormalization
