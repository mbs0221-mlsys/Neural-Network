# Neural-Network
C++神经网络框架
* 初步实现了反向传播算法，支持搭建stacked-model
* 支持5D张量(sample, frame, width, height, channel)
* 已实现如下基本运算类型
  * matmul/add
  * element-wise operation
  * broadcast operation
  * conv2d/conv3d operation
  * max/min/average pooling
  * max/min/average upsampling
  * permute/reshape/flatten
  * reduce_sum/reduce_mean
  * sigmoid/relu/leaky_relu/softmax
* 初步实现了计算图框架，支持多分支网络模型
* 预计进一步实现：
  * 基于梯度的一些优化算法
