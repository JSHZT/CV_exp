# coding: utf-8

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import ParameterTuple, Parameter
from mindspore import dtype as mstype


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        #定义矩阵乘法
        self.matmul = ops.MatMul()
        #添加模型的参数
        self.z = Parameter(Tensor(np.array([1.0, 1.0, 1.0], np.float32)), name='z')

    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out
    
# 实例化模型对象
model = Net()

for m in model.parameters_and_names():
    print(m)

#定义张量
x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
result = model(x, y)
print(result)