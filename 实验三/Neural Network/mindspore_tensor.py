import numpy as np
import mindspore
from mindspore import ops
from mindspore import Tensor, CSRTensor, COOTensor

data = [1, 0, 1, 0]
x_data = Tensor(data)

np_array = np.array(data)
x_np = Tensor(np_array)

from mindspore.common.initializer import One, Normal
# 初始化一个全1的tensor
tensor1 = mindspore.Tensor(shape=(3, 3), dtype=mindspore.float32, init=One())
# 初始化一个符合标准正态分布的Tensor
tensor2 = mindspore.Tensor(shape=(3, 3), dtype=mindspore.float32, init=Normal())
# 打印查看结果(print函数，python的debug神器)
print("tensor1:\n", tensor1)
print("tensor2:\n", tensor2)

from mindspore import ops
x_ones = ops.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")
x_zeros = ops.zeros_like(x_data)
print(f"Zeros Tensor: \n {x_zeros} \n")

# 可尝试其他数据类型和大小,查看返回的维度等信息
x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.int32)
print("x_shape:", x.shape)
print("x_dtype:", x.dtype)
print("x_itemsize:", x.itemsize)
print("x_nbytes:", x.nbytes)
print("x_ndim:", x.ndim)
print("x_size:", x.size)
print("x_strides:", x.strides)

# 尝试使用其他非示例代码,熟悉检索具体行、列、范围内的数值
tensor = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
print("First row: {}".format(tensor[0]))
print("value of bottom right corner: {}".format(tensor[1, 1]))
print("Last column: {}".format(tensor[:, -1]))
print("First column: {}".format(tensor[..., 0]))

x = Tensor(np.array([5, 6, 7]), mindspore.float32)
y = Tensor(np.array([7, 8, 9]), mindspore.float32)
output_add = x + y
output_sub = x - y
output_mul = x * y
output_div = y / x
output_mod = y % x
output_floordiv = y // x

print("add:", output_add)
print("sub:", output_sub)
print("mul:", output_mul)
print("div:", output_div)
print("mod:", output_mod)
print("floordiv:", output_floordiv)

data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.concat((data1, data2), axis=0)
print(output)
print("shape:\n", output.shape)

data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.stack([data1, data2])
print(output)
print("shape:\n", output.shape)