import torch
import torch.nn as nn
import numpy as np


# ================================================================== #
#                     1. Tensors(张量)                  #
# ================================================================== #

# 构建一个 2x3 的矩阵, 未初始化的
x = torch.Tensor(2, 3)
print(x)

# 构建一个随机初始化的矩阵:
x = torch.rand(2, 3)
print(x)

# torch.Size 实际上是一个 tuple（元组）, 所以它支持所有 tuple（元组）的操作.
size = x.size()
print(size)

