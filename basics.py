import torch
import numpy as np
from torch.autograd import Variable
print(torch.__version__)

# Tensors(张量)
# 构建一个 2x3 的矩阵, 未初始化的
x = torch.Tensor(2, 3)
print(x)

# 构建一个随机初始化的矩阵:
x = torch.rand(2, 3)
print(x)

# torch.Size 实际上是一个 tuple（元组）, 所以它支持所有 tuple（元组）的操作.
size = x.size()
print(size)


# 操作
# =======================加法=======================

a = torch.ones(2, 3)
b = torch.zeros(2, 3)

print('a+b:\n', a + b)

print('torch.add(a, b):\n', torch.add(a, b))

# 提供一个输出 tensor 作为参数
result = torch.Tensor(2, 3)
torch.add(a, b, out=result)
print('result:\n', result)

# adds a to b
b.add_(a)
print('b.add_(a):\n', b)

# 任何改变张量的操作方法都是以后缀 _ 结尾的. 例如: a.copy_(b), a.t_(), 将改变张量 a.
print(a.copy_(b))
print(a.t_())  # 转置

# =======================加法=======================
# 用索引处理张量
print(a)
print(a[:, 1])

# =======================view=======================
# 改变tensor的大小, 可以使用view:
c = torch.randn(4, 4)
d = c.view(2, 8)
e = d.view(16)
f = e.view(-1, 4)  # -1是从其他维度推断出来的
print(c, d, e, f)
print(c.size(), d.size(), e.size(), f.size())

# =======================和 NumPy 数组转换=======================
# torch Tensor 和 NumPy 数组将会共享它们的实际的内存位置, 改变一个另一个也会跟着改变.
var_n = np.arange(6).reshape(2, 3)
var_t = torch.from_numpy(var_n)
print(var_n, var_t)
var_n[0, 0] = 100  # 改变一个，两个都变
print(var_n, var_t)

var_n_1 = var_t.numpy()
print(var_n_1)

# 可以使用 .cuda 方法将 Tensors 在GPU上运行.
if torch.cuda.is_available():
    a = a.cuda()
    print(a)


# =======================autograd =======================
# x = Variable(torch.ones(1), requires_grad=True)
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# y = 2 * x + 1
y = w * x + b

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad)    # x.grad = 2  对x求导
print(w.grad)    # w.grad = 1
print(b.grad)    # b.grad = 1
print(x.requires_grad)


