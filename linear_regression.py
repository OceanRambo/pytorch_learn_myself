import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 《深度学习入门之PyTorch》上的一维线性回归例子 y = wx + b
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        x = self.linear(x)
        return x


x_train = np.array([[3.3], [4.4], [5.5], [6.7], [6.93], [4.168],
                   [9.779], [6.182], [7.59], [2.167], [7.042],
                   [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                   [3.366], [2.596], [2.53], [1.221], [2.827],
                   [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# torch.from_numpy() 改为 torch.Tensor()  数据类型变化了 且
# torch.from_numpy(array)是做数组的浅拷贝，torch.Tensor(array)是做数组的深拷贝
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)

flag = False
if torch.cuda.is_available():
    flag = True
# flag = False

if flag:
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

num_epochs = 1000
for epoch in range(num_epochs):
    if flag:
        inputs = x_train.cuda()
        target = y_train.cuda()
    else:
        inputs = x_train
        target = y_train

    # forward
    out = model(inputs)
    loss = criterion(out, target)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1, num_epochs, loss.data[0]))

print("==> Learned function: y = {:.2f} + {:.2f}*x".format(model.linear.bias[0], model.linear.weight[0][0]))

# 在测试模型时在前面使用 model.eval()
model.eval()
if flag:
    x_train = x_train.cuda()
    predict = model(x_train)
    predict = predict.cpu().data.numpy()
    plt.plot(x_train.cpu().numpy(), y_train.cpu().numpy(), 'ro', label='Original data')
    plt.plot(x_train.cpu().numpy(), predict, label='Fitting Line')
    plt.legend()
    plt.show()

else:
    predict = model(x_train)
    predict = predict.data.numpy()
    plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
    plt.plot(x_train.numpy(), predict, label='Fitting Line')
    plt.legend()
    plt.show()


# # 保存和加载整个模型
# torch.save(model, 'model.pkl')
# the_model = torch.load('model.pkl')
# print(the_model)  # 整个模型
#
#
# # 仅保存和加载模型参数(推荐使用)
# torch.save(model.state_dict(), 'params.pkl')
# # torch.load 返回的是一个 OrderedDict.
# new_model = torch.load('params.pkl')
# print(new_model)
# for k, v in new_model.items():
#     print(k, v)
