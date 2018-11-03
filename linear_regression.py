import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# 0.4.0版本 Tensors并Variables已合并， from torch.autograd import Variable
import matplotlib.pyplot as plt


# 《深度学习入门之Pytorch》 上的一维线性回归例子
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


x_train = np.array([[3.3], [4.4], [5.5], [6.7], [6.93], [4.168],
                   [9.779], [6.182], [7.59], [2.167], [7.042],
                   [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                   [3.366], [2.596], [2.53], [1.221], [2.827],
                   [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# plt.plot(x_train, y_train, 'ok')
# plt.show()

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

flag = False
if torch.cuda.is_available():
    flag = True

if flag:
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

# print(model)
# print(model.parameters())
# print(list(model.parameters()))


criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

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


model.eval()
if flag:
    x_train = x_train.cuda()
    predict = model(x_train)
    predict = predict.cpu().data.numpy()
    plt.plot(x_train.cpu().numpy(), y_train.cpu().numpy(), 'ro', label='Original data')
    plt.plot(x_train.cpu().numpy(), predict, label='Fitting Line')
    plt.show()

else:
    predict = model(x_train)
    predict = predict.data.numpy()
    plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
    plt.plot(x_train.numpy(), predict, label='Fitting Line')
    plt.show()
