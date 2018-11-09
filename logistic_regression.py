import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# 设定随机种子
torch.manual_seed(2018)

flag = False
if torch.cuda.is_available():
    flag = True
# flag = False

with open('data.txt', 'r') as f:
    data = f.readlines()
    data_list = [i.split('\n')[0] for i in data]
    data_list = [i.split(',') for i in data_list]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]

    # 标准化
    x0_max = max([i[0] for i in data])
    x1_max = max([i[1] for i in data])
    data = [(i[0] / x0_max, i[1] / x1_max, i[2]) for i in data]

    x_0 = list(filter(lambda x: x[-1] == 0.0, data))
    x_1 = list(filter(lambda x: x[-1] == 1.0, data))

    plot_x0 = [i[0] for i in x_0]
    plot_y0 = [i[1] for i in x_0]

    plot_x1 = [i[0] for i in x_1]
    plot_y1 = [i[1] for i in x_1]


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sigmoid(x)
        return x


if flag:
    model = LogisticRegression().cuda()
else:
    model = LogisticRegression()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


np_data = np.array(data, dtype='float32')

x_data = np_data[:, 0:2]
y_data = np_data[:, -1]

if flag:
    x_data = torch.from_numpy(x_data).cuda()
    y_data = torch.from_numpy(y_data).unsqueeze(1).cuda()
else:
    x_data = torch.from_numpy(x_data)
    y_data = torch.from_numpy(y_data).unsqueeze(1)

num_epochs = 20000
start = time.time()
for epoch in range(num_epochs):
    # forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 计算正确率
    mask = y_pred.ge(0.5).float()  # 大于等于0.5置为True, 反之置为False
    acc = (mask == y_data).sum().item() / y_data.shape[0]
    if (epoch + 1) % 200 == 0:
        print('epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(epoch+1, loss.item(), acc))

time_elapsed = time.time() - start
print('Training complete in {:.0f}s'.format(time_elapsed % 60))

# 结果可视化
w0, w1 = model.lr.weight[0]
w0 = float(w0.item())
w1 = float(w1.item())
b = float(model.lr.bias.item())
plot_x = np.arange(0.2, 1, 0.01)
plot_y = (-w0 * plot_x - b) / w1
plt.plot(plot_x, plot_y)

plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
plt.legend(loc='best')
plt.show()

