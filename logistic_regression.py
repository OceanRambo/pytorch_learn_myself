import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

with open('data.txt', 'r') as f:
    data = f.readlines()
    data_list = [i.split('\n')[0] for i in data]
    data_list = [i.split(',') for i in data_list]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]

    x_0 = list(filter(lambda x: x[-1] == 0.0, data))
    x_1 = list(filter(lambda x: x[-1] == 1.0, data))

    x_0_0 = [i[0] for i in x_0]
    x_0_1 = [i[1] for i in x_0]

    x_1_0 = [i[0] for i in x_1]
    x_1_1 = [i[1] for i in x_1]

    plt.plot(x_0_0, x_0_1, 'ro')
    plt.plot(x_1_0, x_1_1, 'bo')
    plt.show()


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sigmoid(x)
        return x


model = LogisticRegression()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# num_epochs = 1000
# for epoch in range(num_epochs):
#
#     inputs = x_train
#     target = y_train
#
#     # forward
#     out = model(inputs)
#     loss = criterion(out, target)
#
#     # backward
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 20 == 0:
#         print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1, num_epochs, loss.data[0]))

