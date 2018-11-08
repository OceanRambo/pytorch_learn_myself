import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


flag = False
if torch.cuda.is_available():
    flag = True
# flag = False

w = torch.Tensor([0.5, 3]).unsqueeze(1)
b = torch.Tensor([0.9])
# y = 0.9 + 0.5*x + 3*x^2


class PolyRegression(nn.Module):
    def __init__(self):
        super(PolyRegression, self).__init__()
        self.poly = nn.Linear(2, 1)

    def forward(self, x):
        out = self.poly(x)
        return out


def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 3)], 1)


def get_bach(batch_size=32):
    random = torch.randn(batch_size)
    random = np.sort(random)
    random = torch.Tensor(random)
    x = make_features(random)
    y = x.mm(w) + b[0]
    if flag:
        return x.cuda(), y.cuda()
    else:
        return x, y


if flag:
    model = PolyRegression().cuda()
else:
    model = PolyRegression()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

num_epochs = 0
while True:
    x_train, y_train = get_bach()

    # forward
    output = model(x_train)
    loss = criterion(output, y_train)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (num_epochs + 1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(num_epochs+1, num_epochs, loss.item()))
    num_epochs += 1
    if loss.item() < 1e-3:
        break

print("==> Learned function: y = {:.2f} + {:.2f}*x + {:.2f}*x^2".format(model.poly.bias[0],
                                                                        model.poly.weight[0][0],
                                                                        model.poly.weight[0][1]))
print("==> Actual function: y = {:.2f} + {:.2f}*x + {:.2f}*x^2".format(b[0], w[0][0], w[1][0]))

model.eval()
if flag:
    x_train = x_train.cuda()
    predict = model(x_train)
    predict = predict.cpu().data.numpy()
    plt.plot(x_train.cpu().numpy()[:, 0], y_train.cpu().numpy(), 'ro')
    plt.plot(x_train.cpu().numpy()[:, 0], predict, 'b')
    plt.legend(['Actual', 'Learned'])
    plt.show()

else:
    predict = model(x_train)
    predict = predict.data.numpy()
    plt.plot(x_train.numpy()[:, 0], y_train.numpy(), 'ro')
    plt.plot(x_train.numpy()[:, 0], predict, 'b')
    plt.legend(['Actual', 'Learned'])
    plt.show()
