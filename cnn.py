"""
**********卷积神经网络实现 MNIST 手写数字分类*******

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import time


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, 3),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, 3),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2))

        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, 3),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, 3),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(nn.Linear(128*4*4, 1024),
                                nn.ReLU(inplace=True),
                                nn.Linear(1024, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 10))

    def forward(self, x):               # batch_size*1*28*28
        x = self.layer1(x)              # batch_size*16*26*26
        x = self.layer2(x)              # batch_size*32*12*12
        x = self.layer3(x)              # batch_size*64*10*10
        x = self.layer4(x)              # batch_size*128*4*4
        x = x.view(x.size(0), -1)       # batch_size*(128*4*4)
        x = self.fc(x)                  # batch_size*10
        return x


# 定义超参数
batch_size = 64
learning_rate = 1e-2
num_epochs = 5

# 数据预处理：标准化
# transforms.ToTensor(): 将图片转换成PyTorch中处理的对象Tensor,
# 在转换的过程中自动将图片标准化了， Tensor的范围是0~1。
# transforms.Normalize(): 第一个参数是均值，第二个参数是方差，做的处理就是减均值，除以方差。
# 参数中的数值表示每个通道对应的均值和方差
# transforms.Compose()将各种预处理操作组合到一起
data_tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.5], [0.5])])


# 读取数据集： 如果data/mnist文件夹下存在该数据集，直接读取， 不存在则先下载后读取
train_set = datasets.MNIST(root='./data/mnist', train=True, transform=data_tf, download=True)
test_set = datasets.MNIST(root='./data/mnist', train=False, transform=data_tf, download=True)

# 建立数据迭代器
train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_set, batch_size=batch_size)   # 默认shuffle=False

# 导入模型，定义损失函数和优化方法
# 输入图片大小是 28 x 28， 有10个分类
model = Cnn()
# print(model)   查看网络结构
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练网络
start = time.time()
train_loss = 0
for epoch in range(num_epochs):
    # print('current epoch = %d' % epoch)
    for i, (train_img, label) in enumerate(train_data):
        out = model(train_img)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if (i+1) % 100 == 0:
            print('epoch: {}/{}, Iter: {}/{},Loss:{:.4f}'.format(epoch+1, num_epochs, i+1, len(train_data), loss.item()))
            # print('current loss = %.5f' % loss.item())

time_elapsed = time.time() - start
print('Training complete in {:.0f}s'.format(time_elapsed % 60))

# 测试
model.eval()
eval_loss = 0
eval_acc = 0
total = 0
# with torch.no_grad():
for test_img, label in test_data:
    out = model(test_img)
    loss = criterion(out, label)

    eval_loss += loss.item()
    _, predicted = torch.max(out, 1)
    total += label.size(0)
    num_correct = (predicted == label).sum()
    eval_acc += num_correct

print('Accuracy of the network on test images: %d %%' % (100 * eval_acc / total))
