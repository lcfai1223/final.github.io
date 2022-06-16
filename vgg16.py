
import torch
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 4
LR = 0.001
EPOCH = 20
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

ten_transform = transforms.Compose(
    [transforms.ToTensor()])

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def get_params():
    return BATCH_SIZE, LR

import torch
import torch.nn as nn

#M為池化層，數字為輸出
_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

#自定義每一層卷積層和池化層
#kernal_size卷積層的尺寸
#步幅stride
# 填充方式padding
#Conv2d卷積層计算
##ReLU激活函数           
# MaxPool2d池化層：最大池化
def _make_layers(cfg):
    layers = []
    in_channels = 3
    for layer_cfg in cfg:
        if layer_cfg == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=layer_cfg,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True))
            layers.append(nn.BatchNorm2d(num_features=layer_cfg))
            layers.append(nn.ReLU(inplace=True))
            in_channels = layer_cfg
    return nn.Sequential(*layers)

class VGG(nn.Module):
    """
    VGG module for 3x32x32 input, 10 classes
    """

    def __init__(self):
        super(VGG, self).__init__()
        cfg = _cfg['VGG16']
        self.layers = _make_layers(cfg)
        flatten_features = 512
        self.fc1 = nn.Linear(flatten_features, 10)#fully connected layer that outputs our 10 labels
        # self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        
        y = self.layers(x)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        # y = self.fc2(y)
        # y = self.fc3(y)
        return y

train_acc = []
test_acc = []
train_loss = []
def run():
    # 將 dataset 下載下來
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # 將 dataset 製作成 dataloader 方便 model 取資料
    #訓練數據加載器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2)
    net = VGG()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
#定義損失數以及優化方法
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
#訓練
    print('start traning.')
    train_len = len(trainloader)
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        # 跑過整個 dataset EPOCH 次
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 拿到要訓練的 inputs 以及 labels
            inputs, labels = data
            # 將 inputs 以及 labels 丟入 cpu 或 gpu 中
            inputs, labels = inputs.to(device), labels.to(device)

            # 先將過去 model 更新過的參數數值清空
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # CrossEntropyLoss 是 loss function，適合用在分類問題
            loss = criterion(outputs, labels)
            # 做 back propagation
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print('[ %.2f % ] loss: %.3f' %
            #           (100*(i/train_len), running_loss / 2000))
            #     running_loss = 0.0
        #准確度
        correct = 0
        total = 0
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('train: %d %%' % (
            100 * correct / total))
        train_acc.append(correct / total)
        print("loss: %f", running_loss/train_len)
        train_loss.append(running_loss/train_len)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('test: %d %%' % (
            100 * correct / total))
        test_acc.append(correct / total)
    print('Finished Training')

    with open('train_acc.txt', 'w') as f:
        for item in train_acc:
            f.write("%s\n" % item)

    with open('test_acc.txt', 'w') as f:
        for item in test_acc:
            f.write("%s\n" % item)

    with open('train_loss.txt', 'w') as f:
        for item in train_loss:
            f.write("%s\n" % item)

    torch.save(net.state_dict(), 'VGG16.pt')


def load_model():
    net = VGG()
    net.load_state_dict(torch.load("result/vgg16.pt", map_location=torch.device('cpu')))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    return net

def get_loader():
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                    download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                        shuffle=False, num_workers=2)
    return testloader

def get_dataset():
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                            download=True, transform=ten_transform)
    return testset
# run()
# get_dataset()
