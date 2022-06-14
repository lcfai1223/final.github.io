from lib2to3.pytree import convert
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import vgg16
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from torchsummary import summary
import torch
import torchvision
import matplotlib.image as mpimg
from PIL import Image
from torchvision import transforms

classes = ['plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def show_params():
    batch_size, lr = vgg16.get_params()
    print("Batch size: %d" % batch_size)
    print("Learning rate: %f" % lr)
    print("Optimizer: SGD")

def getPxImg(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.resize(img, (800, 800), interpolation= cv2.INTER_LINEAR)

    height, width, channel = img.shape
    bytesPerLine = 3 * width
    qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)

def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

def show_train_image(label):
    data = load_file('./data/cifar-10-batches-py/data_batch_1')
    image_data = data['data']
    labels = data['labels']
    label_count = len(labels)
    picture_data = image_data.reshape(-1,3,32,32)
    picture_data = picture_data.transpose(0,2,3,1)
    show_images_nums = 10


    for i in range(2, 3):  #只挑出 batch_2 的圖片們
        batch_path = './data/cifar-10-batches-py/data_batch_' + str(i)
        data = load_file(batch_path)
        image_data = data['data']
        new_labels = data['labels']
        new_picture_data = image_data.reshape(-1,3,32,32)
        new_picture_data = new_picture_data.transpose(0,2,3,1)
        picture_data = np.append(picture_data, new_picture_data, axis=0)
        labels = np.append(labels, new_labels, axis=0)

    random_list = []
    train_len = len(picture_data)
    for i in range(show_images_nums):
        n = random.randint(1,train_len)  # 隨機從 batch 裏面挑出 10 張照片，並且加到 list 裡
        if not n in random_list:
            random_list.append(n)

    row = 1
    fig,a =  plt.subplots(2,int(show_images_nums/2))
    for idx, rand_num in enumerate(random_list):
        if(idx == show_images_nums/2):
            row += 1
        col = (idx)%5
        a[row-1][col].imshow(picture_data[rand_num])
        a[row-1][col].set_title(classes[labels[rand_num]])
    plt.savefig("tmp.png")
    img = cv2.imread('tmp.png')
    qImg = getPxImg(img)
    label.setPixmap(qImg)
    plt.close(fig)

model = vgg16.load_model() # load 出已經訓練好的 model
# 如果支援 gpu 那就用 gpu 否則 cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def test_image(label):
    # try:
    #     index = int(index)
    # except ValueError:
    #     # Handle the exception
    #     print('Please enter an integer')
    #     return 
    # if index < 0 or index > 10000:
    #     print("wrong index range")
    #     return

    # print("\n... Testing ...")
    # print("Test Image: %d" % index)
    # print("finish loading model")
    # 這邊是等待 user 將圖片上傳進來後做測試
    test_dataset = vgg16.get_dataset()
    # print("finish get loader")

    # 將進來的圖片的 size 都變成 32x32，目的是為了要能送入 model
    # np_frame = np.array(Image.open("./test/test.png"))
    transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.PILToTensor()
    ])
    
    # user 上傳的圖片會先存到 ./test/test.png，之後 model 再從這個位置去拿圖片
    origin_images = Image.open("./test/test.png")
    images = transform(Image.open("./test/test.png"))
    # images = img.permute(1, 2, 0)
    # total = 0
    # correct = 0
    # print(testloader.__len__())



    with torch.no_grad():
        # images, labels = test_dataset.__getitem__(0)
        # print(images.shape)
        
        # 這邊固定使用 CPU
        images = images.to(device)
        
        # 因為單一圖片只有三個維度，但是 model 需要 4 個維度 (batch, w, h, channel)
        
        # 而因為每次只要測一張圖片所以 batch=1
        images = np.expand_dims(images, axis=0)
        images = torch.from_numpy(images)
        output = model(images.float())
        m = nn.Softmax(dim=1)
        output = m(output)
        output = output.cpu().numpy()[0]
        # Make a fake dataset:
        y_pos = np.arange(len(classes))
        
        # Create bars
        # 畫出每個 class 中 model 所預測的機率
        plt.bar(y_pos, output)
        
        # Create names on the x-axis
        plt.xticks(y_pos, classes)
        
        # Show graphic
        # plt.imshow(images.cpu()[0].permute(1, 2, 0))
        plt.savefig("tmp.png")
        img = cv2.imread('tmp.png')
        qImg = getPxImg(img)
        label.setPixmap(qImg)
        # img
        # plt.show()
        plt.cla()
        # fig1, ax1 = plt.subplots( dpi=80)
        plt.imshow(origin_images)
        plt.show()


# 顯示預先存放好的 accuracy/loss 圖片
def show_accuracy(label):
    acc = Image.open('./result/accuracy.png')
    loss = Image.open('./result/loss.png')
    fig, ax = plt.subplots(2, 1, figsize=(5,6))
    # imgplot = plt.imshow(acc)
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])
    ax[0].imshow(acc)
    ax[1].imshow(loss)
    # fig1, ax1 = plt.subplots()
    for i in range(2):
        ax[i].get_xaxis().set_ticks([])
        ax[i].get_yaxis().set_ticks([])
    plt.savefig("tmp.png")
    img = cv2.imread('tmp.png')
    qImg = getPxImg(img)
    label.setPixmap(qImg)

    # plt.show()