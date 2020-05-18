#This is a simple cnn-network similar to vgg.
#It consists of 5 3*3 convolutional layers and 3 fully connected layers.
#After running 30 epochs with Adam, the accuracy rate on the vilidatin set of hw3 data set is about 60%.
#ps: hw3 data set is a 11-category food picture classification problem.
#It is small and run very fast, but basically does not work. 

#类似VGG架构的CNN简单网络
#由5个3*3卷积层和3个全连接层组成
#用Adam运行30个epoch之后hw3数据集的验证集上准确率大约为60%
#网络较小运行较快，但是基本不work

import os
import time
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class MyDataset(data.Dataset):

    def __init__(self, path, transform=None, label_transform=None):
        self.path = path
        self.files = os.listdir(path)
        self.transform, self.label_transform = transform, label_transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path, self.files[index])).convert('RGB')
        label = int(self.files[index].split('_')[0])
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return img, label
 
    def __len__(self):
        return len(self.files)

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



if __name__ == '__main__':

    cwd = '/home/aistudio/data/data35121'
    #cwd = '/kaggle/input/hw3cnn'
    #cwd = os.getcwd()
    num_classes = 11
    num_epoch = 30
    img_size = 128
    batch_size = 128
    #num_workers = 4

    transform_train = transforms.Compose([transforms.Resize([img_size, img_size]), 
                                          transforms.RandomHorizontalFlip(), 
                                          transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose([transforms.Resize([img_size, img_size]), 
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
       
    train_set = MyDataset(path=os.path.join(cwd, 'hw3', 'train'), transform=transform_train)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = MyDataset(path=os.path.join(cwd, 'hw3', 'valid'), transform=transform_test)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    #testset = MyDataset(path=os.path.join(cwd, 'hw3', 'test'), transform=transform_test)
    #testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    #get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print(' '.join('%5s' % labels[j] for j in range(batch_size)))
    imshow(torchvision.utils.make_grid(images))



    model = Classifier(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if(torch.cuda.is_available()):
        model = model.cuda()
        print("gpu enabled")
    else:
        print("gpu disabled")

    for epoch in range(num_epoch):  # loop over the dataset multiple times
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            train_pred = model(data[0].cuda())
            loss = criterion(train_pred, data[1].cuda())
            loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss  += loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].cuda())
                batch_loss = criterion(val_pred, data[1].cuda())

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

            #將結果 print 出來
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % (epoch + 1, num_epoch, time.time()-epoch_start_time, train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

    print('Finished Training')

    #PATH = './cifar_net.pth'
    #torch.save(net.state_dict(), PATH)

    pass
