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

class VGG16Net(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Net, self).__init__()
        self.num_classes = num_classes

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        #maxpool1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        #maxpool2
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        #maxpool3

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #maxpool4

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #maxpool5

        self.fc1 = nn.Linear(in_features=7 * 7 * 512, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))
        x = self.pool(F.relu(self.conv2_2(F.relu(self.conv2_1(x)))))
        x = self.pool(F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(x)))))))
        x = self.pool(F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(x)))))))
        x = self.pool(F.relu(self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(x)))))))
        x = F.softmax(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x.view(-1, 7 * 7 * 512))))))))
        return x
    
class VGG13Net(nn.Module):
    def __init__(self, num_classes):
        super(VGG13Net, self).__init__()
        self.num_classes = num_classes

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        #maxpool1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        #maxpool2
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        #maxpool3

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #maxpool4

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #maxpool5

        self.fc1 = nn.Linear(in_features=7 * 7 * 512, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))
        x = self.pool(F.relu(self.conv2_2(F.relu(self.conv2_1(x)))))
        x = self.pool(F.relu(self.conv3_2(F.relu(self.conv3_1(x)))))
        x = self.pool(F.relu(self.conv4_2(F.relu(self.conv4_1(x)))))
        x = self.pool(F.relu(self.conv5_2(F.relu(self.conv5_1(x)))))
        x = F.softmax(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x.view(-1, 7 * 7 * 512))))))))
        return x
    

class VGG11Net(nn.Module):
    def __init__(self, num_classes):
        super(VGG11Net, self).__init__()
        self.num_classes = num_classes

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        #maxpool1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        #maxpool2
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        #maxpool3

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #maxpool4

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #maxpool5

        self.fc1 = nn.Linear(in_features=7 * 7 * 512, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_1(x)))
        x = self.pool(F.relu(self.conv2_1(x)))
        x = self.pool(F.relu(self.conv3_2(F.relu(self.conv3_1(x)))))
        x = self.pool(F.relu(self.conv4_2(F.relu(self.conv4_1(x)))))
        x = self.pool(F.relu(self.conv5_2(F.relu(self.conv5_1(x)))))
        x = F.softmax(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x.view(-1, 7 * 7 * 512))))))))
        return x

class VGG9Net(nn.Module):
    def __init__(self, num_classes):
        super(VGG9Net, self).__init__()
        self.num_classes = num_classes

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        #maxpool1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        #maxpool2
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        #maxpool3

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        #maxpool4

        self.fc1 = nn.Linear(in_features=7 * 7 * 256, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_1(x)))
        x = self.pool(F.relu(self.conv2_1(x)))
        x = self.pool(F.relu(self.conv3_2(F.relu(self.conv3_1(x)))))
        x = self.pool(F.relu(self.conv4_2(F.relu(self.conv4_1(x)))))
        x = F.softmax(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x.view(-1, 7 * 7 * 256))))))))
        return x



class Le5Net(nn.Module):
    def __init__(self, num_classes):
        super(Le5Net, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        #maxpool1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        #maxpool2

        self.fc1 = nn.Linear(in_features=5 * 5 * 16, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1((x))))
        x = self.pool(F.relu(self.conv2((x))))
        x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x.view(-1, 5 * 5 * 16))))))
        return x


class Le7Net(nn.Module):
    def __init__(self, num_classes):
        super(Le7Net, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        #maxpool1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        #maxpool2

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        #maxpool3

        self.fc1 = nn.Linear(in_features=11 * 11 * 64, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1((x))))
        x = self.pool(F.relu(self.conv2((x))))
        x = self.pool(F.relu(self.conv3((x))))
        x = F.softmax(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x.view(-1, 11 * 11 * 64))))))))
        return x

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
    #num_workers = 0
    num_workers = 4

    transform_train = transforms.Compose([transforms.Resize([img_size, img_size]), 
                                          transforms.RandomHorizontalFlip(), 
                                          transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose([transforms.Resize([img_size, img_size]), 
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    #transform_train = transforms.Compose([transforms.Resize([img_size, img_size]), 
    #                                      transforms.RandomResizedCrop(116), 
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    #transform_test = transforms.Compose([transforms.Resize([img_size, img_size]), 
    #                                      transforms.RandomResizedCrop(116), 
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_set = MyDataset(path=os.path.join(cwd, 'hw3', 'train'), transform=transform_train)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = MyDataset(path=os.path.join(cwd, 'hw3', 'valid'), transform=transform_test)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    #train_set = MyDataset(path=os.path.join(cwd, 'hw3', 'train'), transform=transform_train)
    #train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    #val_set = MyDataset(path=os.path.join(cwd, 'hw3', 'valid'), transform=transform_test)
    #val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

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

    

    #dataiter = iter(trainloader)
    #images, labels = dataiter.next()

    ## print images
    #imshow(torchvision.utils.make_grid(images))
    #print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(batch_size)))

    #net = Le5Net(num_classes)
    #net.load_state_dict(torch.load(PATH))

    #outputs = net(images)

    #_, predicted = torch.max(outputs, 1)

    #print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(batch_size)))

    #correct = 0
    #total = 0
    #with torch.no_grad():
    #    for data in trainloader:
    #        images, labels = data
    #        outputs = net(images)
    #        _, predicted = torch.max(outputs.data, 1)
    #        total += labels.size(0)
    #        correct += (predicted == labels).sum().item()

    #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    #class_correct = list(0. for i in range(num_classes))
    #class_total = list(0. for i in range(num_classes))
    #with torch.no_grad():
    #    for data in trainloader:
    #        images, labels = data
    #        outputs = net(images)
    #        _, predicted = torch.max(outputs, 1)
    #        c = (predicted == labels).squeeze()
    #        for i in range(len(labels)):
    #            label = labels[i]
    #            class_correct[label] += c[i].item()
    #            class_total[label] += 1


    #for i in range(num_classes):
    #    print('Accuracy of %5s : %2d %%' % (
    #        i, 100 * class_correct[i] / class_total[i]))



    #dataiter = iter(validloader)
    #images, labels = dataiter.next()

    ## print images
    #imshow(torchvision.utils.make_grid(images))
    #print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(batch_size)))

    #net = Le7Net(num_classes)
    #net.load_state_dict(torch.load(PATH))

    #outputs = net(images)

    #_, predicted = torch.max(outputs, 1)

    #print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(batch_size)))

    #correct = 0
    #total = 0
    #with torch.no_grad():
    #    for data in validloader:
    #        images, labels = data
    #        outputs = net(images)
    #        _, predicted = torch.max(outputs.data, 1)
    #        total += labels.size(0)
    #        correct += (predicted == labels).sum().item()

    #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    #class_correct = list(0. for i in range(num_classes))
    #class_total = list(0. for i in range(num_classes))
    #with torch.no_grad():
    #    for data in validloader:
    #        images, labels = data
    #        outputs = net(images)
    #        _, predicted = torch.max(outputs, 1)
    #        c = (predicted == labels).squeeze()
    #        for i in range(len(labels)):
    #            label = labels[i]
    #            class_correct[label] += c[i].item()
    #            class_total[label] += 1


    #for i in range(num_classes):
    #    print('Accuracy of %5s : %2d %%' % (
    #        i, 100 * class_correct[i] / class_total[i]))



    ##device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Assuming that we are on a CUDA machine, this should print a CUDA device:

    ##print(device)

    ##net.to(device)

    ##inputs, labels = data[0].to(device), data[1].to(device)

    pass
