#This is a resnet50 cnn-network with pre-trained model.
#After running about 60 epochs with Adam, the accuracy rate on the vilidatin set of hw3 data set is ouver 80%.
#ps: hw3 data set is a 11-category food picture classification problem.

# run env command
# source ~/work/*conda3/bin/activate

import os
import time
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class MyDataset(data.Dataset):

    def __init__(self, root, train=False, test=False, transform=None, label_transform=None):
        self.root = root
        self.files = os.listdir(root)
        self.train, self.test, self.transform, self.label_transform = train, test, transform, label_transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.files[index])).convert('RGB')
        if self.train:
            label = int(self.files[index].split('_')[0])
        if self.test:
            label = int(self.files[index].split('.')[0])
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return img, label
 
    def __len__(self):
        return len(self.files)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    #cwd = '/home/aistudio/data/data35342/hw3'
    #cwd = '/kaggle/input/hw3cnn'
    cwd = os.getcwd() + '/hw3'
    num_classes = 11
    num_epoch = 1000
    img_size = 256
    img_size_in = 224
    batch_size = 128
    #num_workers = 0
    num_workers = 0

    transform_train = transforms.Compose([transforms.Resize([img_size, img_size]), 
                                          transforms.RandomResizedCrop(img_size_in), 
                                          transforms.RandomHorizontalFlip(), 
                                          transforms.RandomRotation(15), 
                                          transforms.ToTensor(), 
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])])

    transform_test = transforms.Compose([transforms.Resize([img_size_in, img_size_in]), 
                                          transforms.ToTensor(), 
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])])
        
    #train_set = torchvision.datasets.ImageFolder(root=os.path.join(cwd, 'train'), transform=transform_train)
    #train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
 
    #val_set = torchvision.datasets.ImageFolder(root=os.path.join(cwd, 'valid'), transform=transform_test)
    #val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)  
    
    train_set = torchvision.datasets.ImageFolder(root=os.path.join(cwd, 'train'), transform=transform_train)
    val_set = torchvision.datasets.ImageFolder(root=os.path.join(cwd, 'valid'), transform=transform_test)
    train_set = train_set + val_set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)  

    test_set = MyDataset(root=os.path.join(cwd, 'test/nan'), test=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    


    #get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print(' '.join('%5s' % labels[j] for j in range(batch_size)))
    imshow(torchvision.utils.make_grid(images))
    #


    #model = models.resnet50()
    #model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'resnet50.pth')))
    ##model.load_state_dict(torch.load('/home/aistudio/data/data35721/resnet50.pth'))
    #fc_features = model.fc.in_features
    #model.fc = nn.Linear(fc_features, num_classes)
    #for param in model.parameters():
    #    param.requires_grad = False
    #model.fc.weight.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    ##optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    model = models.resnet50()
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'resnet50.pth')))
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'cifar_net.pth')))

    if(torch.cuda.is_available()):
        model = model.cuda()
        print("gpu enabled")
    else:
        print("gpu disabled")

    #num_epoch_train1 = 10
    num_epoch_train1 = 0

    for epoch in range(num_epoch_train1):  # loop over the dataset multiple times
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        #model.train()
        #for i, data in enumerate(train_loader):
        #    optimizer.zero_grad()
        #    train_pred = model(data[0].cuda())
        #    loss = criterion(train_pred, data[1].cuda())
        #    loss.backward()
        #    optimizer.step()

        #    train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        #    train_loss  += loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                #val_pred = model(data[0].cuda())
                val_pred = model(data[0])
                #batch_loss = criterion(val_pred, data[1].cuda())
                batch_loss = criterion(val_pred, data[1])
                
                #val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_acc += np.sum(np.argmax(val_pred.data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

            #將結果 print 出來
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % (epoch + 1, num_epoch, time.time()-epoch_start_time, train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

      
            
    #for param in model.layer4.parameters():
    #    param.requires_grad = True
    ##model.layer4.weight.requires_grad = True
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    ##num_epoch_finetune = 20
    num_epoch_finetune = 0

    for epoch in range(num_epoch_train1, num_epoch_finetune):  # loop over the dataset multiple times
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

        #model.eval()
        #with torch.no_grad():
        #    for i, data in enumerate(val_loader):
        #        val_pred = model(data[0].cuda())
        #        batch_loss = criterion(val_pred, data[1].cuda())

        #        val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        #        val_loss += batch_loss.item()

        #    #將結果 print 出來
        #    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % (epoch + 1, num_epoch, time.time()-epoch_start_time, train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))


    print('Finished Training')

    #PATH = './cifar_net.pth'
    #torch.save(net.state_dict(), PATH)

    model.eval()
    pred = np.zeros(2)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
        #for i, data in enumerate(val_loader):
            #test_pred = model(data[0].cuda())
            #pred = np.vstack((pred, np.hstack((np.expand_dims(np.argmax(test_pred.cpu().data.numpy(), axis=1), axis=1), np.expand_dims(data[1].numpy(), axis=1)))))
            test_pred = model(data[0])
            pred = np.vstack((pred, np.hstack((np.expand_dims(np.argmax(test_pred.data.numpy(), axis=1), axis=1), np.expand_dims(data[1].numpy(), axis=1)))))
            print(' '.join('%5s' % data[1][j] for j in range(16)))
            tmp = np.expand_dims(np.argmax(test_pred.data.numpy(), axis=1), axis=1)
            print(' '.join('%5s' % tmp[j] for j in range(16)))
            imshow(torchvision.utils.make_grid(data[0]))
            pass
        np.savetxt("pred.csv", pred, delimiter=",")

    pass