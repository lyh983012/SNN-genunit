# -*- coding: utf-8 -*-
#测试：量化模式2是否能够等同LIF
#测试：同样参数下SCNN结构对MNIST的拟合

from __future__ import print_function
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
import time
import LIAF_module
import util
import LIAF

#current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
#my_writer = SummaryWriter(log_dir='exp1/'+current_time)
#tensorboard --log-dir={/Users/lyh/Desktop/finalproject}

device = LIAF.device
names = 'spiking_model'
data_path = '~/dataset/raw/'

decay = 0.3 # decay constants
num_classes = 10
batch_size  = 100
learning_rate = 1e-3
num_epochs = 50 # max epoch

Qbit = int(input('plz input Qbits:'))
print(Qbit)
train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])
timeWindows=5

cfg_cnn = [(1, 8, 7, 1, 1 ,False),
           (8, 32, 5, 2, 1 ,True),
           ]
cfg_fc = [1024, 10] #使用BN后达到性能的大大提升

actfun = LIAF.LIFactFun.apply
snn = LIAF_module.LIAFCNN(cfg_cnn,
                   cfg_fc,
                   actFun=actfun,
                   padding=0,
                   timeWindows=timeWindows,
                   dropOut=0,
                   decay = 0.3,
                   dataSize=28,
                   useBatchNorm=True,
                   useThreshFiring=False,
                   Qbit=Qbit)

snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)



for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):

        #images = images.view(batch_size, 28*28)
        snn.zero_grad()
        optimizer.zero_grad()
        images = images.float().to(device)
        Is=images.size()
        input = torch.zeros(Is[0],Is[1],Is[2],Is[3],timeWindows)
        for j in range(timeWindows):
            input[:,:,:,:,j]=images

        # for timestep : repeat(select the image, ctrl+c, ctrl+v)--> image sequence! amazing(x
        outputs = snn(input)
        labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (i+1)%100 == 0:
             print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
             running_loss = 0
             print('Time elasped:', time.time()-start_time)
    correct = 0
    total = 0
    optimizer = util.lr_scheduler(optimizer, epoch, 10)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            Is = images.size()
            input = torch.zeros(Is[0], Is[1], Is[2], Is[3], timeWindows)
            for i in range(timeWindows):
                input[:, :, :, :, i] = inputs
            outputs = snn(input)

            labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if batch_idx %100 ==0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

    print('Iters:', epoch,'\n\n\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    if acc > best_acc:
        best_acc = acc
        # save
        #torch.save(snn, modelpath+'model_' + str(int(best_acc * 100)) + '.pkl')
        # load
        # snn = torch.load('\model.pkl')
    print('best=', best_acc)

print(acc_record)