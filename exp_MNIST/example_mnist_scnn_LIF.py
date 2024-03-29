# -*- coding: utf-8 -*-
#CUDA_VISIBLE_DEVICES=1 python example_mnist_scnn.py
from __future__ import print_function
import sys
sys.path.append("..")
from util.util import lr_scheduler
import LIAF
from LIAFnet.LIAFResNet import *

import torch.distributed as dist 
import torch.nn as nn
import argparse, pickle, torch, time, os,sys
from importlib import import_module
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from importlib import import_module
import torchvision
import torch
import torchvision.transforms as transforms

save_folder = 'MNIST_LIF'
writer = SummaryWriter(comment='../runs/'+save_folder)

#todo: input your data path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
names = 'spiking_model'
data_path = '/data/MNIST/'

workpath = os.path.abspath(os.getcwd())

num_classes = 10
batch_size  = 100

train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
loss_train_list = []
loss_test_list = []
acc_train_list = []
acc_test_list = []

timeWindows=10
learning_rate = 1e-2

num_epochs = 30         # max epoch
modules = import_module('LIAFnet.LIAFCNN')
config = modules.Config()
config.cfgFc =[10]
config.cfgCnn=[(1, 64, 3, 1, 1 ,False),(64, 64, 3, 2, 1 ,True),(64, 128, 3, 2, 1 ,True)]
config.decay = 0.5
config.dropOut= 0
config.timeWindows = timeWindows

#config.actFun=torch.selu
config.actFun=LIAF.LIFactFun.apply 

config.useBatchNorm= True
config.useLayerNorm= False
config.useThreshFiring = False
config.padding=0
config.dataSize=[28,28]

snn = modules.LIAFCNN(config).to(device)
snn.to(device)
criterion = nn.CrossEntropyLoss()
######################################################################################
#note:
#CorssEntrophyLoss适用于分类问题（其为Max函数的连续近似）
#它的输入是output（每一类别的概率）和label（第几个类别）
######################################################################################
optimizer = torch.optim.SGD(snn.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=True)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, 
                    milestones=[10,20], 
                    gamma=0.1, 
                    last_epoch=-1)



for epoch in range(num_epochs):
    #training
    snn.train(mode=True)
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        snn.zero_grad()
        optimizer.zero_grad()
        images = images.float().to(device)

        Is=images.size()
        input = torch.zeros(Is[0], Is[1],timeWindows, Is[2], Is[3])
        for j in range(timeWindows):
            input[:,:,j,:,:]=images

        outputs = snn(input)
        loss = criterion(outputs.cpu(), labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (i+1)%100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
            loss_train_list.append(running_loss)
            running_loss = 0
            
            print('Time elasped:', time.time()-start_time)
    correct = 0
    total = 0
    lr_scheduler.step()
    
   #evaluation
    snn.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()

            Is = images.size()
            input = torch.zeros(Is[0], Is[1],timeWindows, Is[2], Is[3])
            for j in range(timeWindows):
                input[:,:,j, :, :] = inputs
            outputs = snn(input)
            
            loss = criterion(outputs.cpu(), labels)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
    print('Iters:', epoch,'\n\n\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    if acc > best_acc:
        best_acc = acc
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        print('===> Saving models...')
        torch.save(snn.state_dict(), workpath+'/'+save_folder+'/'+str(int(best_acc))+'LIFmodelsaved.t7')
        print('===> Saved')
        print("best:",best_acc)
    writer.add_scalar('accuracy', acc, epoch)
    

