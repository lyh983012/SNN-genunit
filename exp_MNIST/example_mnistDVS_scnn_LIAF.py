# -*- coding: utf-8 -*-
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 example_mnistDVS_scnn_LIAF.py
from __future__ import print_function
import sys
sys.path.append("..")
import LIAF
from LIAFnet.LIAFResNet import *
from datasets.dvs_mnist import DVS_MNIST_Dataset

import torch.distributed as dist 
import torch.nn as nn

import argparse, pickle, time, os

from importlib import import_module
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from importlib import import_module

##################### Step1. Env Preparation #####################

writer = None #仅在master进程上输出
master = False 
save_folder = 'MNISTDVS_LIAF'

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank',type = int,default=0)
args = parser.parse_args()

torch.distributed.init_process_group('nccl',init_method='env://')
local_rank = dist.get_rank()
world_size = dist.get_world_size()

device = torch.device('cuda:'+str(local_rank))
torch.cuda.set_device(local_rank)
print(local_rank,' is ready')

##################### Step2. load in dataset #####################

workpath = os.path.abspath(os.getcwd())

num_classes = 10
batch_size  = 4

train_dataset = DVS_MNIST_Dataset(mode='train',scale = 4,data_set_path='/data/MNIST_DVS_mat/',timestep = 50)
test_dataset = DVS_MNIST_Dataset(mode='test',scale = 4,data_set_path='/data/MNIST_DVS_mat/',timestep = 50)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
test_sampler  = torch.utils.data.distributed.DistributedSampler(test_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,drop_last=True,sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)



best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
timeWindows= 50
learning_rate = 1e-2
num_epochs = 30         # max epoch
modules = import_module('LIAFnet.LIAFCNN')
config = modules.Config()
config.cfgFc =[10]
config.cfgCnn=[(2, 64, 5, 2, 1 ,False),(64, 64, 5, 2, 1 ,True),(64, 128, 5, 2, 1 ,True)]
config.decay = 0.5
config.dropOut= 0
config.timeWindows = timeWindows
config.actFun=torch.selu
#config.actFun=LIAF.LIFactFun.apply 
config.useBatchNorm= True
config.useLayerNorm= False
config.useThreshFiring = False
config.padding=0
config.dataSize=[128,128]
snn = modules.LIAFCNN(config).to(device)
snn.to(device)
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in snn.parameters())))
snn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(snn)

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


with torch.cuda.device(local_rank):
    snn = DDP(snn.cuda(),device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    

criterion = nn.CrossEntropyLoss()

###################################################################################    
if local_rank == 0 :
    writer = SummaryWriter(comment='runs/'+save_folder)
    master = True
    print('start recording')
    
training_iter = 0
for epoch in range(num_epochs):
    #training
    snn.train(mode=True)
    running_loss = 0
    start_time = time.time()
    correct = 0
    total = 0
    train_loader.sampler.set_epoch(epoch)
    for i, (images, labels) in enumerate(train_loader):
        training_iter +=1
        snn.zero_grad()
        optimizer.zero_grad()
        images = images.float().to(device)
        outputs = snn(images)
        loss = criterion(outputs.cpu(), labels)
        
        _, predicted = outputs.cpu().max(1)
        total += float(labels.size(0))
        correct += float(predicted.eq(labels).sum().item())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (i+1)%5 == 0 and master:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
            print('acc:',correct / total *100)
            
            print('Time elasped:', time.time()-start_time)
            writer.add_scalar('Loss_th', running_loss, training_iter)
            writer.add_scalar('train_acc', correct / total *100, training_iter)
            correct = 0
            total = 0
            running_loss = 0
            
    correct = 0
    total = 0
    lr_scheduler.step()
   #evaluation
    snn.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            optimizer.zero_grad()
            outputs = snn(inputs)
            loss = criterion(outputs.cpu(), labels)
            _, predicted = outputs.cpu().max(1)
            total += float(labels.size(0))
            correct += float(predicted.eq(labels).sum().item())
    if master:
        print('Iters:', epoch,'\n\n\n')
        print('Test Accuracy of the model on the 2000 test images: %.3f' % (100 * correct / total))
        acc = 100. * float(correct) / float(total)
        if acc > best_acc:
            best_acc = acc
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)
            print('===> Saving models...')
            torch.save(snn.state_dict(), workpath+'/'+save_folder+'/'+str(int(best_acc))+'MNISTDVS_LIAF.t7')
            print('===> Saved')
            print("best:",best_acc)
        writer.add_scalar('accuracy', acc, epoch)


