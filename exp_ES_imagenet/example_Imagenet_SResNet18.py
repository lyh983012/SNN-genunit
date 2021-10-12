# cd SNN_work/genunit_opt/exp_ES_imagenet
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6  python -m torch.distributed.launch --nproc_per_node=7 example_Imagenet_SResNet18.py

from __future__ import print_function
import argparse, pickle, torch, time, os,sys,math
sys.path.append("..")
import LIAF
from LIAFnet.LIAFResNet import *
from tensorboardX import SummaryWriter
from datasets.es_imagenet import ESImagenet_Dataset
from importlib import import_module
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist 
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
__USE_AUTO_CAST__ = False 
if sys.version_info.minor>=6:
    __USE_AUTO_CAST__ = True
    print('use torch.cuda.amp.autocast for fusion prcision training')

def autocast():
    class null_cast:
        def __enter__(self):
            pass
        def __exit__(self,exc_type,exc_val,exc_tb):
            pass
    if __USE_AUTO_CAST__:
        from torch.cuda.amp import autocast
        return autocast()
    else:
        return null_cast()


        
# 配置LIAF包
args = {'allow_print':True,
        'use_gause_approx':False,
        'use_rect_approx':True,
        'decay_trainable':True,
        'thresh_trainable':False,
        'use_td_batchnorm':False,
        'if_clamp_the_output':True,
        'save_featuremap':False,
        'seed_value':1
       }
LIAF.config_LIAF(args)


#仅在master进程上输出
writer = None 
master = False 


##########################################################
# 修改部分1
# 实验名称
##########################################################
#任务名称
#save_folder = 'ResNet18_imagenet_LIF'
#save_folder = 'ResNet18_imagenet_LIAF'
save_folder = 'spikingResNet18_0905'#任务：训练ReLU1下的ResNet，精度大于65%
#save_folder = 'spikingResNet18_0830_time1LIF'#任务：增加一点数据增强，训练ReLU1下的ResNet，看看精度

timeWindows = 1
#timeWindows = 8

train_path = '/data/imagenet2012_png/train' 
test_path = '/data/imagenet2012_png/val' 

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank',type = int,default=0)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group('nccl',init_method='env://')
local_rank = dist.get_rank()
world_size = dist.get_world_size()

print(dist.get_rank(),' is ready')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################### Step2. load in dataset #####################

modules = import_module('LIAFnet.LIAFResNet_18')
config  = modules.Config()
workpath = os.path.abspath(os.getcwd())
config.cfgCnn = [3, 64, 7, True]
config.batch_size = 256//7 #Time = 1
config.num_epochs = 50
batch_size = config.batch_size
num_epochs = config.num_epochs
config.attention_model=None


##########################################################
# 修改部分2
# 激活函数
##########################################################
#Step1. ReLU1 训练50epoch
config.actFun= torch.nn.ReLU()
#Step2. LIF 训练50epoch
#config.actFun= LIAF.LIFactFun.apply


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
training_iter = 0
start_epoch = 0
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = torchvision.datasets.ImageFolder(root= train_path, transform=transform_train)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
test_dataset = torchvision.datasets.ImageFolder(root= test_path,transform=transform_test)
test_sampler  = torch.utils.data.distributed.DistributedSampler(test_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=True,drop_last=True,sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=True)

##################### Step3. establish module #####################

snn = LIAFResNet(config)
snn=torch.nn.SyncBatchNorm.convert_sync_batchnorm(snn)
snn.to(device)

##########################################################
# 修改部分3
# 载入模型
##########################################################
print('using uniformed init')
#checkpoint = torch.load('./spikingResNet18_0825/64.pkl', map_location=torch.device('cpu'))
#checkpoint = torch.load('../../saved_model/ResNet18_imagenet_CNN/sgd_65.pkl', map_location=torch.device('cpu'))
#snn.load_state_dict(checkpoint)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(snn.parameters(),
                lr=1e-1,
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=True)


lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, 
                    milestones=[10,20,30,40], 
                    gamma=0.2, 
                    last_epoch=-1)

#防止进程冲突
with torch.cuda.device(local_rank):
    snn = DDP(snn,device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

    
if local_rank == 0 :
    writer = SummaryWriter(comment=save_folder)
    master = True
    print('no bugs, start recording')
    
    
################step4. training and validation ################
bestacc = 0
def val(optimizer,snn,test_loader,test_dataset,batch_size,epoch):
    print('===> evaluating models...')
    snn.eval()
    correct = 0
    total = 0
    predicted = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if ((batch_idx+1)<=len(test_dataset)//batch_size):
                optimizer.zero_grad()
                imsize = inputs.size()
                image_train = torch.zeros(imsize[0],imsize[1],timeWindows,imsize[2],imsize[3])
                for time in range(timeWindows):
                    image_train[:,:,time,:,:] = inputs
                outputs = snn(image_train.type(LIAF.dtype))
                _ , predicted = outputs.cpu().max(1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets).sum())
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    if master:
        writer.add_scalar('acc_th', acc,epoch)
    return acc

for epoch in range(num_epochs):
    
    #timeWindows = math.floor(epoch/5)+1

    running_loss = 0
    correct = 0.0
    total = 0.0
    
    snn.train()
    start_time = time.time() 
    print('===> training models...')
    torch.cuda.empty_cache()
    train_loader.sampler.set_epoch(epoch)
    for i, (images, labels) in enumerate(train_loader):
        if ((i+1)<=len(train_dataset)//batch_size):
            snn.zero_grad()
            optimizer.zero_grad()
            imsize = images.size()
            image_train = torch.zeros(imsize[0],imsize[1],timeWindows,imsize[2],imsize[3])
            for time2 in range(timeWindows):
                image_train[:,:,time2,:,:] = images
           
            with autocast():
                outputs = snn(image_train.type(LIAF.dtype)).cpu()
                loss = criterion(outputs, labels)
                
            _ , predict = outputs.max(1)
            correct += predict.eq(labels).sum()
            total += float(predict.size(0))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i+1)%10 == 0:
                if master : 
                    if not os.path.isdir(save_folder):
                        os.mkdir(save_folder)
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f \n'
                    %(epoch+start_epoch, num_epochs+start_epoch, i+1, len(train_dataset)//(world_size*batch_size),running_loss ))
                    print('Time elasped: %d \n'  %(time.time()-start_time))
                    writer.add_scalar('Loss_th', running_loss, training_iter)
                    writer.add_scalar('timeWindows', running_loss, timeWindows)
                    train_acc =  correct / total
                    print('Epoch [%d/%d], Step [%d/%d], acc: %.5f \n'
                        %(epoch+start_epoch, num_epochs+start_epoch, i+1, len(train_dataset)//(world_size*batch_size), train_acc)) 
                    writer.add_scalar('train_acc', train_acc*100, training_iter)
                correct = 0.0
                total = 0.0
                running_loss = 0
        training_iter +=1 
    lr_scheduler.step()
    with torch.no_grad():
        torch.cuda.empty_cache()
        if master:
            acc = val(optimizer,snn,test_loader,test_dataset,batch_size,epoch)
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)
            if acc > bestacc:
                bestacc = acc
                print('===> Saving models...')
                torch.save(snn.module.state_dict(),
                         './'+save_folder+'/'+str(int(bestacc))+'__t='+str(int(timeWindows))+'.pkl')

