# Author: lyh 
# Date  : 2020-09-19
# 使用了分布式学习的ImageNet训练代码
# 使用以下命令直接执行
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --nproc_per_node=7 example_ES_res18.py
import sys
sys.path.append("..")
import LIAF
from LIAFnet.LIAFResNet import *
from datasets.es_imagenet import ESImagenet_Dataset
autocast = LIAF.autocast #统一autocast模式

from importlib import import_module
from tensorboardX import SummaryWriter
import argparse, torch, time, os
import torch.distributed as dist 
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


date = '2021_09_02 '
source_dataset = 'ImageNet '
aim_dataset = 'ES_ImageNet '
model = 'ResNet34 '

task = 0 # 直接训练LIAF
#task = 1 # 直接训练LIF
#task = 3 # LIAF固定第一层的Warmup
#task = 4 # warmup后 从ImageNet预训练LIAF
#task = 5 # warmup后 从LIAF预训练LIF

task_name = ['direct_LIAF_on','direct_LIF_on','warmup_LIAF_on',
                    'pretrained_LIAF_on','pretrained_LIF_on']
input_dim = [2,2,2,2,2]
save_folder = task_name[task] + source_dataset + 'to' + aim_dataset + model+date

##################### Step1. Env Preparation #####################
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank',type = int,default=0)
args = parser.parse_args()
torch.distributed.init_process_group('nccl',init_method='env://')
local_rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device('cuda:'+str(local_rank))
torch.cuda.set_device(local_rank)
print(local_rank,' is ready')
writer = None #仅在master进程上输出
master = False 

##################### Step2. load in dataset #####################
modules = import_module('LIAFnet.LIAFResNet_34')
config  = modules.Config()
workpath = os.path.abspath(os.getcwd())
config.cfgCnn = [input_dim, 64, 7, True]
config.actFun= LIAF.LIFactFun.apply

num_epochs = config.num_epochs
batch_size = config.batch_size
timeWindows = config.timeWindows

epoch = 0
bestacc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
training_iter = 0

train_dataset = ESImagenet_Dataset(mode='train',data_set_path='/data/ES-imagenet-0.18/')
test_dataset = ESImagenet_Dataset(mode='test',data_set_path='/data/ES-imagenet-0.18/')
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
test_sampler  = torch.utils.data.distributed.DistributedSampler(test_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,drop_last=True,sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

##################### Step3. establish module #####################

snn = LIAFResNet(config)

#checkpoint = torch.load('./LIAF_18_pure/12.pkl', map_location=torch.device('cpu'))
#snn.load_state_dict(checkpoint)

print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in snn.parameters())))

snn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(snn)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(snn.parameters(),
                lr=3e-2,
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=True)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, 
                    milestones=[8,16], 
                    gamma=0.1, 
                    last_epoch=-1)

with torch.cuda.device(local_rank):
    snn.to(device)
    snn = DDP(snn,device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    
    
if local_rank == 0 :
    writer = SummaryWriter(comment='runs/'+save_folder)
    master = True
    print('start recording')
    


################step4. training and validation ################

for epoch in range(num_epochs):
    #training
    running_loss = 0
    snn.train()
    start_time = time.time() 
    if master:
        print('===> training models...')
    correct = 0.0
    total = 0.0
    torch.cuda.empty_cache()
    # 新增2：设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
    train_loader.sampler.set_epoch(epoch)
    for i, (images, labels) in enumerate(train_loader):
        if ((i+1)<=len(train_dataset)//batch_size):
            snn.zero_grad()
            optimizer.zero_grad()
            
            with autocast():
                outputs = snn(images.type(LIAF.dtype)).cpu()
                labels = labels.view(batch_size)
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
                    train_acc =  correct / total
                    print('Epoch [%d/%d], Step [%d/%d], acc: %.5f \n'
                  %(epoch+start_epoch, num_epochs+start_epoch, i+1, len(train_dataset)//(world_size*batch_size), train_acc)) 
                    writer.add_scalar('train_acc', train_acc*100, training_iter)
                correct = 0.0
                total = 0.0
                running_loss = 0
        training_iter +=1 
    torch.cuda.empty_cache()
    #evaluation
    acc = val(optimizer,snn,test_loader,test_dataset,batch_size,epoch)
    lr_scheduler.step()
    if master:
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        if acc > bestacc:
            bestacc = acc
            print('===> Saving models...')
            torch.save(snn.module.state_dict(),
                     './'+save_folder+'/'+str(int(bestacc))+'.pkl')

