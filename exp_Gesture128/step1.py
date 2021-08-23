# -*- coding: utf-8 -*-
#测试：原始的LIF代码

from dcll.pytorch_libdcll import *
from dcll.experiment_tools import *
from dcll.load_dvsgestures_sparse import *
from tqdm import tqdm

import argparse, pickle, torch, time, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#######
#step1 尝试多一个输出


os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# For how many ms do we present a sample during classification
n_iters = 1500
n_iters_test = 1500 

# How many epochs to run before testing
n_test_interval = 20

batch_size = 36
dt = 1000 #us
ds = 4
target_size = 11 # num_classes
n_epochs = 4000 # 4500
in_channels = 2 # Green and Red
thresh = 0.3
lens = 0.5
decay = 0.3
learning_rate = 1e-4
time_window = 60
im_dims = im_width, im_height = (128//ds, 128//ds)
names = 'dvsGesture_stbp_cnn_test1'
frametime = 1 # x ms for a frame
n_iters = frametime * time_window
n_iters_test = frametime * time_window
parser = argparse.ArgumentParser(description='STBP for DVS gestures')

#Load data
gen_train, _ = create_data(
        batch_size = batch_size,
        chunk_size = n_iters,
        size = [in_channels, im_width, im_height],
        ds = ds,
        dt = dt)


_, gen_test = create_data(
        batch_size = batch_size,
        chunk_size = n_iters_test,
        size = [in_channels, im_width, im_height],
        ds = ds,
        dt = dt)

def generate_test(gen_test, n_test:int, offset=0):
    input_test, labels_test = gen_test.next(offset=offset)
    input_tests = []
    labels1h_tests = []
    n_test = min(n_test,int(np.ceil(input_test.shape[0]/batch_size)))
    for i in range(n_test):
        input_tests.append( torch.Tensor(input_test.swapaxes(0,1))[:,i*batch_size:(i+1)*batch_size].reshape(n_iters_test,-1,in_channels,im_width,im_height))
        labels1h_tests.append(torch.Tensor(labels_test[:,i*batch_size:(i+1)*batch_size]))
    return n_test, input_tests, labels1h_tests

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float() / (2 * lens)

cfg_cnn = [(2, 64, 1),
           (64, 128, 1),
           (128, 128, 1),
           ]

cfg_fc = [256, 11]


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


best_acc = 0
acc = 0
acc_record = list([])

act_fun = ActFun.apply

fun = ActFun.apply

class SNN_Model(nn.Module):

    def __init__(self, num_classes=10):
        super(SNN_Model, self).__init__()
        self.conv0 = nn.Conv2d(2, cfg_cnn[0][1], kernel_size=3, stride=1, padding=1)
        in_planes, out_planes, stride = cfg_cnn[1]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

        in_planes, out_planes, stride = cfg_cnn[2]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(8 * 8 * cfg_cnn[1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

    def forward(self, input, win=15):
        c0_mem = c0_spike = torch.zeros(batch_size, cfg_cnn[0][1], 32, 32, device=device)
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[1][1], 32, 32, device=device)
        p1_mem = p1_spike = torch.zeros(batch_size, cfg_cnn[1][1], 16, 16, device=device)

        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[2][1], 16, 16, device=device)
        p2_mem = p2_spike = torch.zeros(batch_size, cfg_cnn[2][1], 8, 8, device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        for step in tqdm(range(win)):
            x = torch.zeros(input[:,:,:,:,frametime*step].shape)
            for i in range(frametime):
                x = x + input[:,:,:,:,frametime*step+i]
               
            xtmp = torch.ones(x.shape)
            xtmp0 = torch.zeros(x.shape)
            x = torch.where(x>1,xtmp,xtmp0).to(device)
          
            # print(x.size())
            c0_mem, c0_spike, output = mem_update(self.conv0, x, c0_mem, c0_spike)
            del x,xtmp,xtmp0
            torch.cuda.empty_cache()
            c1_mem, c1_spike, output = mem_update(self.conv1, output, c1_mem, c1_spike)
            p1_mem, p1_spike, output = mem_update_pool(F.avg_pool2d, output, p1_mem, p1_spike)

            c2_mem, c2_spike, output = mem_update(self.conv2, output, c2_mem, c2_spike)
            p2_mem, p2_spike, output = mem_update_pool(F.avg_pool2d, output, p2_mem, p2_spike)

            x = output.view(batch_size, -1)

            h1_mem, h1_spike, output = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike

            h2_mem, h2_spike, output = mem_update(self.fc2, output, h2_mem, h2_spike)
            h2_sumspike += output
            del x
            torch.cuda.empty_cache()
        outputs = h2_sumspike / time_window
        #print(torch.mean(outputs,dim=0))
        return outputs

def mem_update(fc, x, mem, spike):
    torch.cuda.empty_cache()
    mem = mem * decay * (1 - spike) + fc(x)
    spike = act_fun(mem)
    output = fun(mem)
    return mem, spike, output

def mem_update_pool(opts, x, mem, spike, pool = 2):
    torch.cuda.empty_cache()
    mem = mem * decay * (1 - spike) + opts(x, pool)
    #mem = opts(x, pool)
    spike = act_fun(mem)
    output = fun(mem)
    return mem, spike, output

snn = SNN_Model()
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

n_test, input_tests, labels1h_tests = generate_test(gen_test, n_test=100, offset = 0)
input_extras = []
labels1h_extra = []

for i in range(8):
    input_extra, labels_extra = gen_train.next()
    n_extra = min(100,int(np.ceil(input_extra.shape[0]/batch_size)))
    for i in range(n_extra):
        input_extras.append( torch.Tensor(input_extra.swapaxes(0,1))[:,i*batch_size:(i+1)*batch_size].reshape(n_iters,-1,in_channels,im_width,im_height))
        labels1h_extra.append(torch.Tensor(labels_extra[:,i*batch_size:(i+1)*batch_size]))


for epoch in range(n_epochs):
    snn.zero_grad()
    optimizer.zero_grad()
    
    running_loss = 0
    start_time = time.time()

    input, labels = gen_train.next()
    input = torch.Tensor(input.swapaxes(0,1)).reshape(n_iters,batch_size,in_channels,im_width,im_height)
    input = input.float()
    input = input.permute([1,2,3,4,0])
    labels = torch.from_numpy(labels).float()
    labels = labels[1, :, :]
    outputs = snn(input, time_window)

    loss = criterion(outputs.cpu(), labels)
    running_loss = running_loss + loss.item()
    loss.backward()
    optimizer.step()
    print('Epoch [%d/%d], Loss:%.5f' % (epoch + 1, n_epochs, running_loss))

    if (epoch + 1) % 20 == 0:
        correct = 0
        total = 0
        optimizer = lr_scheduler(optimizer, epoch, learning_rate, 200)
    
        for i in range(len(input_tests)):
            torch.cuda.empty_cache()
            #input_tests[i] = input_tests[i].swapaxes(0,1).reshape(n_iters,batch_size,in_channels,im_width,im_height)
            inputTest = input_tests[i].float()
            inputTest = inputTest.permute([1,2,3,4,0])
            outputs = snn(inputTest, time_window)
            _, predicted = torch.max(outputs.data, 1)
            _, labelTestTmp = torch.max(labels1h_tests[i].data, 2)
            labelTest, _ = torch.max(labelTestTmp.data, 0)
            total = total + labelTest.size(0)
            correct = correct + (predicted.cpu() == labelTest).sum()
            del inputTest, outputs, predicted, labelTestTmp, labelTest

        print('Test Accuracy of the model on the test images: %.3f' % (100 * correct.float() / total))
        print(total)
        print('total' )
        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)

        print(acc)
        print('Saving..')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        best_acc = acc

    
