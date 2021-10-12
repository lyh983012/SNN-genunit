import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.2 # decay constants
num_classes = 10
batch_size  = 40
learning_rate = 2e-1
num_epochs = 50 # max epoch
time_window=6
is_training = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = math.sqrt(7)
def assign_optimizer(model, lrs=1e-3):

    rate = 1e-1
    fc1_params = list(map(id, model.fc1.parameters()))
    fc2_params = list(map(id, model.fc2.parameters()))
    # fc3_params = list(map(id, model.fc3.parameters()))
    base_params = filter(lambda p: id(p) not in fc1_params + fc2_params , model.parameters())

    optimizer = torch.optim.SGD([
        {'params': base_params},
        {'params': model.fc1.parameters(), 'lr': lrs * rate},
        {'params': model.fc2.parameters(), 'lr': lrs * rate},
          ]
        , lr=lrs,momentum=0.9)

    print('successfully reset lr')
    return optimizer
# define approximate firing function
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
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update
def mem_update(x, mem, spike):
    for i in range(time_window):
        if i>=1 :
            mem[i] = mem[i].clone()*decay*(1 - spike[i-1].clone()) + x[i].clone()
        else:
            mem[i] = mem[i].clone()*decay  + x[i].clone()
        spike[i] = act_fun(mem[i].clone()) # act_fun : approximation firing function
    return mem, spike

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

class batch_norm_1d(nn.Module):
    def __init__(self,num_features,eps=1e-5,momentum=0.1):
        super(batch_norm_1d,self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.moving_mean = torch.zeros(num_features,device=device)
        self.moving_var = torch.zeros(num_features,device=device)
        nn.init.constant_(self.moving_mean,0)
        nn.init.constant_(self.moving_var,1)
        nn.init.uniform_(self.gamma)
        nn.init.constant_(self.beta,0)
    def forward(self,input):
        y = input.transpose(0,2).to(device)

        return_shape = y.shape
        y = y.contiguous().view(input.size(2),-1)
        z = y**2
        var = torch.sqrt(z.mean(dim=1)).to(device)
        mean = y.mean(dim=1).to(device)
        if is_training:
            y = (y-mean.view(-1,1)) / (var.view(-1,1)+  self.eps)*self.gamma.view(-1,1)+self.beta.view(-1,1)
            self.moving_var = self.momentum * self.moving_var + (1. - self.momentum) * var
            self.moving_mean = self.momentum * self.moving_mean + (1. - self.momentum) * mean
        else:
            y = (y-mean.view(-1,1)) / (var.view(-1,1)+  self.eps)*self.gamma.view(-1,1)+self.beta.view(-1,1)

        return y.view(return_shape).transpose(0,2)
class batch_norm_2d(nn.Module):
    def __init__(self,num_features,eps=1e-5,momentum=0.1):
        super(batch_norm_2d,self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.moving_mean = torch.zeros(num_features,device=device)
        self.moving_var = torch.zeros(num_features,device=device)
        nn.init.constant_(self.moving_mean,0)
        nn.init.constant_(self.moving_var,1)
        nn.init.uniform_(self.gamma)
        nn.init.constant_(self.beta,0)
    def forward(self,input):
        y = input.transpose(0,2).to(device)

        return_shape = y.shape
        y = y.contiguous().view(input.size(2),-1)
        z = y**2
        var = torch.sqrt(z.mean(dim=1)).to(device)
        mean = y.mean(dim=1).to(device)
        if is_training:
            y = (y-mean.view(-1,1)) / (var.view(-1,1)+  self.eps)*self.gamma.view(-1,1)+self.beta.view(-1,1)
            self.moving_var = self.momentum * self.moving_var + (1. - self.momentum) * var
            self.moving_mean = self.momentum * self.moving_mean + (1. - self.momentum) * mean
        else:
            y = (y-mean.view(-1,1)) / (var.view(-1,1)+  self.eps)*self.gamma.view(-1,1)+self.beta.view(-1,1)

        return y.view(return_shape).transpose(0,2)
class Snn_Conv2d(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,stride,padding=0,bias=False):
        super(Snn_Conv2d,self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,bias=bias)
        torch.nn.init.kaiming_normal_(self.conv.weight,a=a)
    def forward(self,input):
        h = (input.size()[3]-self.kernel+2*self.padding)//self.stride+1
        w = (input.size()[4]-self.kernel+2*self.padding)//self.stride+1
        c1 = torch.zeros(time_window,input.size()[1],self.out_ch , h, w, device=device)
        for i in range(time_window):
            c1[i] = self.conv(input[i])
        return c1
class ResidualBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.conv1 = Snn_Conv2d(in_ch,out_ch,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = batch_norm_2d(out_ch)
        self.conv2 = Snn_Conv2d(out_ch,out_ch,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = batch_norm_2d(out_ch)
        self.right = shortcut

    def forward(self,input):
        h = (input.size()[3]-1)//self.stride+1
        w = (input.size()[4]-1)//self.stride+1
        input = input.float().to(device)
        c1_prespike = c1_mem = c1_spike = torch.zeros(time_window,batch_size,self.out_ch , h, w, device=device)
        out = c2_prespike = c2_mem = c2_spike = torch.zeros(time_window,batch_size,self.out_ch , h, w, device=device)
        c1_prespike = self.conv1(input)
        c1_prespike = self.bn1(c1_prespike)
        c1_mem,c1_spike = mem_update(c1_prespike,c1_mem,c1_spike)
        c2_prespike = self.conv2(c1_spike)
        out = self.bn2(c2_prespike)
        residual = input if self.right is None else self.right(input)
        out += residual
        c2_mem,c2_spike = mem_update(out,c2_mem,c2_spike)
        return c2_spike

class Snn_ResNet20(nn.Module):

    def __init__(self,num_class=1):
        super(Snn_ResNet20,self).__init__()
        self.pre_conv = Snn_Conv2d(3,128,3,stride=1,padding=1,bias=False)
        self.pre_bn = batch_norm_2d(128)
        self.layer1 = self.make_layer(128,128,3)
        self.layer2 = self.make_layer(128,256,3,stride=2)
        self.layer3 = self.make_layer(256,512,2,stride=2)
        self.fc1 = nn.Linear(8192,256)
        self.fc2 = nn.Linear(256,10)
        self.bn1 = batch_norm_1d(256)
        torch.nn.init.kaiming_normal_(self.fc1.weight,a=a)
        torch.nn.init.kaiming_normal_(self.fc2.weight,a=a)

    def make_layer(self,in_ch,out_ch,block_num,stride=1):
        shortcut = nn.Sequential(
            Snn_Conv2d(in_ch,out_ch,1,stride,bias=False),
            batch_norm_2d(out_ch)
            )
        layers = []
        layers.append(ResidualBlock(in_ch,out_ch,stride,shortcut))
        for i in range(1,block_num):
            layers.append(ResidualBlock(out_ch,out_ch))
        return nn.Sequential(*layers)

    def forward(self,input_,is_train=True):
        is_training=is_train
        input = torch.zeros(time_window,batch_size,3,32,32,device=device)

        for i in range(time_window):
            input[i]=input_
        input = self.pre_conv(input)
        input = self.pre_bn(input)
        mem = spike = torch.zeros(time_window,batch_size,128,32,32,device=device)
        mem,spike = mem_update(input,mem,spike)

        out = self.layer1(spike)

        out = self.layer2(out)

        out = self.layer3(out)

    


        features = torch.zeros(time_window,batch_size,512,4,4,device=device)
        for i in range(time_window):
            features[i] = F.avg_pool2d(out[i],2)
        features = features.view(time_window,batch_size,-1)
        out_prespike = out_mem = out_spike = torch.zeros(time_window,batch_size,256,device=device)
        out_sumspike = torch.zeros(batch_size,256,device=device)
        for i in range(time_window):
            out_prespike[i] = self.fc1(features[i])
        out_prespike = self.bn1(out_prespike)
        out_mem,out_spike = mem_update(out_prespike,out_mem,out_spike)
        out_sumspike = out_spike.sum(dim=0)
        out1 = self.fc2(out_sumspike /time_window)

        return out1
