from dcll.load_dvsgestures_sparse import *
import argparse, torch, time, os
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import pandas as pd

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())

class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            #用来创造一系列的卷积层，其中上一个层的隐层维度是下一个层的输入维度
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first: #batch first参数：batch的大小是第一个维度
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        # 为了实现结构化的设计，cell需要和batch无关，因此要交给LSTM模型顶层来初始化隐层来符合batchSize
        if hidden_state is not None:
            raise NotImplementedError() #为了实现训练时对状态进行修改？
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1) #第二维 时间序列长度
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output #把每一个时间步的输出状态叠加成下一层的输入

            layer_output_list.append(layer_output) #每一个时间步的状态表
            last_state_list.append([h, c]) #最终状态列表

        if not self.return_all_layers: #如果不需要返回所有层的话 就只返回最后一个就好
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# For how many ms do we present a sample during classification
n_iters = 60
n_iters_test = 60

# How epochs to run before testing
n_test_interval = 20

batch_size = 10
dt = 25000 #us
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
names = 'dvsGesture_lstm_origin'

parser = argparse.ArgumentParser(description='STDP for DVS gestures')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

cfg_conv = [[in_channels,64],[64,128],[128,128]]
cfg_kernel = [(3,3),(3,3),(3,3)]
cfg_pool = [[2,2],[2,2]]
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

class ConvLSTM(nn.Module):

    def __init__(self, input_size, conv_dim, pool_size, kernel_size, fc_size,
                 batch_first=False, bias=True):
        super(ConvLSTM, self).__init__()
        # conv_dim: dims of all these three conv layers, in a shape of a 3x2 list
        # pool_size: sizes of all these pool layers, in a shape of a 2x2 list
        # kernel_size: sizes of kernels of conv layers, in a shape of a 3x2 list
        # fc_size: sizes of the ouput side of fc layers, 1x2 list
        self.height, self.width = input_size
        self.conv_dim = conv_dim
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.fc_size = fc_size
        self.batch_first = batch_first
        self.bias = bias
        self.conv1 = ConvLSTMCell(input_size=(self.height, self.width),
                                  input_dim=conv_dim[0][0],
                                  hidden_dim=conv_dim[0][1],
                                  kernel_size=self.kernel_size[0],
                                  bias=self.bias)
        self.conv2 = ConvLSTMCell(input_size=(self.height, self.width),
                                  input_dim=conv_dim[1][0],
                                  hidden_dim=conv_dim[1][1],
                                  kernel_size=self.kernel_size[1],
                                  bias=self.bias)
        self.pool1 = nn.AvgPool2d(self.pool_size[0][0],self.pool_size[0][1])
        conv3Width = self.width//self.pool_size[0][0]
        conv3Height = self.height//self.pool_size[0][0]
        self.conv3 = ConvLSTMCell(input_size=(conv3Height, conv3Width),
                                  input_dim=conv_dim[2][0],
                                  hidden_dim=conv_dim[2][1],
                                  kernel_size=self.kernel_size[2],
                                  bias=self.bias)
        self.pool2 = nn.AvgPool2d(self.pool_size[0][0], self.pool_size[0][1])
        finalWidth = conv3Width//self.pool_size[1][0]
        finalHeight = conv3Width//self.pool_size[1][0]
        self.fc1 = nn.Linear(finalHeight*finalWidth*self.conv_dim[2][1],self.fc_size[0])
        self.fc2 = nn.Linear(self.fc_size[0],self.fc_size[1])
        print(self)

    def forward(self, input_tensor):
        if not self.batch_first: #batch first鍙傛暟锛歜atch鐨勫ぇ灏忔槸绗�涓€涓�缁村害
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        batch_size = input_tensor.size(0)
        seq_len = input_tensor.size(1)
        # convolution layer1
        h, c = self.conv1.init_hidden(batch_size)
        output_inner = []
        cur_layer_input = input_tensor
        for t in range(seq_len):
            h, c = self.conv1(input_tensor=cur_layer_input[:,t,:,:,:],
                              cur_state=[h, c])
            output_inner.append(h)

        layer_output = torch.stack(output_inner,dim=1)

        # convolution layer 2
        h, c = self.conv2.init_hidden(batch_size)
        output_inner = []
        cur_layer_input = layer_output
        for t in range(seq_len):
            h, c = self.conv2(input_tensor=cur_layer_input[:, t, :, :, :],
                              cur_state=[h, c])
            #if t%10==0:
            	#print(h.size(0))
            output_inner.append(self.pool1(h))

        layer_output = torch.stack(output_inner, dim=1)

        # pool layer 1
        # layer_output = self.pool1(layer_output)

        # convolution layer 3
        h, c = self.conv3.init_hidden(batch_size)
        output_inner = []
        cur_layer_input = layer_output
        for t in range(seq_len):
            h, c = self.conv3(input_tensor=cur_layer_input[:, t, :, :, :],
                              cur_state=[h, c])
            #if t%10==0:
            	#print(h.size())
            output_inner.append(self.pool2(h))
        #print('final output size:',output_inner.size())
        layer_output = output_inner[-1]
        #print('hidden shape: ',layer_output.size())
        # pool layer 2: extract the last layer
        # layer_output = self.pool2(layer_output)[:,-1,:,:,:]

        # Linear layers

        cur_layer_input = layer_output.view(batch_size,-1)
        #print('reshape result: ',cur_layer_input.size())
        layer_output = self.fc1(cur_layer_input)
        outputs = self.fc2(layer_output)
        return outputs

conv_rnn_model = ConvLSTM(input_size=im_dims,
                          conv_dim=cfg_conv,
                          pool_size=cfg_pool,
                          kernel_size=cfg_kernel,
                          fc_size=cfg_fc,
                          batch_first=True,
                          bias=True)
conv_rnn_model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(conv_rnn_model.parameters(),lr=learning_rate,weight_decay=0.0001)

act_fun = ActFun.apply
n_test, input_tests, labels1h_tests = generate_test(gen_test, n_test=100, offset = 0)


loss_train_list=[]
loss_test_list=[]
acc_train_list=[]
acc_test_list=[]
train_correct = 0
test_epoch=20


for epoch in range(n_epochs):
    conv_rnn_model.zero_grad()
    optimizer.zero_grad()

    running_loss = 0
    start_time = time.time()

    input, labels = gen_train.next()
    input = torch.Tensor(input.swapaxes(0,1)).reshape(n_iters,batch_size,in_channels,im_width,im_height)
    input = input.float().to(device)
    input = input.permute([1,0,2,3,4])
    #print(input.size())
    labels = torch.from_numpy(labels).float()
    labels = labels[1, :, :]
    #print(labels.size())
    outputs = conv_rnn_model(input)

    _, predicted = torch.max(outputs.data, 1)
    _, labelTest = torch.max(labels.data, 1)
    total = labelTest.size(0)
    train_correct += (predicted.cpu() == labelTest).sum()

    loss = criterion(outputs.cpu(), labels)
    running_loss = running_loss + loss.item()
    loss.backward()
    optimizer.step()
    print('Epoch [%d/%d], Loss:%.5f' % (epoch + 1, n_epochs, running_loss))

    if (epoch + 1) % n_test_interval == 0:
        correct = 0
        total = 0
        optimizer = lr_scheduler(optimizer, epoch, learning_rate, 1000)
        test_loss = 0

        for i in range(len(input_tests)):
            torch.cuda.empty_cache()
            conv_rnn_model.zero_grad()
            optimizer.zero_grad()
            inputTest = input_tests[i].float().to(device)
            #print('test input size: ',inputTest.size())
            inputTest = inputTest.permute([1,0,2,3,4])
            outputs = conv_rnn_model(inputTest)
            #print('output size: ',outputs.size())
            _, predicted = torch.max(outputs.data, 1)
            _, labelTestTmp = torch.max(labels1h_tests[i].data, 2)
            labelTest, _ = torch.max(labelTestTmp.data, 0)
            total = total + labelTest.size(0)

            calloss = torch.mean(labels1h_tests[i].data, 0)
            correct = correct + (predicted.cpu() == labelTest).sum()
            loss = criterion(outputs.cpu().data, calloss.data)
            test_loss += loss.item()

            del inputTest, outputs, predicted, labelTestTmp, labelTest


        test_loss = test_loss/len(input_tests)
        running_loss = running_loss/test_epoch
        print('test loss:%.5f' % (test_loss))
        print('tarin loss:%.5f' % (running_loss))

        loss_test_list.append(test_loss)
        loss_train_list.append(running_loss)
        running_loss = 0

        acc = 100. * float(correct) / float(total)
        acc_test_list.append(acc)
        acc_train_list.append(float(train_correct)/(batch_size*test_epoch))

        print('test acc: %.3f' % (100 * correct.float() / total))
        print('train acc: %.3f' % (100 * float(train_correct)/(batch_size*test_epoch)))
        train_correct = 0

        if acc>best_acc:
            best_acc=acc
        print('best=',best_acc)

print(acc_test_list)
print(acc_train_list)
print(loss_test_list)
print(loss_train_list)

list=[acc_test_list,acc_train_list,loss_test_list,loss_train_list]
test = pd.DataFrame(data=list)  # 数据有三列，列名分别为one,two,three
test.to_csv('/home/lyh/dvs_gestrue_model/result_CONV.csv')
