# -*- coding: utf-8 -*-
#step4 尝试使用自建模型训练LIAF


from dcll.load_dvsgestures_sparse import *
import argparse, pickle, torch, time, os
import torch.nn as nn
import LIAF_module
import LIAF
import numpy as np
import pandas as pd
import util

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = LIAF_module.device


#################################
batch_size = 36
actFun = LIAF.LIFactFun.apply

decay = 0.3
learning_rate = 1e-4
time_window = 60

#################################

# How many epochs to run before testing
n_test_interval = 20

dt = 25000  # us
ds = 4
target_size = 11  # num_classes
n_epochs = 4000  # 4500
in_channels = 2  # Green and Red
im_dims = im_width, im_height = (128 // ds, 128 // ds)
names = 'dvsGesture_stbp_cnn_test1'

n_iters =  time_window
n_iters_test =  time_window

#################################
parser = argparse.ArgumentParser(description='STBP for DVS gestures')
loss_train_list = []
loss_test_list = []
acc_train_list = []
acc_test_list = []
train_correct = 0
test_epoch = 20

# Load data
gen_train, _ = create_data(
    batch_size=batch_size,
    chunk_size=n_iters,
    size=[in_channels, im_width, im_height],
    ds=ds,
    dt=dt)

_, gen_test = create_data(
    batch_size=batch_size,
    chunk_size=n_iters_test,
    size=[in_channels, im_width, im_height],
    ds=ds,
    dt=dt)

def generate_test(gen_test, n_test: int, offset=0):
    input_test, labels_test = gen_test.next(offset=offset)
    input_tests = []
    labels1h_tests = []
    n_test = min(n_test, int(np.ceil(input_test.shape[0] / batch_size)))
    for i in range(n_test):
        input_tests.append(
            torch.Tensor(input_test.swapaxes(0, 1))[:, i * batch_size:(i + 1) * batch_size].reshape(n_iters_test, -1,
                                                                                                    in_channels,
                                                                                                    im_width,
                                                                                                    im_height))
        labels1h_tests.append(torch.Tensor(labels_test[:, i * batch_size:(i + 1) * batch_size]))
    return n_test, input_tests, labels1h_tests


cfg_cnn = [(2, 64, 3, 1, 1, False),
           (64, 128, 3, 2, 1, True),
           (128, 128, 3, 2, 1, True),
           ]

cfg_fc = [256, 11]

best_acc = 0
acc = 0

snn = LIAF_module.LIAFCNN(cfg_cnn,
                   cfg_fc,
                   dataSize=im_width,
                   actFun=actFun,
                   padding=1,
                   timeWindows=time_window,
                   dropOut=0,
                   decay=decay,
                   useBatchNorm=True,
                   useThreshFiring=True)
#trial Qutified --
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

n_test, input_tests, labels1h_tests = generate_test(gen_test, n_test=100, offset=0)
input_extras = []
labels1h_extra = []

for i in range(8):
    input_extra, labels_extra = gen_train.next()
    n_extra = min(100, int(np.ceil(input_extra.shape[0] / batch_size)))
    for i in range(n_extra):
        input_extras.append(
            torch.Tensor(input_extra.swapaxes(0, 1))[:, i * batch_size:(i + 1) * batch_size].reshape(n_iters, -1,
                                                                                                     in_channels,
                                                                                                     im_width,
                                                                                                     im_height))
        labels1h_extra.append(torch.Tensor(labels_extra[:, i * batch_size:(i + 1) * batch_size]))

running_loss = 0

for epoch in range(n_epochs):
    snn.zero_grad()
    optimizer.zero_grad()
    start_time = time.time()
    input, labels = gen_train.next()
    input = torch.Tensor(input.swapaxes(0, 1)).reshape(n_iters, batch_size, in_channels, im_width, im_height)
    input = input.float()
    input = input.permute([1, 2, 3, 4, 0])
    labels = torch.from_numpy(labels).float()
    labels = labels[1, :, :]
    outputs = snn(input)

    _, predicted = torch.max(outputs.data, 1)
    _, labelTest = torch.max(labels.data, 1)
    total = labelTest.size(0)
    train_correct += (predicted.cpu() == labelTest).sum()

    loss = criterion(outputs.cpu(), labels)
    running_loss = running_loss + loss.item()
    loss.backward()
    optimizer.step()

    print('Epoch [%d/%d], Loss:%.5f' % (epoch + 1, n_epochs, loss.item()))

    if (epoch + 1) % test_epoch == 0:
        correct = 0
        total = 0
        optimizer = util.lr_scheduler(optimizer, epoch, learning_rate, 1000)
        test_loss = 0
        for i in range(len(input_tests)):
            torch.cuda.empty_cache()
            # input_tests[i] = input_tests[i].swapaxes(0,1).reshape(n_iters,batch_size,in_channels,im_width,im_height)
            inputTest = input_tests[i].float()
            inputTest = inputTest.permute([1, 2, 3, 4, 0])
            outputs = snn(inputTest)

            calloss = torch.mean(labels1h_tests[i].data, 0)
            _, predicted = torch.max(outputs.data, 1)
            _, labelTestTmp = torch.max(labels1h_tests[i].data, 2)

            labelTest, _ = torch.max(labelTestTmp.data, 0)
            total = total + labelTest.size(0)

            correct = correct + (predicted.cpu() == labelTest).sum()
            loss = criterion(outputs.cpu().data, calloss.data)
            test_loss += loss.item()
            del inputTest, outputs, predicted, labelTestTmp, labelTest

        test_loss = test_loss / len(input_tests)
        running_loss = running_loss / test_epoch
        print('test loss:%.5f' % (test_loss))
        print('tarin loss:%.5f' % (running_loss))

        loss_test_list.append(test_loss)
        loss_train_list.append(running_loss)
        running_loss = 0

        acc = 100. * float(correct) / float(total)
        acc_test_list.append(acc)
        acc_train_list.append(float(train_correct) / (batch_size * test_epoch))

        print('test acc: %.3f' % (100 * correct.float() / total))
        print('train acc: %.3f' % (100 * float(train_correct) / (batch_size * test_epoch)))
        train_correct = 0

        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_test_list,
        }
        if acc > best_acc:
            best_acc = acc
            # 保存
            if best_acc > 0.9:
                torch.save(snn, '/home/lyh/dvs_gestrue_model/Lmodel_' + str(int(best_acc * 100)) + '.pkl')
            # 加载
            # snn = torch.load('\model.pkl')
        print('best=', best_acc)

print(acc_test_list)
print(acc_train_list)
print(loss_test_list)
print(loss_train_list)

list = [acc_test_list, acc_train_list, loss_test_list, loss_train_list]
test = pd.DataFrame(data=list)  # 数据有三列，列名分别为one,two,three
test.to_csv('/home/lyh/dvs_gestrue_model/result_LIF.csv')

