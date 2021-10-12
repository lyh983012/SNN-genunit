



def val(optimizer,snn,test_loader,test_dataset,batch_size,epoch):
    if master:
        print('===> evaluating models...')
    snn.eval()
    correct = 0
    total = 0
    predicted = 0
    with torch.no_grad():
        if master:
            for name,parameters in snn.module.named_parameters():
                writer.add_histogram(name, parameters.detach().cpu().numpy(),epoch)
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if ((batch_idx+1)<=len(test_dataset)//batch_size):
                optimizer.zero_grad()
                try:
                    targets=targets.view(batch_size)#tiny bug
                    outputs = snn(inputs.type(LIAF.dtype))
                    _ , predicted = outputs.cpu().max(1)
                    total += float(targets.size(0))
                    correct += float(predicted.eq(targets).sum())
                except:
                    print('sth. wrong')
                    print('val_error:',batch_idx, end='')
                    print('taret_size:',targets.size())
    acc = 100. * float(correct) / float(total)
    if master:
        writer.add_scalar('acc', acc,epoch)
    return acc

def train(optimizer,snn,test_loader,test_dataset,batch_size,epoch)
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

