import torch
from metrics import AverageMeter
from tqdm import tqdm 
from test import result
import matplotlib.pyplot as plt
import numpy as np
from arcface import loss_fn

def save(save_path, model, optimizer, scheduler):
    if save_path==None:
        return
    checkpoint = { 
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
    }
    save_path = './models/' + save_path 
    torch.save(checkpoint, save_path)
    print(f'Model saved to ==> {save_path}')

def load(save_path):
    save_path = './models/' + save_path 
    checkpoint = torch.load(save_path)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
    print(f'Model loaded from <== {save_path}')
    return model, optimizer, scheduler

# train function for arcface
def train(model,train_loader,valid_loader1,valid_loader2,metric_crit,optimizer,scheduler,num_epochs,eval_every,num_class,device,name):
    IOU_list = []
    best_IOU = 1
    global_step = 0
    # loss from 1st epoch to nth epoch
    train_loss = AverageMeter()
    # loss to n step/eval_every
    local_train_loss = AverageMeter()
    # a list host the loss every eval_every
    loss_list = []
    best_train_loss = float("Inf")
    total_step = len(train_loader)*num_epochs
    print(f'total steps: {total_step}')
    for epoch in range(num_epochs):
        print(f'epoch {epoch+1}')
        for _, data in enumerate(tqdm(train_loader)):
            model.train()
            # original image
            inputs = data['image'].to(device)
            #targets = data['target'].to(device)
            outputs = model(inputs)
            loss = loss_fn(metric_crit, data, outputs, num_class, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.cpu().item(), inputs.size(0))
            local_train_loss.update(loss.cpu().item(), inputs.size(0))
            global_step += 1
            current_lr = optimizer.param_groups[0]['lr']
            if global_step % eval_every == 0:
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f} ({:.4f}), lr: {:.4f}'
                          .format(epoch+1, num_epochs, global_step, total_step, local_train_loss.avg, train_loss.avg, current_lr))
                if local_train_loss.avg < best_train_loss:
                    best_train_loss = local_train_loss.avg
                    print('Best trian loss:',local_train_loss.avg)
                loss_list.append(local_train_loss.avg)
                local_train_loss.reset()
        # val
        dist1 = result(model,valid_loader1,device)
        dist2 = result(model,valid_loader2,device)
        dist_threshold = 0
        try:

            same_hist = plt.hist(dist1, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='same')
            diff_hist = plt.hist(dist2, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='diff')
            # rest of the code

            same_hist_0 = same_hist[0]
            same_hist_0[same_hist_0 == 0] = 0.00001
            diff_hist[diff_hist == 0] = 0.00001
           
            print('same_hist', same_hist[1])
            print('diff_hist', diff_hist[1])
            print('type', type(diff_hist[1]))
            #same_hist = plt.hist(dist1, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='same')
            #diff_hist = plt.hist(dist2, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='diff')
            plt.legend(loc='upper right')
            plt.savefig('result/distribution_epoch'+str(epoch+1)+'.png')
            difference = same_hist[0] - diff_hist[0]
            difference[:same_hist[0].argmax()] = np.Inf
            difference[diff_hist[0].argmax():] = np.Inf
            dist_threshold = (same_hist[1][np.where(difference <= 0)[0].min()] + same_hist[1][np.where(difference <= 0)[0].min() - 1])/2
            overlap = np.sum(dist1>=dist_threshold) + np.sum(dist2<=dist_threshold)
            IOU = overlap / (dist1.shape[0] * 2 - overlap)
        except Exception as e:
            print("Model results in collapse") # if the collapse to 0 then, the result cannot be printed
            print(e)
            for i in range(len(same_hist[1])):
                if same_hist[1][i].size == 0:
                    print('same_hist: ', same_hist[1][i])
            for i in range(len(diff_hist[1])):
                if diff_hist[1][i].size == 0:
                    print('diff_hist: ', diff_hist[1][i])
            continue

        try:
            print('dist_threshold:',dist_threshold,'overlap:',overlap,'IOU:',IOU)
            plt.clf()
            IOU_list.append(IOU)
            if IOU < best_IOU:
                best_IOU = IOU
                save(name,model,optimizer,scheduler)
        except Exception as e:
            print(e)
            pass


        scheduler.step()
    # loss graph
    steps = range(len(loss_list))
    plt.plot(steps, loss_list)
    plt.title('Train loss')
    plt.ylabel('Loss')
    plt.xlabel('Steps')
    plt.savefig('train_loss.png')
    plt.clf()
    print('Finished Training')