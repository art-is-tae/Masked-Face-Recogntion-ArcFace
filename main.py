# pytorch
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# utils
import pandas as pd
import numpy as np
import multiprocessing

# functions
from dataset import customized_dataset
from model import FaceNet
from util import get_Optimizer, get_Scheduler
from arcface import ArcFaceLoss
from train import train
from test import evalulate, test

if __name__ == "__main__":
    # load dataframe
    df_train = pd.read_csv('./label/train_CASIA_masked.csv')
    df_eval1 = pd.read_csv('./label/eval_same_RWMFVD.csv') # same
    df_eval2 = pd.read_csv('./label/eval_diff_RWMFVD.csv') # diff
    df_test = pd.read_csv('./label/test_RWMFVD.csv')    

    # params
    BATCH_SIZE=128
    # NUM_WORKERS = multiprocessing.cpu_count()
    NUM_WORKERS = torch.cuda.device_count() * 4
    #NUM_WORKERS = 1
    num_classes = df_train.target.nunique()
    print('NUM_WORKERS: ', NUM_WORKERS)
    lr = 1e-4 # learning rate
    weight_decay = 0.00001
    dropout = 0.3
    num_epochs = 5
    embedding_size = 512
    eval_every = 100
    name = 'arcface.pth'

    # arcface loss setting
    arcface_s = 45
    arcface_m = 0.4
    class_weights_norm = 'batch'

    # GPU setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #if (device.type == 'cuda') and torch.cuda.device_count() > 1:
    #    print('Multi GPU actuvate')
    
    print(device)

    # dataset setting
    train_dataset = customized_dataset(df_train, mode='train')
    eval_dataset1 = customized_dataset(df_eval1, mode='eval')
    eval_dataset2 = customized_dataset(df_eval2, mode='eval')
    test_dataset = customized_dataset(df_test, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=False)
    eval_loader1 = DataLoader(eval_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
    eval_loader2 = DataLoader(eval_dataset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, drop_last=False)
    
    # class_weights for arcface loss
    val_counts = df_train.target.value_counts().sort_index().values
    class_weights = 1/np.log1p(val_counts)
    class_weights = (class_weights / class_weights.sum()) * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # arcface
    metric_crit = ArcFaceLoss(arcface_s, arcface_m, crit='focal', weight=class_weights, class_weights_norm=class_weights_norm)
    facenet = FaceNet(num_classes=num_classes, model_name='resnet', pool='gem', embedding_size=embedding_size, dropout=dropout, device=device, pretrain=False)
    optimizer = get_Optimizer(facenet, metric_crit, optimizer_type='Adam') # optimizer
    scheduler = get_Scheduler(optimizer, lr, scheduler_name='multistep') # scheduler

    # train
    train(facenet.to(device),train_loader,eval_loader1,eval_loader2,metric_crit,optimizer,scheduler,num_epochs,eval_every,num_classes,device,name)
    #facenet = torch.load('./models/arcface.pth') # load model
    dist_threshold = evalulate(facenet, eval_loader1, eval_loader2, device)
    print('Distance threshold:',dist_threshold)
    test(facenet,test_loader,dist_threshold,device)
