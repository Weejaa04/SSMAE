#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:36:43 2022
This part was improved from MyMAE-SpatioSpectralLearning_Colab
@author: research
"""
#CUDA_LAUNCH_BLOCKING=1

import os 
import argparse
import torch
import torch.utils.data as datatorch 

import numpy as np
import random
import math


from HSI import *
from experiment import *
from MyHyperX2 import *
from model import *
#from model2 import *
from torchsummary import summary



def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def chek_band(x_train, x_test, mask_spectral):
    """
    This function is for:
        1. check wether the band is devisible by the mask spectral
        2. if not reduce the band so it can be devided by the spectral
    """
    band = x_train.shape[3]
    rest = band % mask_spectral
    #num_group = band//mask_spectral
    if rest!=0:
        #new_band = num_group * mask_spectral
        x_train = x_train [:,:,:,0:-rest]
        x_test = x_test[:,:,:, 0:-rest]                 
    return x_train, x_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default = 60)
    parser.add_argument('--batch_size', type=int, default = 5)
    parser.add_argument('--max_device_batch_size',type=int, default=64)
    parser.add_argument('--lr', type=float, default =1.5e-4)
    parser.add_argument('--weight_decay',type=float, default= 0.05)
    parser.add_argument ('--mask_ratio', type = float, default = 0.7)
    parser.add_argument('--epoch',type= int, default=1)
    parser.add_argument('--warmup_epoch',type= int, default = 200)
    parser.add_argument('--model_path',type=str, default='SS_MAE_')
    
    #some arguments for my case
    parser.add_argument('--dataset', type=str, default='PaviaU')
    parser.add_argument('--folder', type=str, default = 'D:\\Wijayanti\\Datasets\\')
    parser.add_argument('--norm_type', type=str, default = 'scale')
    parser.add_argument('--model', type=str, default='MAE')    
    parser.add_argument('--patch_size', type=int, default = 9)#previous is 9
    parser.add_argument('--mask_spatial', type=int, default=3)#spatial size of each mask
    parser.add_argument('--mask_spectral', type=int, default =25) #spectral size of each mask, previous is 5
    parser.add_argument('--sampling_mode',type=str, default='random')  
    parser.add_argument('--training_sample', type=float, default ='0.8')
    parser.add_argument('--padded', type=int, default = 1)
    parser.add_argument('--val_size', type=float, default = 0)
   
    #some arguments for the transformer model
    parser.add_argument('--emb_size', type= int, default = 96)
    parser.add_argument('--nhead_encoder',type = int, default = 3)
    parser.add_argument('--nhead_decoder',type = int, default=3)
    parser.add_argument('--depth_encoder',type = int, default = 12)
    parser.add_argument('--depth_decoder', type=int, default=3)
    
    
    #1. Setting parameter
    args = parser.parse_args()
    setup_seed (args.seed)
    batch_size = args.batch_size
    load_batch_size = min (args.max_device_batch_size, batch_size)
    
    assert batch_size%load_batch_size == 0
    steps_per_update = batch_size // load_batch_size    
    
    #2. Open the dataset until get the training_loader
    hyperparams = vars(args)
    
    HSIs = HSI(**hyperparams)
    
    #3. Devide the training and testing
    Ex = Experiment (HSIs.gt, **hyperparams)
    index_range = 0
    data = HSIs.Normalize(args.norm_type)
    
    """
    if args.dataset =='IndianPines': #just try PCA for indian pines because the #bands in IP is more than 200
        data=Ex.preprocess(data,'PCA',100)
        args.model_path = args.model_path+"PCA_"
    """
    Ex.set_train_test_SS (data, index_range)

    
    x_train = Ex.get_neighbour_patch(data, Ex.training_indices[0], Ex.training_indices[1], data.shape[2])
    x_test = Ex.get_neighbour_patch(data, Ex.testing_indices[0], Ex.testing_indices[1], data.shape[2])
    
    #check wether the number of band can be devided to the mask_spectral
    mask_spectral=args.mask_spectral
    x_train, x_test = chek_band(x_train, x_test, mask_spectral)
    num_band=x_train.shape[3]
    
    print ("num_band: ", num_band)
    
    #4. Making data loader, decide later for the training need to be augent or no
    
    train_dataset = MyHyperX2(x_train, x_train, **hyperparams) #because self, supervised learning, we do not need the label
    train_loader = datatorch.DataLoader(train_dataset, Ex.batch_size, shuffle=True)
    

    test_dataset = MyHyperX2(x_test, x_test, **hyperparams)
    val_loader = datatorch.DataLoader(test_dataset, Ex.batch_size)
    
    #5. Create model and training process
    
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    
    #create model first
    model = MAE_VIT_SS(args.patch_size,args.mask_spatial, args.mask_spectral, args.emb_size, 
                       args.depth_encoder, args.depth_decoder, args.nhead_encoder, args.nhead_decoder,args.mask_ratio, num_band).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr * args.batch_size / 32, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)    
        
    
    step_count = 0 #this is to update the learning rate
    optim.zero_grad()
    
    min_val_loss = 100
    
    #summary(model.cuda(),(9,9,num_band))
    
    for e in range (args.epoch):
        model.train()
        losses = []
        
        for i, datas in enumerate(train_loader):
            step_count += 1
            data, label = datas
            #print ("label size: ", label.size())
            data = data.to(device)
            label = label.to(device)
            predicted_data, mask = model(data.float()) #the content of mask is only 0 and 1, 0 means not masked, 1 means masked 
            #print ("predicted data size:", predicted_data.size())
            loss= torch.mean((predicted_data-label)**2*mask)/args.mask_ratio
            loss.backward()
            losses.append(loss.item())
            if step_count % steps_per_update ==0:
                optim.step()
                optim.zero_grad()
            
        lr_scheduler.step()
        avg_loss = sum (losses)/len(losses)
        #writer.add_scalar("mae loss ",avg_loss,global_steps = e)
        print ("in epoch ", e, "average training loss is ", avg_loss)
        
        #validation process
        model.eval()
        
        for data_val, label_val in val_loader:
            
            data_val = data_val.to(device)
            label_val = label_val.to(device)
            
          
            predicted_target, val_mask = model(data_val.float())
            val_loss= torch.mean((predicted_target-data_val)**2*val_mask)/args.mask_ratio
           

        if min_val_loss > val_loss:
            print(f'Validation Loss Decreased({min_val_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
            min_val_loss = val_loss
            # Saving State Dict
            name = 'model/'+str(args.model_path)+str(hyperparams["dataset"])+'_'+str(hyperparams["patch_size"])+'x'+str(hyperparams["patch_size"])+'_'+str(hyperparams["mask_spatial"])+ 'x'+str(hyperparams["mask_spectral"])+'_'+str(hyperparams["mask_ratio"])+'_'+str(hyperparams["emb_size"])+".pth"
  
            #name = 'model/SS_MAE_%s.pth'%(args.dataset)
            torch.save(model, name)

    
    
    
    
    
    
    