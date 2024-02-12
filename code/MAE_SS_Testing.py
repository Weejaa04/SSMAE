#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:47:55 2023

@author: research
"""


import torch 
import torch.utils.data as datatorch

import numpy as np
import argparse


from sklearn import metrics
import random 
from scipy.ndimage import rotate

from HSI import *
from experiment import *

from Augmentation import Augment_data
from model_meta_learn import MAE_Embedding2
from MyHyperX2 import *
from MetaData import get_meta_train_data
from Episode2 import *
from model import *

from scipy.special import softmax


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False #add this, will the result consistent?

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



def chek_band(x_train, x_test, mask_spectral):
    """
    This function is for:
        1. check wether the band is devisible by the mask spectral
        2. if not reduce the band so it can be devided by the spectral
    """
    band = x_train.shape[3]
    rest = band % mask_spectral
    if rest!=0:
        x_train = x_train [:,:,:,0:-rest]
        x_test = x_test[:,:,:, 0:-rest]                 
    return x_train, x_test

def set_train_val (gt, training_sample):
    gt = gt+1
    indices=np.nonzero(gt) 
    X=list(zip(*indices))
    y=gt[indices].ravel()

    training_indices, testing_indices = sklearn.model_selection.train_test_split(X, train_size=training_sample,stratify=y)
            
    training_indices = [list(t) for t in zip(*training_indices)]
    val_indices = [list(t) for t in zip(*testing_indices)]    
    
    return training_indices, val_indices





def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.class_for_tr
        num_samples = opt.support_tr + opt.query_tr
    else:
        classes_per_it = opt.class_for_val
        num_samples = opt.support_val + opt.query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)

def init_dataloader(x, y, mode, opt):

    sampler = init_sampler(opt, y, mode)
    #here I have to define the dataset
    hyperparams = vars (opt)
    dataset = MyHyperX2(x, y, **hyperparams)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 60) #default 60, for PU seed 42 produce the same result with before
    parser.add_argument('--batch_size', type = int, default = 256) #32
    parser.add_argument('--lr', type = float, default = 0.0003) #0.0003
    parser.add_argument('--mask_ratio', type = float, default =0.7)
    parser.add_argument('--epochs',type = int, default = 50)
    parser.add_argument('--epoch',type = int, default = 2)   
    parser.add_argument('--dataset', type = str, default = 'PaviaU' )
    parser.add_argument('--folder', type=str, default = 'D:\\Wijayanti\\Datasets\\')
    parser.add_argument('--norm_type', type=str, default = 'scale')
    parser.add_argument('--model', type=str, default='Transformer_MAE_Meta') #default: 'Transformer_MAE_Meta' 'Transformer_MAE_Meta_Multiscale'
    parser.add_argument('--patch_size', type=int, default = 9)
    parser.add_argument('--mask_spatial', type=int, default=3)#spatial size of each mask
    parser.add_argument('--mask_spectral', type=int, default = 10) #spectral size of each mask
    parser.add_argument('--emb_size', type=int, default = 96)
    parser.add_argument('--sampling_mode',type=str, default='kfold_FSL')  
    parser.add_argument('--training_sample', type=float, default ='5')
    parser.add_argument('--padded', type=int, default = 2) #1 is for zero padding, 2 is for reflect padding
    parser.add_argument('--val_size', type=float, default = 0) 
    parser.add_argument('--run', type = int , default = 1)
    parser.add_argument('--episode', type= int, default = 200)
    parser.add_argument('--SS_train_size', type=float, default=0.8)
    
    #several parameters for the meta-training (Prototypical Network)
    parser.add_argument('--lrS', type = int, default= 20, help = 'learning rate scheduler')
    parser.add_argument('--lrG', type= float, default = 0.5, help= 'learning rate gamma')
    parser.add_argument('--iterations', type = int, default=100, help= 'number of episodes per epoch')
    parser.add_argument('--class_for_tr', type = int, default =16, help='number of random classes per episode for training')
    parser.add_argument('--support_tr', type = int, default = 5, help='number of support samples per class for training')
    parser.add_argument('--query_tr', type = int, default = 15, help = 'number of query samples per class for training') #usually 15
    parser.add_argument('--class_for_val', type = int, default = 5, help ='number of classes per episode for validation')
    parser.add_argument('--support_val',type = int, default =5, help ='number of support samples per class for validation')
    parser.add_argument('--query_val', type = int, default = 15, help = 'number of query samples per class for validation')
    parser.add_argument('--pretrained_model', type = str, default = 'SS_MAE_Partial_')
    parser.add_argument('--experiment_root', type = str, default ='output')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')    
    


    #1 Setting parameter 
    args = parser.parse_args()
    batch_size = args.batch_size
    
    #2. open the dataset
    hyperparams = vars (args)
    HSIs = HSI(**hyperparams)
    data = HSIs.Normalize(args.norm_type)
    
      
    #3. devide the training and testing 
    Ex = Experiment (HSIs.gt, **hyperparams)
    

    total_OA = 0
    total_AA = 0
    total_kappa = 0
    e_A = np.zeros(HSIs.category)
    total_f_mean=0



    listOA = []
    listAA = []
    listkappa = []
    listEA = []

    
    ITER = args.run
    
    start_run=4
    for index_range in range(start_run,start_run+args.run): #proses in each run
        setup_seed (args.seed)
        
        print ("----------------------Iter :", index_range)
        Ex.patch_size = args.patch_size
        Ex.set_train_test(HSIs.gt, index_range)
        
        x_train = Ex.get_neighbour_patch (data, Ex.training_indices[0], Ex.training_indices[1], data.shape[2])
        x_test = Ex.get_neighbour_patch(data, Ex.testing_indices[0], Ex.testing_indices[1], data.shape[2])
        
        #get the label 
        y_train = HSIs.gt[tuple(Ex.training_indices)] - 1 
        y_test = HSIs.gt[tuple(Ex.testing_indices)]-1
        
        #check wether the number of band can be devided to the mask_spectral
        mask_spectral=args.mask_spectral
        x_train, x_test = chek_band(x_train, x_test, mask_spectral)
        num_band=x_train.shape[3]
        
        #4. Process of the meta learning
        #4.1 Augment the datatrain by calling augment function 
        
        x_train, y_train = x_train, y_train
        
        #4.2 Make train data loader (from data train), and test data loader (from data test). But I still confuse with the test data loader  
        
        train_dataset = MyHyperX2(x_train, y_train, **hyperparams)
        train_loader = datatorch.DataLoader(train_dataset, batch_size=int(HSIs.category*args.training_sample), shuffle=False)

        test_dataset = MyHyperX2(x_test, y_test, **hyperparams)
        test_loader = datatorch.DataLoader(test_dataset, Ex.batch_size, shuffle=False) # still question here ??? is it correct??
        print ("batch_size: ", Ex.batch_size)
    
      
        
        #4.3 load the pretrained model
        device ='cuda' if torch.cuda.is_available() else 'cpu'


        #if using train episode 2
        y_pred_logic, y_test_save=Test_episode26_1(index_range, args,HSIs.category, train_loader, test_loader,num_band)

        n = y_test_save.shape[0]
        y_pred_logic = np.reshape(y_pred_logic,((n,HSIs.category)))
            
        y_pred = np.argmax(y_pred_logic, axis=1)

        overall_acc = metrics.accuracy_score(y_pred, y_test_save)
        print("Accuracy: ", overall_acc)
        confusion_matrix = metrics.confusion_matrix(y_pred, y_test_save)
        each_acc, average_acc = Ex.get_Ave_Accuracy(confusion_matrix)
        kappa = metrics.cohen_kappa_score(y_pred, y_test_save)
        acc = overall_acc
        
        """
        Draw the result
        """

        
        X_result = Ex.draw_result(HSIs.gt,y_train,y_pred,hyperparams['dataset'])
        
        
        name='Result/Result_'+hyperparams['dataset']+'_'+str(index_range)+'_'+hyperparams['model']+'_sample_{}'.format(hyperparams['training_sample'])+'_OA_{}'.format(acc)+'.png'
        
        plt.imsave(name,X_result)
        
        
        X_train, X_gt = Ex.draw_train_data(HSIs.gt, y_train, hyperparams['dataset'])
        name_train='Result/Train_'+hyperparams['dataset']+'_'+str(index_range)+'_'+hyperparams['model']+'_sample_{}'.format(hyperparams['training_sample'])+'_OA_{}'.format(acc)+'.png'
        name_gt='Result/GT_'+hyperparams['dataset']+'.png'
 
        plt.imsave(name_train,X_train)
        plt.imsave(name_gt,X_gt)
        

        listAA.append(average_acc)
        listEA.append(each_acc)
        listOA.append(overall_acc)
        listkappa.append(kappa)


        total_OA = total_OA + overall_acc
        total_AA = total_AA + average_acc
        total_kappa = total_kappa + kappa
        e_A = e_A + each_acc
    
    OA = total_OA / ITER
    AA = total_AA / ITER
    K = total_kappa / ITER
    EA = e_A / ITER
    
    print("OA: {}".format(OA))
    print("AA: {}".format(AA))
    print("Kappa: {}".format(K))
    print("Each_Accuracy: {}".format(EA))

    if ITER > 1:
        print("ListOA: {}".format(listOA))
        print("ListAA: {}".format(listAA))
        print("ListKappa: {}".format(listkappa))
        print("ListEach_Accuracy: {}".format(listEA))