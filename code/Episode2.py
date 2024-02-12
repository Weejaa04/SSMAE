#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:20:38 2023

@author: research
"""


import torch
import numpy as np
from MetaData import MetaData, MD_distance, MD_distance_test1, build_class_reps_and_covariance_estimates,\
                     get_meta_loader, Predict_Data, euclidean_dist, MD_distance_test1_1, MD_distance_test1_2
import torch.nn as nn
import torch.optim as optim  


from sklearn.preprocessing import normalize

from argparse import Namespace
from torch.autograd import Variable

from sklearn import metrics
import math
from model import *
from model_meta_learn import MAE_Embedding2


from torchsummary import summary


from sklearn.svm import SVC


torch.autograd.set_detect_anomaly(True)

def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    length_data = len(labels.data)
    return acc, correct, length_data


  
def Prototype_Initialization (features, labels):
    #this function is to get global prototype initialization 

    
    class_representations, class_precision_matrices = build_class_reps_and_covariance_estimates(features, labels)
    
    #the means of each class
    class_means = torch.stack(list(class_representations.values())).squeeze(1)
    class_precision_matrices = torch.stack(list(class_precision_matrices.values()))
    
    return class_means, class_precision_matrices  


def Compute_Similarity2 (global_class_means, local_class_means, global_class_precision_matrices, local_class_precision_matrices):
    """
    Different to Compute_Similarity, to compute the covariance, I used correlation matric distance like in [794] and
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html
    """
    #test this function latter
    H1 = euclidean_dist( local_class_means, global_class_means)
    H1 = -H1
    a = local_class_precision_matrices
    b = global_class_precision_matrices
    temp_local_matrices = torch.reshape(a, (a.shape[0], a.shape[1]*a.shape[2]))
    temp_global_matrices = torch.reshape(b, (b.shape[0], b.shape[1]*b.shape[2]))
    #del a, b, local_class_means, global_class_means, local_class_precision_matrices, global_class_precision_matrices

    H2 = torch.cdist(temp_local_matrices, temp_global_matrices, p=2)
    H2 = -H2
    
    return H1, H2    
 
def Compute_Similarity (global_class_means, local_class_means, global_class_precision_matrices, local_class_precision_matrices):
    
    #test this function latter
    H1 = euclidean_dist( local_class_means, global_class_means)
    H1 = -H1
    a = local_class_precision_matrices
    b = global_class_precision_matrices
    temp_local_matrices = torch.reshape(a, (a.shape[0], a.shape[1]*a.shape[2]))
    temp_global_matrices = torch.reshape(b, (b.shape[0], b.shape[1]*b.shape[2]))
    H2 = euclidean_dist(temp_local_matrices, temp_global_matrices)
    H2 = -H2
    
    return H1, H2


def Decide_using_Confidence_Level(y_pred_logic1, y_pred_logic2, y_test):
    
    size = y_pred_logic1.shape[1]
    sorted_y1 = np.sort(y_pred_logic1, axis=1)
    y_pred1 = np.argmax(y_pred_logic1, axis=1)
    y1= sorted_y1[:,size-1]
    y_sum1 = np.sum(y_pred_logic1, axis=1)
    conf1 = y1/y_sum1
    
    sorted_y2 = np.sort(y_pred_logic2, axis=1)
    y_pred2 = np.argmax (y_pred_logic2, axis=1)
    y2 = sorted_y2[:, size-1]
    y_sum2 = np.sum (y_pred_logic2, axis=1)
    conf2 = y2/y_sum2
    print ("test")

def Train_episode23_1 (index_range,args,aug_data, aug_loader,nCategory, feature_encoder, train_loader, test_loader):
    #From Train_episode3_1, but try to seperate train and test 
    
    
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    
    EPISODE = args.episode
    TEST_CLASS_NUM = args.class_for_tr
    SHOT_NUM_PER_CLASS = args.support_tr
    QUERY_NUM_PER_CLASS = args.query_tr
    
    

    feature_encoder.to(device)
    #protoNet = model.encoder
        
    feature_encoder_optim = optim.Adam(feature_encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=feature_encoder_optim, gamma=args.lrG,step_size=args.lrS)

    
    print ("Training....")
    
    last_accuracy = 0.0    
    best_episode = 0 
    train_loss = []
    test_acc = []
    total_hit , total_num = 0.0, 0.0
    test_acc_list = []
    
    crossEntropy = nn.CrossEntropyLoss().to(device)

    
    #Get the initial Global Prototypical Representation, like in Eq 2 paper 759
    aug_datas, aug_labels = aug_loader.__iter__().next()
    
    aug_datas = aug_datas.to(device)

    
    loss_min = 1000

    for episode in range (EPISODE): #how if I try to use only 200 episode?
        feature_encoder.zero_grad()
        aug_features = feature_encoder(aug_datas.float())
        global_class_means, global_class_precision_matrices = Prototype_Initialization(aug_features, aug_labels)


        #FSL for target domain because I do not have source domain
        meta_data = MetaData(aug_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        #make support data loader
        support_dataloader =get_meta_loader(meta_data, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle = "False")
        query_dataloader = get_meta_loader(meta_data, num_per_class=QUERY_NUM_PER_CLASS, split ="test", shuffle="True")
        
        supports, support_labels = support_dataloader.__iter__().next()
        querys, query_labels = query_dataloader.__iter__().next()
        
        supports, querys = supports.to(device), querys.to(device)
        
        support_features = feature_encoder (supports.float())
        query_features = feature_encoder(querys.float())
        
        #Get local representation like equation 3 paper 759 
        local_class_means, local_class_precision_matrices = Prototype_Initialization(support_features, support_labels)
        


        support_classes=np.unique(support_labels)
        selected_classes = meta_data.the_selected_classes()
        real_class = dict(zip(support_classes, selected_classes))
        #update the global representation based on the difference between local representation and global representation 
        #1. Compute the similarity between local representation and global representation
        H1, H2 = Compute_Similarity (global_class_means, local_class_means, global_class_precision_matrices, local_class_precision_matrices) 



        #2. Normalize like in Equation 6 

        min_H1 = H1.min(axis =1)
        min_H2 = H2.min(axis = 1)
        max_H1 = H1.max(axis=1)
        max_H2 = H2.max(axis=1)
        
        cloneH1= H1.clone()
        cloneH2 = H2.clone()
        #change the value into between zero and one 
        for i in range (TEST_CLASS_NUM):
            H1[i,:]=(cloneH1[i,:]-min_H1.values[i])/(max_H1.values[i]-min_H1.values[i])
            H2[i,:]=(cloneH2[i,:]-min_H2.values[i])/(max_H2.values[i]-min_H2.values[i])

        sum_H1 = torch.sum(H1, axis = 1)
        sum_H2 = torch.sum(H2, axis = 1)
        H1 = torch.transpose(H1, 0, 1)
        H2 = torch.transpose(H2, 0, 1)
        normed_H1 = torch.div (H1, sum_H1)
        normed_H2 = torch.div (H2, sum_H2)
        normed_H1 = torch.transpose(normed_H1, 0, 1)
        normed_H2 = torch.transpose(normed_H2, 0, 1)

        #3.update the global representation, only update the class ci, eq (7)
        
        n = len(support_classes)
        
        G = global_class_means.clone()
        M = global_class_precision_matrices.clone()
        

        
        for i in range (n):
            ci = real_class[i]
            global_class_means[ci,:]=torch.matmul(normed_H1[i],G).view(1,args.emb_size)
            temp = normed_H2[i]
            global_class_precision_matrices[ci,:,:] = torch.matmul(temp, M.view(nCategory,args.emb_size*args.emb_size)).view(1,args.emb_size, args.emb_size)

        #4. get the label of the queryes
        query_preds = Predict_Data (global_class_means[selected_classes,:], global_class_precision_matrices[selected_classes,:,:], query_features)
        query_loss = crossEntropy(query_preds, query_labels.type(torch.LongTensor).to(device))

        #5. fet discriminat Loss

        #update parameters
        loss = query_loss
        
        loss.backward()
        feature_encoder_optim.step()
        
        total_hit += torch.sum(torch.argmax(query_preds, dim = 1).cpu() ==query_labels).item()
        total_num += querys.shape[0]

        
        if (episode+1) %100 == 0:
            train_loss.append (loss.item())#not sure this is here
        
            print('episode {:>3d}: loss: {:6.4f}, query_sample_num: {:>3d}, acc {:6.4f}'.format(episode + 1, loss.item(), querys.shape[0], total_hit / total_num))        
        
        if loss.item() < loss_min:
            # save networks
            torch.save(feature_encoder.state_dict(),str( "checkpoints/"+str(args.model)+"_finetune_" + str(args.dataset)+'_'+str(args.patch_size)+'x'+str(args.patch_size)+'_'+str(args.mask_spatial)+ 'x'+str(args.mask_spectral)+'_'+str(args.mask_ratio)+'_'+str(args.emb_size)+"_iter_"+str(index_range)+".pkl"))
            print("save networks for episode:",episode+1)
            
            loss_min = loss.item()
            best_episode = episode
 


            print('best episode:[{}], loss={}'.format(best_episode + 1, loss_min))        
 

    return 0


def Test_episode26_1 (index_range,args, nCategory, train_loader, test_loader, num_band):
    """ The same as Test_episode3_2, but with simpler implementation"""
    
    #for testing outside episode 
    print ("Testing")
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    SHOT_NUM_PER_CLASS = args.support_tr
    
    flag = 0 #if using convolution using flag = 1
    if flag ==0:
        model = MAE_SS_Encoder(patch_size = args.patch_size, mask_patch_size = args.mask_spatial, 
                             mask_spectral=args.mask_spectral,
                             emb_dim = args.emb_size, 
                             num_layer = 12, 
                             num_head= 3, 
                             mask_ratio = args.mask_ratio, 
                             num_band=num_band)
    else:
        model = MAE_SS_Encoder2(patch_size = args.patch_size, mask_patch_size = args.mask_spatial, 
                     mask_spectral=args.mask_spectral,
                     emb_dim = args.emb_size, 
                     num_layer = 12, 
                     num_head= 3, 
                     mask_ratio = args.mask_ratio, 
                     num_band=num_band)
        
    feature_encoder = MAE_Embedding2(model)
    model_proto_path = str("checkpoints/"+str(args.model)+"_finetune_" + str(args.dataset)+'_'+str(args.patch_size)+'x'+str(args.patch_size)+'_'+str(args.mask_spatial)+ 'x'+str(args.mask_spectral)+'_'+str(args.mask_ratio)+'_'+str(args.emb_size)+"_iter_"+str(index_range)+".pkl")    
    print ("model checkpoints: ", model_proto_path)
    
    feature_encoder.load_state_dict(torch.load(model_proto_path))
    feature_encoder.to(device)
    feature_encoder.eval()
    total_rewards = 0 
    counter = 0 
    accuracies = []
    predicts = np.array([], dtype=np.int64)
    labels = np.array ([], dtype=np.int64)
    
    test_labels_all = np.array([], dtype=np.int64)
    
    train_datas, train_labels = train_loader.__iter__().next()
    train_datas = train_datas.to(device)
    train_features = feature_encoder(train_datas.float())
    predict_logits_all=np.array([[]],dtype=np.int64)

    
    flag = 1
    
    class_representations,class_precision_matrices, number_of_classes, class_means = MD_distance_test1_1(train_features, train_labels)
    
    for test_datas, test_labels in test_loader:
        batch_size = test_labels.shape[0]
        test_datas = test_datas.to(device)
        test_features = feature_encoder(test_datas.float())
        
        predict_logits=MD_distance_test1_2(number_of_classes, class_means, class_precision_matrices,test_features)
        predict_logits_temp = predict_logits.cpu().detach().numpy()
        
        predict_logits_all=np.append(predict_logits_all, predict_logits_temp)


        test_labels = test_labels.numpy()
        

        labels = np.append(labels, test_labels)
        
   
    return predict_logits_all, labels

