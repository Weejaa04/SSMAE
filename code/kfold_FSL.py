#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sunday 29/10/2022

This code is to create K fold for FSL
@author: research
"""

#import numpy as np
import sklearn.model_selection 

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

from scipy import io, misc
import os
import spectral
import random

from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit


np.random.seed(58)
random.seed(58)



def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return misc.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    elif ext=='.npy':
        return np.load(dataset, allow_pickle=True)
    else:
        raise ValueError("Unknown file format: {}".format(ext))
"""

def create_kfold(name,k,size):
    
   
    
    #this function is to create the kflod
    if name=='IndianPines':
        folder='../Datasets/IndianPines/'
        gt=open_file(folder+'Indian_pines_gt.mat')['indian_pines_gt']
    elif name=='KSC':
        folder='../Datasets/KSC/'
        gt=open_file(folder+'KSC_gt.mat')['KSC_gt']
    if name=='PaviaU':
        folder='../Datasets/PaviaU/'
        gt=open_file(folder+'PaviaU_gt.mat')['paviaU_gt']
    
        
    indices=np.nonzero(gt) #this means the unused label is removed
    X=list(zip(*indices))
    y=gt[indices].ravel()
        #save the train and test indices into file 
       
    for i in range(k):
        train_index, test_index=sklearn.model_selection.train_test_split(X,train_size=size,stratify=y)
        with open('../Datasets/Data/train_test_'+name+'_'+str(i)+'.pkl','wb') as f:
            pickle.dump([train_index,test_index],f)
                
"""
def indexToAssignment(index_, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col 
        assign_1 = value % Col 
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def create_kfold_FSL(name,k,size):
    

    
    is3D=1 
    FSL =1
    
    #this function is to create the kflod
    if name=='IndianPines':
        folder='/media/research/New Volume/wija/Python/Datasets/IndianPines/'
        gt=open_file(folder+'Indian_pines_gt.mat')['indian_pines_gt']
        data=open_file(folder+'Indian_pines_corrected.mat')['indian_pines_corrected']
    elif name=='KSC':
        folder='/media/research/New Volume/wija/Python/Datasets/KSC/'
        gt=open_file(folder+'KSC_gt.mat')['KSC_gt']
        data=open_file(folder+'KSC.mat')['KSC']
    elif name=='PaviaU':
        folder='/media/research/New Volume/wija/Python/Datasets/PaviaU/'
        gt=open_file(folder+'PaviaU_gt.mat')['paviaU_gt']
        data=open_file(folder+'PaviaU.mat')['paviaU']
    elif name=='PaviaC':
        folder='/media/research/New Volume/wija/Python/Datasets/PaviaC/'
        gt=open_file(folder+'Pavia_gt.mat')['pavia_gt']
        data=open_file(folder+'Pavia.mat')['pavia']
    elif name=='IndianPines100':
        folder='/media/research/New Volume/wija/Python/Datasets/IndianPines100/'
        data=open_file(folder+'IndianPines100.mat')['IndianPines100']
        gt=open_file(folder+'IndianPines100_gt.mat')['indian_pines_gt']
    elif name=='Salinas':
        folder='/media/research/New Volume/wija/Python/Datasets/Salinas/'
        data=open_file(folder+'Salinas_corrected.mat')['salinas_corrected']
        gt=open_file(folder+'Salinas_gt.mat')['salinas_gt']

    if is3D==1:
        X=data.reshape(data.shape[0]*data.shape[1],data.shape[2])
        Y=gt.reshape(gt.shape[0]*gt.shape[1])
    else:
        X=data
        Y=gt

    indices=np.nonzero(gt) #this means the unused label is removed
    indices=list(zip(*indices))

    #save the train and test indices into file 
    print ("X shape: ", X.shape)
    print ("Y shape: ", Y.shape)
    
    if FSL ==1:
        print ("test")
        for i in range (k):

            
            train_indices, test_indices = [], []
            # if the total number per class is less then the training_sample, we omit it
            
            unique, counts=np.unique(gt,return_counts=True)
            
            for c in np.unique(gt):
                if c == 0 or counts[c]<size:
                    continue
                indices = np.nonzero(gt == c)
                X = list(zip(*indices)) # x,y features
    
                train, test = sklearn.model_selection.train_test_split(X, train_size=size, shuffle=True)
                train_indices += train
                test_indices += test       
                
            with open('/media/research/New Volume/wija/Python/Datasets/Data/FSL_train_test_'+name+'_'+str(i)+'.pkl','wb') as f:
                pickle.dump([train_indices,test_indices],f)
        
    

            



def create_kfold_equally(name,k,training_sample):
    

    flag=1   
    

    if name=='Fusarium':
        flag=0
        #print ("test")
        folder='../Datasets/Fusarium/'
        
        data=np.load(folder+'wheat_data809200.npy')
        dataX=data[:,70:326]
        Y=data[:,-1]

        dataY=Y.astype(int)
        data = dataX
        

        label=dataY
       
        gt=label

    
    #create k fold train and test

    for i in range (k):
        #print (i)
        
        train_indices, test_indices = [], []
        # if the total number per class is less then the training_sample, we omit it
    
        unique, counts=np.unique(gt,return_counts=True)
        
        for c in np.unique(gt):
            if counts[c]<training_sample:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices)) # x,y features
    
            train, test = sklearn.model_selection.train_test_split(X, train_size=training_sample)
            train_indices += train
            test_indices += test
        
        training_fix=train_indices
        testing_fix=test_indices
        with open('../Datasets/Data/train_test_'+name+'_'+str(i)+'.pkl','wb') as f:
            pickle.dump([training_fix,testing_fix],f)
        


        



def create_test_set(name, size):
    #this function is used to create test set from another test set for ablasive analysis
    if name=='IndianPines':
        folder='../Datasets/IndianPines/'
        gt=open_file(folder+'Indian_pines_gt.mat')['indian_pines_gt']
        data=open_file(folder+'Indian_pines_corrected.mat')['indian_pines_corrected']
    elif name=='KSC':
        folder='../Datasets/KSC/'
        gt=open_file(folder+'KSC_gt.mat')['KSC_gt']
        data=open_file(folder+'KSC.mat')['KSC']
    if name=='PaviaU':
        folder='../Datasets/PaviaU/'
        gt=open_file(folder+'PaviaU_gt.mat')['paviaU_gt']
        data=open_file(folder+'PaviaU.mat')['paviaU']
        
    i=0
    for i in range (10):
        #sss=StratifiedShuffleSplit(n_splits=k, train_size=size, random_state=345)
        data_name='../Datasets/Data/Ablasive Analysis/train_70_'+name+'_'+str(i)+'.pkl'
        data_name2='../Datasets/Data/Ablasive Analysis/test_70_'+name+'_'+str(i)+'.pkl'
        with open(data_name,'rb') as f:
            indices1=pickle.load(f)
        indices2= [list(t) for t in zip (*indices1)]
        X=indices1
        y=gt[tuple(indices2)].ravel()
        
        training_fix, test=sklearn.model_selection.train_test_split(X,train_size=size,stratify=y)
        with open(data_name2,'rb') as f1:
            testing_fix=pickle.load(f1)
        
        with open('../Datasets/Data/train_test_'+name+'_'+str(i)+'.pkl','wb') as f2:
            pickle.dump([training_fix,testing_fix],f2)  
   
    
def create_disjoint(name, train_size):
    #this function is to create disjoint training and test sets
    if name=='IndianPines':
        folder='../Datasets/IndianPines/'
        gt=open_file(folder+'Indian_pines_gt.mat')['indian_pines_gt']
        #data=open_file(folder+'Indian_pines_corrected.mat')['indian_pines_corrected']
    elif name=='KSC':
        folder='../Datasets/KSC/'
        gt=open_file(folder+'KSC_gt.mat')['KSC_gt']
        #data=open_file(folder+'KSC.mat')['KSC']
    if name=='PaviaU':
        folder='../Datasets/PaviaU/'
        gt=open_file(folder+'PaviaU_gt.mat')['paviaU_gt']
        #data=open_file(folder+'PaviaU.mat')['paviaU']


    if train_size>1:
        train_size=int(train_size)

    train_gt = np.copy(gt)
    test_gt = np.copy(gt)
    for c in np.unique(gt):
        mask = gt == c
        for x in range(gt.shape[0]): #
            first_half_count = np.count_nonzero(mask[:x, :])
            second_half_count = np.count_nonzero(mask[x:, :])
            try:
                ratio = first_half_count / second_half_count
                #if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                if ratio > 0.9 * train_size:
                    break
            except ZeroDivisionError:
                continue
        mask[:x, :] = 0
        train_gt[mask] = 0

    test_gt[train_gt > 0] = 0  
    
    train=np.nonzero(train_gt)
    train_indices=list(zip(*train))
    test=np.nonzero(test_gt)
    test_indices=list(zip(*test))
    
    i=0
    
    with open('../Datasets/Data/train_test_'+name+'_'+str(i)+'.pkl','wb') as f:
        pickle.dump([train_indices,test_indices],f)

    
    return train_gt, test_gt, train_indices, test_indices
    
    

def create_disjoint2(name, train_size):
    #this function is to create disjoint training and test sets
    if name=='IndianPines':
        folder='../Datasets/IndianPines/'
        gt=open_file(folder+'Indian_pines_gt.mat')['indian_pines_gt']
        #data=open_file(folder+'Indian_pines_corrected.mat')['indian_pines_corrected']
    elif name=='KSC':
        folder='../Datasets/KSC/'
        gt=open_file(folder+'KSC_gt.mat')['KSC_gt']
        #data=open_file(folder+'KSC.mat')['KSC']
    if name=='PaviaU':
        folder='../Datasets/PaviaU/'
        gt=open_file(folder+'PaviaU_gt.mat')['paviaU_gt']
        #data=open_file(folder+'PaviaU.mat')['paviaU']

    print ("sampling mode disjoint2")
    
    train_indices, test_indices = [], []
    # if the total number per class is less then the training_sample, we omit it
    
    unique, counts=np.unique(gt,return_counts=True)
    
    if train_size>1:
        for c in np.unique(gt):
            if c == 0 or counts[c]<train_size:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices)) # x,y features
    
            test=X[:-train_size]
            train = X[-train_size:]
            
            train_indices += train
            test_indices += test
            
    else:
        for c in np.unique(gt):
            n_train=int(train_size*counts[c])

            indices = np.nonzero(gt == c)
            X = list(zip(*indices)) # x,y features
    
            test=X[:-n_train]
            train = X[-n_train:]
            
            train_indices += train
            test_indices += test        

    #train_indices = list (zip(*train_indices))
    #test_indices = list(zip(*test_indices))

    i=0
    
    with open('../Datasets/Data/train_test_'+name+'_'+str(i)+'.pkl','wb') as f:
        pickle.dump([train_indices,test_indices],f)

    
    return train_indices, test_indices
    

def create_disjoint_DeepHyperX(name, train_size):
    #this function is to create disjoint training and test sets
    if name=='IndianPines':
        folder='../Datasets/IndianPines/'
        gt=open_file(folder+'Indian_pines_gt.mat')['indian_pines_gt']
        #data=open_file(folder+'Indian_pines_corrected.mat')['indian_pines_corrected']
    elif name=='KSC':
        folder='../Datasets/KSC/'
        gt=open_file(folder+'KSC_gt.mat')['KSC_gt']
        #data=open_file(folder+'KSC.mat')['KSC']
    if name=='PaviaU':
        folder='../Datasets/PaviaU/'
        gt=open_file(folder+'PaviaU_gt.mat')['paviaU_gt']
        #data=open_file(folder+'PaviaU.mat')['paviaU']

    print ("sampling mode disjoint DeepHyperX")
    
#def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.
    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels
    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    #train_gt = np.zeros_like(gt)
    #test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
 
    #elif mode == 'disjoint':
    train_gt = np.copy(gt)
    test_gt = np.copy(gt)
    for c in np.unique(gt):
        mask = gt == c
        for x in range(gt.shape[0]):
            first_half_count = np.count_nonzero(mask[:x, :])
            second_half_count = np.count_nonzero(mask[x:, :])
            try:
                ratio = first_half_count / (first_half_count + second_half_count)
                if ratio > 0.9 * train_size:
                    break
            except ZeroDivisionError:
                continue
        mask[:x, :] = 0
        train_gt[mask] = 0

    test_gt[train_gt > 0] = 0
    #else:
    #    raise ValueError("{} sampling is not implemented yet.".format(mode))
    train=np.nonzero(train_gt)
    train_indices=list(zip(*train))
    test=np.nonzero(test_gt)
    test_indices=list(zip(*test))
    
    i=0
    
    with open('../Datasets/Data/train_test_'+name+'_'+str(i)+'.pkl','wb') as f:
        pickle.dump([train_indices,test_indices],f)
        print ("test")

    
    return train_gt, test_gt, train_indices, test_indices 
    
    


def create_disjoint_368(name):
    #this function is to create disjoint training and test sets by reading the matlab file
    if name=='IndianPines':
        folder='../Datasets/IndianPines/'
        data=open_file(folder+'IP_train_test.mat')
        a=145
        b=145
        #data=open_file(folder+'Indian_pines_corrected.mat')['indian_pines_corrected']
    elif name=='KSC':
        folder='../Datasets/KSC/'
        data=open_file(folder+'KSC_train_test.mat')
        #data=open_file(folder+'KSC.mat')['KSC']
    if name=='PaviaU':
        folder='/media/research/New Volume/wija/Python/Datasets/PaviaU/'
        data=open_file(folder+'PaviaU_train_test.mat')
        #data=open_file(folder+'PaviaU.mat')['paviaU']

    print ("sampling mode disjoint paper 368")
    

    train=data['train']-1
    train=train[0,:]
    test=data['test']-1
    test=test[:,0]
    training_indices=np.unravel_index(train,(a,b))
    testing_indices=np.unravel_index(test,(a,b))
    
    training_indices_fix=list(zip(training_indices[1],training_indices[0]))
    testing_indices_fix=list(zip(testing_indices[1],testing_indices[0]))

    train_indices=training_indices_fix
    test_indices=testing_indices_fix
    
    i=0
    
    with open('../Datasets/Data/train_test_'+name+'_'+str(i)+'.pkl','wb') as f:
        pickle.dump([train_indices,test_indices],f)
        print ("test")

    
    return train_indices, test_indices 
  
def open_kfold(name,i):
    
    data_name='/media/research/New Volume/wija/Python/Datasets/Data/train_test_'+name+'_'+str(i)+'.pkl'
    
    with open(data_name,'rb') as f:
        train_index,test_index=pickle.load(f)
    
    return train_index,test_index
  

def open_kfold_FSL(name,i):
    
    data_name='D:\\Wijayanti\\Datasets\\Data\\FSL_train_test_'+name+'_'+str(i)+'.pkl'
    print ("Kfold file name: ", data_name)
    
    with open(data_name,'rb') as f:
        train_index,test_index=pickle.load(f)
    
    return train_index,test_index


def open_kfold(name,i):
    
    data_name='/media/research/New Volume/wija/Python/Datasets/Data/train_test_'+name+'_'+str(i)+'.pkl'
    
    with open(data_name,'rb') as f:
        train_index,test_index=pickle.load(f)
    
    return train_index,test_index

   
def draw_result(dataName,i=0):
     

    name=dataName
     #num_class = labels.max()+1
     
    if name=='IndianPines':
        folder='../Datasets/IndianPines/'
        gt=open_file(folder+'Indian_pines_gt.mat')['indian_pines_gt']

    elif name=='KSC':
        folder='../Datasets/KSC/'
        gt=open_file(folder+'KSC_gt.mat')['KSC_gt']

    elif name=='PaviaU':
        folder='../Datasets/PaviaU/'
        gt=open_file(folder+'PaviaU_gt.mat')['paviaU_gt']


    if dataName == 'PaviaU':

        palette = np.array([[0,0,0],
                            [216,191,216],
                            [0,255,0],
                            [0,255,255],
                            [45,138,86],
                            [255,0,255],
                            [255,165,0],
                            [159,31,239],
                            [255,0,0],
                            [255,255,0]])

    elif dataName == 'IndianPines':

        palette = np.array([[0,0,0],
                            [255,0,0],
                            [0,255,0],
                            [0,0,255],
                            [255,255,0],
                            [0,255,255],
                            [255,0,255],
                            [176,48,96],
                            [46,139,87],
                            [160,32,240],
                            [255,127,80],
                            [127,255,212],
                            [218,112,214],
                            [160,82,45],
                            [127,255,0],
                            [216,191,216],
                            [238,0,0]])

    elif dataName == 'KSC':

       palette = np.array([[0,0,0],[94, 203, 55],[55, 0, 255],
                            [217, 115, 0],
                            [179, 30, 0],
                            [0, 52, 0],
                            [72, 0, 0],
                            [255, 255, 255],
                            [145, 132, 135],
                            [255, 255, 172],
                            [255, 197, 80],
                            [60, 201, 255],
                            [11, 63, 124],
                            [0, 0, 255]])

    
    

    
    
    train_index,test_index=open_kfold(dataName,i)
    
    
    
    result=np.zeros((gt.shape[0],gt.shape[1],3),dtype=np.uint8)

    
       
    index_train=np.asarray(train_index)
    train_indices = [list(t) for t in zip(*train_index)]
    label=gt[tuple(train_indices)]
    result[index_train[:,0],index_train[:,1],:]=palette[label,:]
    


    plt.imshow(result)

    #show the classification result
    #result=np.zeros((gt.shape[0],gt.shape[1]),3,dtype=np.uint8)
    #result[self.testing_indices]=y_pred+1
    #result[self.training_indices]=y_train+1
    #plt.imshow(result)
    
    return result            
            
def draw_separate_train_test(dataName,i=0):
     

    name=dataName
     #num_class = labels.max()+1
     
    if name=='IndianPines':
        folder='../Datasets/IndianPines/'
        gt=open_file(folder+'Indian_pines_gt.mat')['indian_pines_gt']

    elif name=='KSC':
        folder='../Datasets/KSC/'
        gt=open_file(folder+'KSC_gt.mat')['KSC_gt']

    elif name=='PaviaU':
        folder='../Datasets/PaviaU/'
        gt=open_file(folder+'PaviaU_gt.mat')['paviaU_gt']


    if dataName == 'PaviaU':

        palette = np.array([[0,0,0],
                            [216,191,216],
                            [0,255,0],
                            [0,255,255],
                            [45,138,86],
                            [255,0,255],
                            [255,165,0],
                            [159,31,239],
                            [255,0,0],
                            [255,255,0]])

    elif dataName == 'IndianPines':

        palette = np.array([[0,0,0],
                            [255,0,0],
                            [0,255,0],
                            [0,0,255],
                            [255,255,0],
                            [0,255,255],
                            [255,0,255],
                            [176,48,96],
                            [46,139,87],
                            [160,32,240],
                            [255,127,80],
                            [127,255,212],
                            [218,112,214],
                            [160,82,45],
                            [127,255,0],
                            [216,191,216],
                            [238,0,0]])

    elif dataName == 'KSC':

       palette = np.array([[0,0,0],[94, 203, 55],[55, 0, 255],
                            [217, 115, 0],
                            [179, 30, 0],
                            [0, 52, 0],
                            [72, 0, 0],
                            [255, 255, 255],
                            [145, 132, 135],
                            [255, 255, 172],
                            [255, 197, 80],
                            [60, 201, 255],
                            [11, 63, 124],
                            [0, 0, 255]])

    
    

    
    
    train_index,test_index=open_kfold(dataName,i)
    
    
    
    result=np.zeros((gt.shape[0],gt.shape[1],3),dtype=np.uint8)

    
       
    index_train=np.asarray(train_index)
    train_indices = [list(t) for t in zip(*train_index)]
    label=gt[tuple(train_indices)]
    result[index_train[:,0],index_train[:,1],:]=palette[label,:]
    

    #show the the train set
    #fig=plt.figure()
    plt.imshow(result)
    ax = plt.gca()
    rect = ptc.Rectangle((100,68),20,20,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    
    plt.show()
    #fig.savefig('train4_ablasive.png', dpi=90, bbox_inches='tight')
    #plt.savefig('train.png')
    
    plt.imsave('train4_ablasive.pdf',result)
   
    
    #show the test set
    result2=np.zeros((gt.shape[0],gt.shape[1],3),dtype=np.uint8)
    index_test=np.asarray(test_index)
    test_indices = [list(t) for t in zip(*test_index)]
    label=gt[tuple(test_indices)]
    result2[index_test[:,0],index_test[:,1],:]=palette[label,:]    
    plt.figure()
    plt.imshow(result2)
    plt.imsave('test_ablasive.pdf',result2)
    

    #show the classification result
    #result=np.zeros((gt.shape[0],gt.shape[1]),3,dtype=np.uint8)
    #result[self.testing_indices]=y_pred+1
    #result[self.training_indices]=y_train+1
    #plt.imshow(result)
    
    return result



    
#create_kfold_equally('Fusarium',10,75828)
            
#train_index,test_index=open_kfold('Fusarium',0)        
 
#create_kfold_FSL('PaviaC',5,5)
#train_index,test_index=open_kfold_FSL('PaviaC',0)


#create_kfold_FSL('IndianPines',5,5)
#train_index,test_index=open_kfold_FSL('IndianPines',0)

#train_index1,test_index1=open_kfold_FSL('IndianPines',1)
      
#create_kfold_FSL('Salinas',5,5)
#train_index,test_index=open_kfold_FSL('Salinas',0)

#create_disjoint2('IndianPines',0.5)

#train_index,test_index=open_kfold('alldata',0)   

#draw_separate_train_test('IndianPines',0)

#train_indices,test_indices=create_disjoint_368('IndianPines')
#draw_separate_train_test('IndianPines',0)
        

