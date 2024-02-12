# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:51:51 2020

@author: 22594658
"""

import random
import numpy as np
#from sklearn.metrics import confusion_matrix
#import sklearn.model_selection
#from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from Utils import zeroPadding 
from operator import truediv
import pickle
from kfold_FSL import open_kfold_FSL
import sklearn.model_selection

from sklearn.decomposition import PCA




def indexToAssignment(index_, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col 
        assign_1 = value % Col 
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def ReflectPadding_3D (data, half_patch):
    """
    this for reflection 
    """
    nband = data.shape[2]
    a = data.shape[0]
    b = data.shape[1]
    
    output = np.zeros((a+2*half_patch,b+2*half_patch,nband))
   
    
    for i in range (nband):
        temp = data [:,:,i]
       
        out = np.pad(temp,((half_patch, half_patch),(half_patch,half_patch)),'reflect')
        output[:,:,i] =out
        
    
    return output

class Experiment():
    
    """
    This class is used as a framework for the experiment
    
    """
    
    def __init__(self,gt,**parameters):
        super(Experiment,self).__init__()
        
        self.training_sample=parameters['training_sample']
        self.sampling_mode=parameters['sampling_mode']
        self.name=parameters['dataset']

        self.padded=parameters['padded']
        self.val_sample=parameters['val_size']
        
        if self.sampling_mode is None:
            self.sampling_mode = 'random'
            
        self.patch_size=parameters['patch_size']
        
        if self.patch_size is None:
            self.patch_size = 7
            
        if self.padded is None:
            self.padded=1
            
        
        self.epoch=parameters['epoch']
        self.lr=parameters['lr']
        self.batch_size=parameters['batch_size']
        self.training_indices={} #save the indices of the training sample 
        self.testing_indices={} #save the indices of the testing sample
        #self.train_gt=np.zeros_like(gt)
        #self.test_gt=np.zeros_like(gt)
        self.val_indices={}
        self.training_standard={}
        
    def set_train_test(self,gt,i,num_train=None):
        indices=np.nonzero(gt) #this means the unused label is removed
        X=list(zip(*indices))
        y=gt[indices].ravel()
        

        
        if self.training_sample > 1:
            self.training_sample = int (self.training_sample)
        
        if self.sampling_mode == 'random':
            training_indices, testing_indices = sklearn.model_selection.train_test_split(X, train_size=self.training_sample,stratify=y)
            
            self.training_indices = [list(t) for t in zip(*training_indices)]
            self.testing_indices = [list(t) for t in zip(*testing_indices)]
            #self.train_gt[self.training_indices]=gt[self.training_indices]
            #self.test_gt[self.testing_indices]=gt[self.testing_indices]
            #print (self.train_gt)
            #print(self.test_gt)
        
        elif self.sampling_mode =='kfold':

            print ("i",i)
            training_indices, testing_indices=open_kfold(self.name,i)

            self.training_indices=[list(t) for t in zip (*training_indices)]
            self.testing_indices=[list(t) for t in zip (*testing_indices)]  

        elif self.sampling_mode =='kfold_FSL':

            print ("i",i)
            training_indices, testing_indices=open_kfold_FSL(self.name,i)

            self.training_indices=[list(t) for t in zip (*training_indices)]
            self.testing_indices=[list(t) for t in zip (*testing_indices)]          
        
        elif self.sampling_mode=='fixed': #means the sampling mode fix per class
            print ("Sampling {} with train size = {}".format(self.sampling_mode,self.training_sample))
            train_indices, test_indices = [], []
            # if the total number per class is less then the training_sample, we omit it
            
            unique, counts=np.unique(gt,return_counts=True)
            
            for c in np.unique(gt):
                if c == 0 or counts[c]<self.training_sample:
                    continue
                indices = np.nonzero(gt == c)
                X = list(zip(*indices)) # x,y features

                train, test = sklearn.model_selection.train_test_split(X, train_size=self.training_sample)
                train_indices += train
                test_indices += test
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            #self.train_gt[train_indices] = gt[train_indices]
            #self.test_gt[test_indices] = gt[test_indices] 
            self.training_indices=train_indices
            self.testing_indices=test_indices
        
        elif self.sampling_mode=='standard': #divide the training and testing data based on the standard of GRSS DASE initiative
            print ("Sampling with train the standard data of GRSS DASE initiative")
            train_indices, test_indices = [], []
            # if the total number per class is less then the training_sample, we omit it
            
            #unique, counts=np.unique(gt,return_counts=True)
            
            
            for c in np.unique(gt):
                if c == 0: 
                    continue
                indices = np.nonzero(gt == c)
                X = list(zip(*indices)) # x,y features

                train, test = sklearn.model_selection.train_test_split(X, train_size=num_train[c])
                train_indices += train
                test_indices += test
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            #self.train_gt[train_indices] = gt[train_indices]
            #self.test_gt[test_indices] = gt[test_indices] 
            self.training_indices=train_indices
            self.testing_indices=test_indices
        
        
        elif self.sampling_mode=='manual': # I try to create the sampling mode manual 
            print ("sampling mode manual")
            train = {}
            test = {}
            groundTruth=gt
            proptionVal=1-self.training_sample
            m = np.amax(groundTruth) #return the max of groundtruth (the number of class)
            for i in range(m):
                indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1] #return the index of class i+1
                np.random.shuffle(indices) #these indics is shuffled
                #labels_loc[i] = indices
                nb_val = int(proptionVal * len(indices)) #80% of indices
                train[i] = indices[:-nb_val] #consist the index of the training
                test[i] = indices[-nb_val:] #consist the indices of the testing
#    whole_indices = []
            train_indices = []
            test_indices = []
            for i in range(m):
#        whole_indices += labels_loc[i]
                train_indices += train[i] #concate the training data
                test_indices += test[i] #concate the testing data
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
            train_assign = indexToAssignment(train_indices, gt.shape[0], gt.shape[1]) #train_assign is in dictionary
            train_assign_list=list(train_assign.values()) #the list version of train assign
            train_indices_test=[list(t) for t in zip(*train_assign_list)]
            self.training_indices=train_indices_test            
            
            test_assign = indexToAssignment(test_indices, gt.shape[0], gt.shape[1])
            test_assign_list=list(test_assign.values())
            self.testing_indices=[list(t) for t in zip(*test_assign_list)]


    def set_train_test_val(self,gt,i):
        indices=np.nonzero(gt) #this means the unused label is removed
        X=list(zip(*indices))
        y=gt[indices].ravel()
        

        
        if self.training_sample > 1:
            self.training_sample = int (self.training_sample)
        
        if self.sampling_mode == 'random':
            training_indices, XX=sklearn.model_selection.train_test_split(X,train_size=self.training_sample,stratify=y)
            validation_indices,testing_indices=sklearn.model_selection.train_test_split(XX,train_size=self.val_sample)
            
            self.training_indices=[list(t) for t in zip (*training_indices)]
            
            self.testing_indices=[list(t) for t in zip (*testing_indices)]
            self.val_indices=[list(t) for t in zip (*validation_indices)]
            #self.train_gt[self.training_indices]=gt[self.training_indices]
            #self.test_gt[self.testing_indices]=gt[self.testing_indices]
            #print (self.train_gt)
            #print(self.test_gt)
        elif self.sampling_mode=='fixed': #means the sampling mode fix per class
            print ("Sampling {} with train size = {}".format(self.sampling_mode,self.training_sample))
            train_indices, val_indices, test_indices = [],[], []
            # if the total number per class is less then the training_sample, we omit it
            
            unique, counts=np.unique(gt,return_counts=True)
            
            for c in np.unique(gt):
                if c == 0 or counts[c]<self.training_sample:
                    continue
                indices = np.nonzero(gt == c)
                X = list(zip(*indices)) # x,y features

                train, XX = sklearn.model_selection.train_test_split(X, train_size=self.training_sample)
                val,test=sklearn.model_selection.train_test_split(XX,train_size=self.val_sample)
                train_indices += train
                val_indices+=val
                test_indices += test
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            val_indices=[list(t) for t in zip(*val_indices)]
            #self.train_gt[train_indices] = gt[train_indices]
            #self.test_gt[test_indices] = gt[test_indices] 
            self.training_indices=train_indices
            self.testing_indices=test_indices
            self.val_indices=val_indices
        
        
        elif self.sampling_mode=='manual': # I try to create the sampling mode manual 
            print ("sampling mode manual")
            train = {}
            test = {}
            groundTruth=gt
            proptionVal=1-self.training_sample
            m = np.amax(groundTruth) #return the max of groundtruth (the number of class)
            for i in range(m):
                indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1] #return the index of class i+1
                np.random.shuffle(indices) #these indics is shuffled
                #labels_loc[i] = indices
                nb_val = int(proptionVal * len(indices)) #80% of indices
                train[i] = indices[:-nb_val] #consist the index of the training
                test[i] = indices[-nb_val:] #consist the indices of the testing
#    whole_indices = []
            train_indices = []
            test_indices = []
            for i in range(m):
#        whole_indices += labels_loc[i]
                train_indices += train[i] #concate the training data
                test_indices += test[i] #concate the testing data
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
            train_assign = indexToAssignment(train_indices, gt.shape[0], gt.shape[1]) #train_assign is in dictionary
            train_assign_list=list(train_assign.values()) #the list version of train assign
            train_indices_test=[list(t) for t in zip(*train_assign_list)]
            self.training_indices=train_indices_test            
            
            test_assign = indexToAssignment(test_indices, gt.shape[0], gt.shape[1])
            test_assign_list=list(test_assign.values())
            self.testing_indices=[list(t) for t in zip(*test_assign_list)]

    def set_train_test_SS(self, data, i, num_train=None):

        temp = data[:, :, 0]
        temp = temp + 1
        indices = np.nonzero(temp)  # this means the unused label is removed
        X = list(zip(*indices))

        if self.training_sample > 1:
            self.training_sample = int(self.training_sample)

        if self.sampling_mode == 'random':
            training_indices, testing_indices = sklearn.model_selection.train_test_split(X,
                                                                                         train_size=self.training_sample)

            self.training_indices = [list(t) for t in zip(*training_indices)]
            self.testing_indices = [list(t) for t in zip(*testing_indices)]
            # self.train_gt[self.training_indices]=gt[self.training_indices]
            # self.test_gt[self.testing_indices]=gt[self.testing_indices]
            # print (self.train_gt)
            # print(self.test_gt)

    def set_train_test_SS_Partial(self, data, i, part_data, num_train=None):

        temp = data[:, :, 0]
        temp = temp + 1
        indices = np.nonzero(temp)  # this means the unused label is removed
        X = list(zip(*indices))

        if self.training_sample > 1:
            self.training_sample = int(self.training_sample)

        if self.sampling_mode == 'random':
            training_indices, testing_indices = sklearn.model_selection.train_test_split(X,
                                                                                         train_size=self.training_sample,
                                                                                         shuffle=True)
            n = len(training_indices)
            m = len(testing_indices)
            n = n // part_data
            m = m // part_data
            training_indices = training_indices[0:n]
            testing_indices = testing_indices[0:m]
            self.training_indices = [list(t) for t in zip(*training_indices)]
            self.testing_indices = [list(t) for t in zip(*testing_indices)]

    def save_history(self,history):
        #print (history)
        
        with open('history.pkl','wb') as file:
            pickle.dump(history,file)
        
        
    
    def draw_result(self,gt,labels,y_pred,dataName):
        
        #labels=y_train.argmax(axis=1)
        #y_val=y_val.argmax(axis=1)
        draw_patch = 0
        
        #num_class = labels.max()+1
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

        elif dataName == 'IndianPines' or dataName == 'IndianPines100' or dataName == 'Salinas' or dataName == 'Salinas100':

            palette = np.array([[0,0,0],
                                [255,0,0], #1
                                [0,255,0], #2
                                [0,0,255], #3
                                [255,255,0], #4
                                [0,255,255], #5
                                [255,0,255], #6
                                [176,48,96], #7 
                                [46,139,87], #8
                                [160,32,240], #9
                                [255,127,80], #10
                                [127,255,212], #11
                                [218,112,214], #12
                                [160,82,45], #13
                                [127,255,0], #14
                                [216,191,216], #15
                                [238,0,0]]) #16 like red

        elif dataName == 'KSC':

            palette = np.array([[0,0,0],
                                [94, 203, 55],
                                [255, 0, 255],
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
            
        elif dataName == 'PaviaC':

            palette = np.array([[0,0,0],
                                [94, 203, 55],
                                [255, 0, 255],
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


      

        result=np.zeros((gt.shape[0],gt.shape[1],3),dtype=np.uint8)
        groundtruth=np.zeros((gt.shape[0],gt.shape[1],3),dtype=np.uint8)
        
        index_test=np.asarray(self.testing_indices)
        #index_val=np.asarray(self.val_indices)
        index_train=np.asarray(self.training_indices)
        y_pred = y_pred.reshape(y_pred.shape[0])
        result[index_test[0,:],index_test[1,:],:]=palette[y_pred+1,:]
        #result[index_val[0,:],index_val[1,:],:]=palette[y_val+1,:]
        result[index_train[0,:],index_train[1,:],:]=palette[labels+1,:]
        
        groundtruth[:,:,:]=palette[gt,:]
        
        if draw_patch == 1:

            result = self.get_rectangle (result, index_train)
        
        plt.imshow (groundtruth)
        plt.imshow(result)
    
       
        return result


    def draw_train_data(self,gt,labels,dataName):
        
        #labels=y_train.argmax(axis=1)
        #y_val=y_val.argmax(axis=1)
        
        
        #num_class = labels.max()+1
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

        elif dataName == 'IndianPines' or dataName == 'IndianPines100' or dataName == 'Salinas' or dataName == 'Salinas100':

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

            palette = np.array([[0,0,0],
                                [94, 203, 55],
                                [255, 0, 255],
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
        elif dataName == 'PaviaC':

            palette = np.array([[0,0,0],
                                [94, 203, 55],
                                [255, 0, 255],
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

      

        result=np.zeros((gt.shape[0],gt.shape[1],3),dtype=np.uint8)
        groundtruth=np.zeros((gt.shape[0],gt.shape[1],3),dtype=np.uint8)
        
        index_train=np.asarray(self.training_indices)

        result[index_train[0,:],index_train[1,:],:]=palette[labels+1,:]
        
        groundtruth[:,:,:]=palette[gt,:]

        
        plt.imshow (groundtruth)
        plt.imshow(result)
    
       
        return result, groundtruth


    def get_rectangle (self, result, index_train):
        #print ("index_train: ", index_train)
        red = [255,0,0]
        patch = self.patch_size
        half_patch = patch //2
        
        x_max = result.shape[0]-1
        y_max=result.shape[1]-1
        num = index_train.shape[1]
        
        for i in range (num):
            x,y = index_train[0,i], index_train[1,i]
            #draw the horizontal line and vertical line
            for j in range(patch):
                x1 = x-half_patch
                x2 = x+half_patch
                y1 = y-half_patch
                y2 = y+half_patch
                if x1<0:
                    x1 =0
                if y1<0:
                    y1=0
                if x2>x_max:
                    x2 = x_max
                if y2>y_max:
                    y2=y_max
                try: 
                    result[x1+j,y2,:]=red
                    result[x1+j,y1,:]=red
                    result[x2,y1+j,:]=red
                    result[x1,y1+j,:]=red
                except:
                    print ("x1="+ str(x1) +" x2="+ str(x2) +" y1=" +str(y1)+ " y2="+str(y2)+" j=" +str(j))
         
        return result
        
    
    def get_neighbour_patch (self,data,x_pos, y_pos, n_bands):
        
        
        print ("Padding type: ", self.padded)
        half_patch = self.patch_size//2
        n=len(x_pos)
        
        selected_patch=np.zeros((n, self.patch_size, self.patch_size, n_bands)) # number 200 is temporary change it with the spectral dimension
        if self.padded == 1:
            print ("Calling zero padding")
            #this is the code when the data is padded
            matrix=zeroPadding.zeroPadding_3D(data,half_patch) #add 0 in every side of the data
            #x_pos=np.asarray(x_pos) #change the list to array
            #x_pos=x_pos+half_patch
            #y_pos=np.asarray(y_pos)
            #y_pos=y_pos+half_patch
            
            for i in range (n): #if padded the index are changing
                selected_rows = matrix[range(x_pos[i],x_pos[i]+2*half_patch+1), :]
                selected_patch[i] = selected_rows[:, range(y_pos[i], y_pos[i]+2*half_patch+1)]
        elif self.padded == 2:
            print ("Calling reflect padding")
            print ("Half patch: ", half_patch)
            matrix = ReflectPadding_3D (data, half_patch)
            for i in range (n): #if padded the index are changing
                selected_rows = matrix[range(x_pos[i],x_pos[i]+2*half_patch+1), :]
                selected_patch[i] = selected_rows[:, range(y_pos[i], y_pos[i]+2*half_patch+1)]
        else:
            matrix=data
            for i in range (n): #if padded the index are changing
                selected_rows = matrix[range(x_pos[i]-half_patch,x_pos[i]+half_patch+1), :]
                selected_patch[i] = selected_rows[:, range(y_pos[i]-half_patch, y_pos[i]+half_patch+1)]
                
        return selected_patch
                


    def preprocess(self, img, process,n_components=10):
        
        
        if process=='PCA':
            #change the demension of the data into 2 dimensional
            data = img.reshape(np.prod(img.shape[:2]),np.prod(img.shape[2:])) #reshape data from 145*145*200 into 21025*200
            #do the PCA

            pca=PCA(n_components)
            pca.fit(data)
            x_pca=pca.transform(data)
            #return the dimension into 3D
            data=x_pca.reshape(img.shape[0],img.shape[1],n_components)
        

        return data
    
    def get_Ave_Accuracy(self, confusion_matrix):
        #counter = confusion_matrix.shape[0]
        list_diag = np.diag(confusion_matrix)
        list_col_sum = np.sum(confusion_matrix, axis=0)
        each_acc = np.nan_to_num(truediv(list_diag, list_col_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc
    
    def get_Ave_Accuracy2(self, confusion_matrix):
        #counter = confusion_matrix.shape[0]
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc