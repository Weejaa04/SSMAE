#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:57:02 2022
This file is for augmentation process

1. rotation
2. flipping 

@author: research
"""
from scipy.ndimage import rotate
import numpy as np
import math


def rotation (x, degree):
    x = rotate (x, angle = degree, reshape=False)
    
    return x

def flip(x, axis):
    if axis=='x':
        x = np.fliplr(x)
    else:
        x = np.flipud(x)
    
    return x

def mix_and_match(x, x1,x2):
    #just take out one and change with other spectral signals
    pos=[[0,0],[1,0],[2,0],[0,1],[2,1],[0,2],[1,2],[2,2]]
    
    pos_x1=pos[x1]
    pos_x2=pos[x2]
    
    temp_x1 = x[pos_x1[0],pos_x1[1],:]
    temp_x2 = x[pos_x2[0],pos_x2[1],:]
    x[pos_x1[0],pos_x1[1],:]=temp_x2
    x[pos_x2[0],pos_x2[1],:]=temp_x1
    return x

def mix_and_match2 (x, pos_x1,pos_x2):
    # just take out one and change with other spectral signals from the outer sinals
    #print ("test")
    temp_x1 = x[pos_x1[0],pos_x1[1],:]
    temp_x2 = x[pos_x2[0],pos_x2[1],:]
    x[pos_x1[0],pos_x1[1],:]=temp_x2
    x[pos_x2[0],pos_x2[1],:]=temp_x1
    return x


def Augment_data (x_train,y_train, n_category, n_patch, n_band, num_per_category ):
    '''
    This augmentation process is designed by me
    '''
    
    
    data_augment=np.zeros((n_category*num_per_category, n_patch, n_patch, n_band))
    label_augment =np.zeros((n_category*num_per_category),dtype=int)
    
    number = x_train.shape[0]
    n = number//n_category
    num_per_class = num_per_category//n
    #number_need = (num_per_category-num_per_class)//num_per_class
    #print ("number need per class: ", number_need)
    #rotation 
    j=0
    for i in range(number):
        
        need= num_per_class
        
        if need > 0:
            temp_x = x_train[i]
            data_augment[j]= temp_x
            label_augment[j]= y_train[i]
            j= j+1
            need = need -1
        
        if need > 0:
            data_augment[j]=flip(temp_x,'x')
            label_augment[j]=y_train[i]
            j = j+1
            need = need-1
            
        if need > 0:
            data_augment[j]=flip(temp_x, 'y')
            label_augment[j]=y_train[i]
            j = j+1
            need = need-1
        
        if need > 0:
            temp_x1 = rotation(temp_x,90)
            data_augment[j]=temp_x1
            label_augment[j]=y_train[i]
            j=j+1
            need = need-1
            
        if need>0:   
            data_augment[j]=flip(temp_x1,'x')
            label_augment[j]=y_train[i]
            j = j+1
            need = need-1
        
        if need> 0:
            data_augment[j]=flip(temp_x1, 'y')
            label_augment[j]=y_train[i]
            j = j+1     
            need = need-1
        
        if need > 0:
            temp_x2= rotate(temp_x, 180)
            data_augment[j]=temp_x2
            label_augment[j]=y_train[i]
            j=j+1
            need = need-1
            
        """
        if need > 0:
            data_augment[j]=flip(temp_x2,'x')
            label_augment[j]=y_train[i]
            j = j+1
            need = need-1
        
        if need > 0:
            data_augment[j]=flip(temp_x2, 'y')
            label_augment[j]=y_train[i]
            j = j+1   
            need = need-1
        """

        if need > 0:        
            temp_x3= rotation(temp_x, 270)
            data_augment[j]=temp_x3
            label_augment[j]=y_train[i]
            j=j+1
            need = need-1

        """
        if need > 0:        
            data_augment[j]=flip(temp_x3,'x')
            label_augment[j]=y_train[i]
            j = j+1
            need = need-1
        
        if need > 0:        
            data_augment[j]=flip(temp_x3, 'y')
            label_augment[j]=y_train[i]
            j = j+1  
            need = need-1

        """
        """
        if need > 0:  
            for k in range (number_need-12):
                x1=np.random.randint(0,8)
                x2=np.random.randint(0,8)
                while (x1==x2):
                    x1=np.random.randint(0,8)
                    x2=np.random.randint(0,8) 
                temp =mix_and_match(temp_x, x1, x2)
                data_augment[j]=temp
                label_augment[j]=y_train[i]
                j=j+1
      
        """
        #print ("need: ", need)
        if need>0:
            pos = [[] for _ in range(4*n_patch-4)]
            k =0
            for m in range (n_patch):
                for n in range (n_patch):
                    if (m>0) & (m<n_patch-1):
                        if (n>0) & (n<n_patch-1):
                            continue
                    pos[k]=[m,n]
                    k = k+1
            o = len(pos)          
            for k in range (need):
                
                x1=np.random.randint(0,o)
                x2=np.random.randint(0,o)
                while (x1==x2):
                    x2=np.random.randint(0,o) 
                pos_x1=pos[x1]
                pos_x2=pos[x2]
                temp =mix_and_match2(temp_x, pos_x1, pos_x2)
                #print ("j: ", j)
                data_augment[j]=temp
                #print ("data_augment j: ", data_augment[j])
                label_augment[j]=y_train[i]
                j=j+1
                
    print ("j: ",j)
    return data_augment, label_augment


def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

def Augment_data2 (x_train,y_train, n_category, n_patch, n_band ):
    '''
    This augmentation uses Augmentation process in paper 732
    They use radiation_noise
    '''
    
    
    data_augment=np.zeros((n_category*200, n_patch, n_patch, n_band))
    label_augment =np.zeros((n_category*200),dtype=int)
    
    number = x_train.shape[0]
    nlabeled = number //n_category
    n_copy = math.ceil((200 - nlabeled) / nlabeled) + 1
    #rotation 
    k = 0
    for i in range(number):
        
        for j in range (n_copy):
            data = x_train[i]
            data_augment[k]=radiation_noise(data)
            label_augment[k]=y_train[i]
            k +=1

                
    
    return data_augment, label_augment