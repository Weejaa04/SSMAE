#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 11:49:04 2022

@author: research
"""
import torch
import random
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset

class MetaData(object): #here to select support and query like class Task in paper 732

    def __init__(self, data, num_classes, shot_num, query_num):
        self.data = data
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        sorted_data = sorted(list(data)) #just sort the classes

        class_list = random.sample(sorted_data, self.num_classes) #from all the classes available in source dataset, i.e., 21, only take 9 classes randomly in case of pavia university
        #e,g :[0,6,8]
        self.class_list = class_list
        
        labels = np.array(range(len(class_list)))#give a new label to the selected list
        #e.g : [0 1 2]

        labels = dict(zip(class_list, labels))

        samples = dict()

        self.support_datas = []
        self.query_datas = []
        self.support_labels = []
        self.query_labels = []
        for c in class_list:
            temp = self.data[c]  # list
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])
            
            #print ("c: ", c)
            #print ("shot num: ", shot_num)
            self.support_datas += samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]

            self.support_labels += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]


    def the_selected_classes(self):
        return self.class_list

            

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class' '''
    # 参数：
    #   num_per_class: 每个类的样本数量
    #   num_cl: 类别数量
    #   num_inst：support set或query set中的样本数量
    #   shuffle：样本是否乱序
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

def get_meta_loader(meta_data, num_per_class=1, split='train',shuffle = False):

    #  this function is to generate support loader and query loader 
    
    dataset = Meta_dataset(meta_data, split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, meta_data.num_classes, meta_data.support_num, shuffle=shuffle) # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, meta_data.num_classes, meta_data.query_num, shuffle=shuffle) # query set

    loader = DataLoader(dataset, batch_size=num_per_class*meta_data.num_classes, sampler=sampler)

    return loader

def Predict_Data (class_means, class_precision_matrices, features ):
    
    num_classes = class_means.size(0)
    num_targets = features.size(0)
    
    
    repeated_target = features.repeat(1,num_classes).view(-1,class_means.size(1)) #repeat each feature based on the number of class
    repeated_class_means = class_means.repeat (num_targets,1) #repeat the class means
    repeated_difference =(repeated_class_means-repeated_target)
    repeated_difference=repeated_difference.view(num_targets, num_classes, 
                                                 repeated_difference.size(1)).permute(1,0,2)
    
    first_half = torch.matmul(repeated_difference, class_precision_matrices)
    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1
    #sample_logits is the minus of mahalanobis distance
    #del class_means, class_precision_matrices, features
    return sample_logits

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def MD_distance(support_feature, support_labels, query_features):
    NUM_SAMPLES=1
    class_representations, class_precision_matrices = build_class_reps_and_covariance_estimates(support_feature, support_labels)

    class_means = torch.stack(list(class_representations.values())).squeeze(1)
    class_precision_matrices = torch.stack(list(class_precision_matrices.values()))

    # grabbing the number of classes and query examples for easier use later in the function
    number_of_classes = class_means.size(0)
    number_of_targets = query_features.size(0)

    repeated_target = query_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
    repeated_class_means = class_means.repeat(number_of_targets, 1)
    repeated_difference = (repeated_class_means - repeated_target)
    repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                   repeated_difference.size(1)).permute(1, 0, 2)
    first_half = torch.matmul(repeated_difference, class_precision_matrices)
    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

    # return split_first_dim_linear(sample_logits, [NUM_SAMPLES, query_features.shape[0]])
    return sample_logits

def MD_distance_test1(support_feature, support_labels, query_features):
    NUM_SAMPLES=1
    class_representations, class_precision_matrices = build_class_reps_and_covariance_estimates(support_feature, support_labels)

    class_means = torch.stack(list(class_representations.values())).squeeze(1)
    class_precision_matrices = torch.stack(list(class_precision_matrices.values()))

    # grabbing the number of classes and query examples for easier use later in the function
    number_of_classes = class_means.size(0)
    number_of_targets = query_features.size(0)

    repeated_target = query_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
    repeated_class_means = class_means.repeat(number_of_targets, 1)
    repeated_difference = (repeated_class_means - repeated_target)
    repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                   repeated_difference.size(1)).permute(1, 0, 2)
    first_half = torch.matmul(repeated_difference, class_precision_matrices)
    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

    # return split_first_dim_linear(sample_logits, [NUM_SAMPLES, query_features.shape[0]])
    return sample_logits,class_representations, class_precision_matrices

def MD_distance_test1_1(support_feature, support_labels):
    """
    The same as MD_distance_test1, but devided into 2: MD_distance_test1_1 and MD_distance_test1_2
    """
    NUM_SAMPLES=1
    class_representations, class_precision_matrices = build_class_reps_and_covariance_estimates(support_feature, support_labels)

    class_means = torch.stack(list(class_representations.values())).squeeze(1)
    class_precision_matrices = torch.stack(list(class_precision_matrices.values()))

    # grabbing the number of classes and query examples for easier use later in the function
    number_of_classes = class_means.size(0)

    # return split_first_dim_linear(sample_logits, [NUM_SAMPLES, query_features.shape[0]])
    return class_representations, class_precision_matrices, number_of_classes, class_means

def MD_distance_test1_2 (number_of_classes, class_means, class_precision_matrices, query_features):
    """
    The same as MD_distance_test1, but devided into 2: MD_distance_test1_1 and MD_distance_test1_2
    """    
    number_of_targets = query_features.size(0)

    repeated_target = query_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
    repeated_class_means = class_means.repeat(number_of_targets, 1)
    repeated_difference = (repeated_class_means - repeated_target)
    repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                   repeated_difference.size(1)).permute(1, 0, 2)
    first_half = torch.matmul(repeated_difference, class_precision_matrices)
    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1
    #sample_logits is actually the distance with mean
    
    return sample_logits


def MD_distance_test2(query_features,class_representations, class_precision_matrices):
    # class_representations, class_precision_matrices = build_class_reps_and_covariance_estimates(support_feature, support_labels)
    #
    class_means = torch.stack(list(class_representations.values())).squeeze(1)
    # class_precision_matrices = torch.stack(list(class_precision_matrices.values()))
    #
    # # grabbing the number of classes and query examples for easier use later in the function
    number_of_classes = class_means.size(0)
    number_of_targets = query_features.size(0)

    repeated_target = query_features.repeat(1, number_of_classes).view(-1, query_features.size(1))
    repeated_class_means = class_means.repeat(number_of_targets, 1)
    repeated_difference = (repeated_class_means - repeated_target)
    repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                   repeated_difference.size(1)).permute(1, 0, 2)
    first_half = torch.matmul(repeated_difference, class_precision_matrices)
    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

    return sample_logits

def build_class_reps_and_covariance_estimates(context_features, context_labels):
    class_representations={}
    class_precision_matrices={}
    task_covariance_estimate = estimate_cov(context_features) #this is sigma
    for c in torch.unique(context_labels):
        # filter out feature vectors which have class c
        class_mask = torch.eq(context_labels, c)
        class_mask_indices = torch.nonzero(class_mask)
        class_features = torch.index_select(context_features, 0, torch.reshape(class_mask_indices, (-1,)).cuda())
        # mean pooling examples to form class means
        class_rep = mean_pooling(class_features)
        # updating the class representations dictionary with the mean pooled representation
        class_representations[c.item()] = class_rep
        """
        Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
        Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
        inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
        dictionary for use later in infering of the query data points.
        """
        lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
        class_precision_matrices[c.item()] = torch.inverse(
            (lambda_k_tau * estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
            + torch.eye(class_features.size(1), class_features.size(1)).cuda(0))
    return class_representations,class_precision_matrices

def estimate_cov(examples, rowvar=False, inplace=False):
    if examples.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if examples.dim() < 2:
        examples = examples.view(1, -1)
    if not rowvar and examples.size(0) != 1:
        examples = examples.t()
    factor = 1.0 / (examples.size(1) - 1)
    if inplace:
        examples -= torch.mean(examples, dim=1, keepdim=True)
    else:
        examples = examples - torch.mean(examples, dim=1, keepdim=True)
    examples_t = examples.t()
    return factor * examples.matmul(examples_t).squeeze()


def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)


def get_meta_train_data (target_da_labels, target_da_datas):
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set #devide the data per label
    print(target_da_metatrain_data.keys())
    
    return target_da_metatrain_data

class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Meta_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(Meta_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.labels[idx]
        return image, label