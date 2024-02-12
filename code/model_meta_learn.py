#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:23:41 2022

This file contain models for meta-learning process

@author: research
"""
import torch

class MAE_Embedding2(torch.nn.Module):
    """
    Different to MAE_Embedding, this class only return the class token as features
    """
    
    def __init__(self,model):
        super(MAE_Embedding2,self).__init__()

        self.model = model     
   
    
    def forward (self, x):
        x, index = self.model (x)
        x = x[0]


        return x
