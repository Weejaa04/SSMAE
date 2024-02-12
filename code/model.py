#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 15:36:26 2022

This file is about the model

@author: research
"""
import torch 
import numpy as np

from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

def random_indexes (size: int):
    forward_indexes = np.arange (size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    #print ("Sequences shape: ", sequences.shape)
    #print ("Indexes shape: ", indexes.shape)
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self,ratio):
        super().__init__()
        self.ratio = ratio
    
    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape #B=batch size, C=channel, T= number of sequence
        
        #print ("patches shape: ", patches.shape)
        remain_T = int (T*(1-self.ratio))
        
        indexes = [random_indexes(T) for _ in range(B)] #doing random_indexes for each sample
        #print ("indexes: ", indexes)
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        #print ("in patch shuffle")
        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes #the remain patches, the index for forward and back ward
    

class MAE_SS_Encoder(torch.nn.Module):
    def __init__(self,
                 patch_size = 3, 
                 mask_patch_size = 2, 
                 mask_spectral=5,
                 emb_dim = 96, 
                 num_layer = 6, 
                 num_head= 3, 
                 mask_ratio = 0.75, 
                 num_band=100):
        super().__init__()
        
        self.cls_token = torch.nn.Parameter(torch.zeros(1,1,emb_dim))
        #note: later the position embedding is different for 3D because we have to used mask in depth also, read 738
        self.pos_embedding = torch.nn.Parameter(torch.zeros((patch_size//mask_patch_size)**2*(num_band//mask_spectral),1, emb_dim))
        self.shuffle = PatchShuffle (mask_ratio)
        #self.patchify = torch.nn.Conv2d(num_band, emb_dim, mask_patch_size, mask_patch_size) #this is to change each patch into tokens
        self.patchify = torch.nn.Conv3d(1, emb_dim, (mask_patch_size, mask_patch_size,mask_spectral),(mask_patch_size, mask_patch_size,mask_spectral)) #this is to change each patch into tokens

        self.transformer = torch.nn.Sequential(*[Block(emb_dim,num_head) for _ in range(num_layer) ])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.init_weight()
        
    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)
        
    def forward(self, x):
        B, W,H,D = x.shape
        x=rearrange(x,'B W H D ->B 1 W H D')
        
        #print ("input x shape in encoder: ", x.shape)
        patches = self.patchify(x)
        #print ("patches after patchify (convolution) in encoder: ", patches.shape)        
        patches = rearrange(patches, 'b c h w d ->(h w d) b c')
        #print ("patches after patchify in encoder: ", patches.shape)
        #print ("Position embedding size: ", patches.shape)
        patches = patches+self.pos_embedding
        
        #print ("patches after positional embedding in encoder: ", patches.shape)           
        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        #after this, the patches only the visible one
        #print ("Patches after shuffle", patches.shape)
        #print ("Forward indexes: ", forward_indexes)
        #print ("backward indexes: ", backward_indexes)
        
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        #print ("patches before transformer: ", patches.shape)
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        #print ("features output of the concoder: ", features.shape)        
        #backward_indexes is used to return the correct order of the mask
        
        return features, backward_indexes
        
        
class MAE_SS_Decoder(torch.nn.Module):
    def __init__(self,
                 patch_size = 3, 
                 mask_patch_size = 2,
                 mask_spectral=5,
                 emb_dim = 96, 
                 num_layer=3, 
                 num_head = 3, 
                 num_band=100):
        super().__init__()
        
        self.mask_token = torch.nn.Parameter(torch.zeros(1,1,emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((patch_size//mask_patch_size)**2*(num_band//mask_spectral)+1,1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, mask_spectral * mask_patch_size ** 2)
        #self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=mask_patch_size, p2=mask_patch_size, h=patch_size//mask_patch_size)
        self.patch2img = Rearrange('(h w d) b (p1 p2 p3) -> b (h p1) (w p2) (d p3)', p1=mask_patch_size, p2=mask_patch_size, p3=mask_spectral, h=patch_size//mask_patch_size, d = num_band//mask_spectral)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        #print ("backward_indexes: ", backward_indexes.shape)
        
        features = torch.cat ([features, self.mask_token.expand(backward_indexes.shape[0]-features.shape[0], features.shape[1],-1)], dim =0)
        #print ("features shape after concat with backword indexes: ", features.shape)
        
        #print ("take_indexes in decoder")
        features = take_indexes (features, backward_indexes)
        #print ("feature after take index", features.shape)
        
    
        features = features+self.pos_embedding
        
        
        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange (features, 'b t c -> t b c')
        features  = features[1:] #remove global features
        #print ("Feature size before head: ", features.shape)
        patches = self.head(features)
        #print ("patch_size after head: ", patches.shape)
        mask = torch.zeros_like(patches)
        mask[T:] =1
        
        #print ("task indexes for mask")
        mask = take_indexes (mask, backward_indexes[1:]-1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)
        
        return img, mask
        
class MAE_VIT_SS(torch.nn.Module):
    def __init__(self, 
                 patch_size =3, #patch size of each sample
                 mask_patch_size = 2,#patch size of the mask
                 mask_spectral = 5,
                 emb_dim = 96, 
                 encoder_layer = 6, 
                 encoder_head = 3, 
                 decoder_layer = 3,
                 decoder_head = 3, 
                 mask_ratio = 0.75, 
                 num_band=100
                 ) -> None:
        super().__init__()
        
        self.encoder = MAE_SS_Encoder(patch_size=patch_size, mask_patch_size=mask_patch_size, mask_spectral=mask_spectral, emb_dim=emb_dim, num_layer=encoder_layer, num_head=encoder_head, mask_ratio=mask_ratio, num_band=num_band)
        self.decoder = MAE_SS_Decoder(patch_size=patch_size, mask_patch_size=mask_patch_size, mask_spectral=mask_spectral, emb_dim=emb_dim, num_layer=decoder_layer, num_head=decoder_head, num_band=num_band)
        
    def forward(self,x):
        features, backward_indexes = self.encoder(x)
        

        predicted_data, mask = self.decoder (features, backward_indexes)
        return predicted_data, mask
        




if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)
    encoder = MAE_SS_Encoder()
    decoder = MAE_SS_Decoder()
    features, backward_indexes = encoder(img)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)
