# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:20:56 2020
This file is to save the information about HSI
@author: 22594658
"""

from scipy import io, misc

import numpy as np
import os
import spectral
from sklearn import preprocessing


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
    elif ext == '.npy':
        return np.load(dataset, allow_pickle=True)
    else:
        raise ValueError("Unknown file format: {}".format(ext))


class HSI():
    """ Generic class for a hyperspectral scene """

    def __init__(self, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
        """
        super(HSI, self).__init__()
        self.name = hyperparams['dataset']
        self.norm_type = hyperparams['norm_type']

        self.folder = str(hyperparams['folder']) + str(self.name) + '/'
        print(self.folder)
        self.ignored_labels = []

        img = self.HSI_open()
        # self.patch_size = hyperparams['patch_size']

        if self.name == 'IndianPines':

            # load the image
            # img=open_file (self.folder+'Indian_pines_corrected.mat')
            # img=img['indian_pines_corrected']

            self.rgb_bands = (43, 21, 11)  # because AVIRIS SENSOR
            self.img_channels = 200

            self.gt = open_file(self.folder + 'Indian_pines_gt.mat')['indian_pines_gt']
            self.label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                                 "Corn", "Grass-pasture", "Grass-trees",
                                 "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                                 "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                                 "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                                 "Stone-Steel-Towers"]
            self.category = 16

            self.ignored_labels = [0]
            #self.traing_standard = (0, 15, 50, 50, 50, 50, 50, 15, 50, 15, 50, 50, 50, 50, 50, 50, 50)
        if self.name == 'IndianPines100':

            # load the image
            #img=open_file (self.folder+'IndianPines100.mat')
            #img=img['IndianPines100']

            self.rgb_bands = (43, 21, 11)  # because AVIRIS SENSOR
            self.img_channels = 200

            self.gt = open_file(self.folder + 'IndianPines100_gt.mat')['indian_pines_gt']
            self.label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                                 "Corn", "Grass-pasture", "Grass-trees",
                                 "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                                 "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                                 "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                                 "Stone-Steel-Towers"]
            self.category = 16

            self.ignored_labels = [0]
            #self.traing_standard = (0, 15, 50, 50, 50, 50, 50, 15, 50, 15, 50, 50, 50, 50, 50, 50, 50)


        elif self.name == 'PaviaC':
            # Load the image
            # img = open_file(self.folder + 'Pavia.mat')['pavia']

            self.rgb_bands = (55, 41, 12)
            self.img_channels = img.shape[2]

            self.gt = open_file(self.folder + 'Pavia_gt.mat')['pavia_gt']

            self.label_values = ["Undefined", "Water", "Trees", "Asphalt",
                                 "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                                 "Meadows", "Bare Soil"]
            self.category = 9
            #self.training_standard = (0, 50, 50, 50, 50, 50, 50, 50, 50, 50)

            self.ignored_labels = [0]


        elif self.name == 'KSC':
            # Load the image
            # img = open_file(self.folder + 'KSC.mat')['KSC']

            self.rgb_bands = (43, 21, 11)  # AVIRIS sensor
            self.img_channels = img.shape[2]

            self.gt = open_file(self.folder + 'KSC_gt.mat')['KSC_gt']

            self.label_values = ["Undefined", "Scrub", "Willow swamp",
                                 "Cabbage palm hammock", "Cabbage palm/oak hammock",
                                 "Slash pine", "Oak/broadleaf hammock",
                                 "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                                 "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]
            self.category = 13
            #self.traing_standard = (0, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 15)
            self.ignored_labels = [0]

        elif self.name == 'Botswana':
            # Load the image
            # img = open_file(self.folder + 'Botswana.mat')['Botswana']

            self.rgb_bands = (75, 33, 15)
            self.img_channels = img.shape[2]

            self.gt = open_file(self.folder + 'Botswana_gt.mat')['Botswana_gt']

            self.label_values = ["Undefined", "Water", "Hippo grass",
                                 "Floodplain grasses 1", "Floodplain grasses 2",
                                 "Reeds", "Riparian", "Firescar", "Island interior",
                                 "Acacia woodlands", "Acacia shrublands",
                                 "Acacia grasslands", "Short mopane", "Mixed mopane",
                                 "Exposed soils"]
            self.category = 14
            #self.traing_standard = (0, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 15)

            self.ignored_labels = [0]

        elif self.name == 'Salinas' or self.name == 'Salinas100':
            # Load the image
            # img = open_file(self.folder + 'Salinas_corrected.mat')['salinas_corrected']

            self.rgb_bands = (75, 33, 15)  # don't know this value
            self.img_channels = img.shape[2]

            self.gt = open_file(self.folder + 'Salinas_gt.mat')['salinas_gt']

            self.label_values = ["Undefined", "Brocoli_green_weeds_1", "Brocoli_green_weeds_2",
                                 "Fallow", "Fallow_rough_plow",
                                 "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained",
                                 "Soil_vinyard_develop", "Corn_senesced_green_weeds",
                                 "Lettuce_romaine_4wk", "Lettuce_romaine_5wk", "Lettuce_romaine_6wk",
                                 "Lettuce_romaine_7wk", "Vinyard_untrained", "Vinyard_vertical_trellis"]
            self.category = 16
            #self.traing_standard = (0, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 15, 15, 15)

            self.ignored_labels = [0]

        elif self.name == 'PaviaU':
            # Load the image
            # img = open_file(self.folder + 'PaviaU.mat')['paviaU']

            self.rgb_bands = (75, 33, 15)  # don't know this value
            self.img_channels = img.shape[2]

            self.gt = open_file(self.folder + 'PaviaU_gt.mat')['paviaU_gt']

            self.label_values = ["Undefined", "Asphalt", "Meadows",
                                 "Gravel", "Trees",
                                 "Painted metal sheets ", "Bare Soil ", "Bitumen", "Self-Blocking Bricks ",
                                 "Shadows"]
            self.category = 9

            self.ignored_labels = [0]

            #self.traing_standard = (0, 548, 540, 392, 524, 265, 532, 375, 514, 231)

        elif self.name == 'Xiu':
            # Load the image
            img = open_file(self.folder + 'data.npy')
            img = img.astype(np.float64)
            self.rgb_bands = (55, 41, 12)
            self.img_channels = img.shape[1]

            self.gt = open_file(self.folder + 'labels.npy')
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = ["normal", "desease"]
            self.category = 2
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0]

        elif self.name == 'XiuBackground':
            # Load the image
            # Load the Xiu dataset with the background information
            self.folder = str(hyperparams['folder']) + 'Xiu' + '/'
            img = open_file(self.folder + 'wheat.npy')
            img = img.astype(np.float64)
            self.rgb_bands = (55, 41, 12)
            self.img_channels = img.shape[1]

            self.gt = open_file(self.folder + 'label.npy')
            self.gt = self.gt.reshape(self.gt.shape[0])
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = ["background", "normal", "desease"]
            self.category = 3
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0]

        elif self.name == 'co':
            # Load the data

            self.folder = str(hyperparams['folder']) + 'salt_moghimi' + '/'
            data = open_file(self.folder + 'co(CS).mat')
            data = data["data"]
            img = data[:, 0:-1]
            label = data[:, -1]
            self.rgb_bands = (55, 41, 12)  # this value is random
            self.img_channels = img.shape[1]

            self.gt = label
            self.gt = self.gt.reshape(self.gt.shape[0])
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = ["salt", "control"]
            self.category = 2
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0]

        elif self.name == 'Kharchia':
            # Load the data

            self.folder = str(hyperparams['folder']) + 'salt_moghimi' + '/'
            data = open_file(self.folder + 'Kharchia.mat')
            data = data["data"]
            img = data[:, 0:-1]
            label = data[:, -1]
            self.rgb_bands = (55, 41, 12)  # this value is random
            self.img_channels = img.shape[1]

            self.gt = label
            self.gt = self.gt.reshape(self.gt.shape[0])
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = ["salt", "control"]
            self.category = 2
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0]

        elif self.name == 'sp':
            # Load the data

            self.folder = str(hyperparams['folder']) + 'salt_moghimi' + '/'
            data = open_file(self.folder + 'sp(CS).mat')
            data = data["data"]
            img = data[:, 0:-1]
            label = data[:, -1]
            self.rgb_bands = (55, 41, 12)  # this value is random
            self.img_channels = img.shape[1]

            self.gt = label
            self.gt = self.gt.reshape(self.gt.shape[0])
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = ["salt", "control"]
            self.category = 2
            self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0]

        elif self.name == 'cs':
            # Load the data

            self.folder = str(hyperparams['folder']) + 'salt_moghimi' + '/'
            data = open_file(self.folder + 'CS.mat')
            data = data["data"]
            img = data[:, 0:-1]
            label = data[:, -1]
            self.rgb_bands = (55, 41, 12)  # this value is random
            self.img_channels = img.shape[1]

            self.gt = label
            self.gt = self.gt.reshape(self.gt.shape[0])
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = ["salt", "control"]
            self.category = 2
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0]

        elif self.name == 'Barley':
            self.folder = str(hyperparams['folder']) + 'cropvariety' + '/'
            all_data = open_file (self.folder +'Barley.npy')
            data = all_data[0]
            img = data[:, 0:-1]
            label = data [:, -1]
            self.rgb_bands = (55, 41, 12)  # this value is random
            self.img_channels = img.shape[1]
            self.gt = label
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = all_data[1]
            self.category = len(self.label_values)
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0]             

        elif self.name == 'Grapevine':
            self.folder = str(hyperparams['folder']) + 'Grapevine' + '/'
            all_data = open_file (self.folder +'Grapevine.npy')
            data = all_data[0]
            img = data[:, 0:-1]
            label = data [:, -1]
            self.rgb_bands = (55, 41, 12)  # this value is random
            self.img_channels = img.shape[1]
            self.gt = label
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = all_data[1]
            self.category = len(self.label_values)
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0]             
        elif self.name == 'Oak':
            self.folder = str(hyperparams['folder']) + 'OakData' + '/'
            all_data = open_file (self.folder +'Oak.npy')
            data = all_data[0]
            img = data[:, 0:-1]
            label = data [:, -1]
            self.rgb_bands = (55, 41, 12)  # this value is random
            self.img_channels = img.shape[1]
            self.gt = label
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = all_data[1]
            self.category = len(self.label_values)
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0] 

        elif self.name == 'Oak1':
            #the data is from 2017 dataset
            self.folder = str(hyperparams['folder']) + 'OakData' + '/'
            all_data = open_file (self.folder +'Oak1.npy')
            data = all_data[0]
            img = data[:, 0:-1]
            label = data [:, -1]
            self.rgb_bands = (55, 41, 12)  # this value is random
            self.img_channels = img.shape[1]
            self.gt = label
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = all_data[1]
            self.category = len(self.label_values)
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0] 

        elif self.name == 'Oak2':
            #the data is from 2018 dataset
            self.folder = str(hyperparams['folder']) + 'OakData' + '/'
            all_data = open_file (self.folder +'Oak2.npy')
            data = all_data[0]
            img = data[:, 0:-1]
            label = data [:, -1]
            self.rgb_bands = (55, 41, 12)  # this value is random
            self.img_channels = img.shape[1]
            self.gt = label
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = all_data[1]
            self.category = len(self.label_values)
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0] 

        elif self.name == 'Oak3':
            #the data is from 2018 dataset with symptomatic
            self.folder = str(hyperparams['folder']) + 'OakData' + '/'
            all_data = open_file (self.folder +'Oak3.npy')
            data = all_data[0]
            img = data[:, 0:-1]
            label = data [:, -1]
            self.rgb_bands = (55, 41, 12)  # this value is random
            self.img_channels = img.shape[1]
            self.gt = label
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = all_data[1]
            self.category = len(self.label_values)
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0] 

        elif self.name == 'Cassava2':
            #the data is from 2018 dataset with symptomatic
            self.folder = str(hyperparams['folder']) + 'cassava_dataset2' + '/'
            all_data = open_file (self.folder +'Cassava2.npy')
            data =all_data[0]
            img = data[:, 0:-1]
            label = data [:, -1]
            self.rgb_bands = (55, 41, 12)  # this value is random
            self.img_channels = img.shape[1]
            self.gt = label
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values =  ["Healthy","CBSD", "CMD"]
            self.category = len(self.label_values)
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0] 

        elif self.name == 'CassavaScreenH':
            #the data is from 2018 dataset with symptomatic
            self.folder = str(hyperparams['folder']) + 'Cassava1' + '/'
            all_data = open_file (self.folder +'Cassava_ScreenHouse.npy')
            data =all_data[0]
            img = data[:, 0:-1]
            label = data [:, -1]
            self.rgb_bands = (55, 41, 12)  # this value is random
            self.img_channels = img.shape[1]
            self.gt = label
            self.gt = self.gt 
            self.gt = self.gt.astype(np.uint8)

            self.label_values =  ["Healthy","CBSD", "CMD"]
            self.category = len(self.label_values)
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0] 

        elif self.name == 'CassavaField':
            #the data is from 2018 dataset with symptomatic
            self.folder = str(hyperparams['folder']) + 'Cassava1' + '/'
            all_data = open_file (self.folder +'Cassava_Field.npy')
            data =all_data[0]
            img = data[:, 0:-1]
            label = data [:, -1]
            self.rgb_bands = (55, 41, 12)  # this value is random
            self.img_channels = img.shape[1]
            self.gt = label
            self.gt = self.gt
            self.gt = self.gt.astype(np.uint8)

            self.label_values =  ["Healthy","CBSD", "CMD"]
            self.category = len(self.label_values)
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0] 

        elif self.name == 'Frost':
            # Load the data

            print("The folder: ", self.folder)
            data1 = open_file(self.folder + 'TOS3.npy')
            data2 = open_file(self.folder + 'TOS6.npy')
            data3 = open_file(self.folder + 'TOS4.npy')
            data4 = open_file(self.folder + 'TOS5.npy')
            data = np.concatenate((data1, data2, data3, data4), axis=0)
            # data=np.concatenate(( data2, data4), axis=0)
            # data=np.concatenate(( data2, data3), axis=0)
            # data=np.concatenate(( data1, data2), axis=0)

            l1 = len(data1)
            l2 = len(data2)
            l3 = len(data3)
            l4 = len(data4)

            label1 = np.zeros((l1, 1))
            label2 = np.ones((l2, 1))
            label3 = np.zeros((l3, 1))
            label4 = np.zeros((l4, 1))
            label = np.concatenate((label1, label2, label3, label4), axis=0)
            # label=np.concatenate((label2,label4),axis=0)
            # label=np.concatenate((label2,label3),axis=0)
            # label=np.concatenate((label1,label2),axis=0)
            img = data

            self.rgb_bands = (55, 41, 12)  # this value is random
            self.img_channels = img.shape[1]

            self.gt = label
            self.gt = self.gt.reshape(self.gt.shape[0])
            self.gt = self.gt + 1
            self.gt = self.gt.astype(np.uint8)

            self.label_values = ["salt", "control"]
            self.category = 2
            #self.traing_standard = (0, 50, 50)

            self.ignored_labels = [0]

        elif self.name == 'alldata':
            # Load the data
            # Load the image
            # Load the Xiu dataset with the background information
            self.folder = str(hyperparams['folder']) + '/'
            img = open_file(self.folder + 'alldata.mat')["data"]
            gt = open_file(self.folder + 'alldata.mat')["label"]
            img = img.astype(np.float64)

            self.gt = gt
            print ("shape img: ", img.shape)
            print ("shape gt: ", self.gt.shape)
            #self.gt = self.gt.reshape(self.gt.shape[1])

            self.gt = self.gt.astype(np.uint8)
            self.category = 77
            #self.traing_standard = (0, 50, 50)


        # filter NaN out, if there is any none
        nan_mask = np.isnan(img.sum(axis=-1))
        if np.count_nonzero(nan_mask) > 0:
            print("Warning: NaN has been found in this dataset and the Nan mask will be disabled")

        img[nan_mask] = 0
        self.gt[nan_mask] = 0
        self.ignored_labels.append(0)

        self.ignored_labels = list(set(self.ignored_labels))
        # normalization
        img = np.asarray(img, dtype='float32')
        self.img = img

    def HSI_open(self):
        if self.name == 'IndianPines':

            # load the image
            img = open_file(self.folder + 'Indian_pines_corrected.mat')
            img = img['indian_pines_corrected']
            return img
        if self.name == 'IndianPines100':

            # load the image
            img=open_file (self.folder+'IndianPines100.mat')
            img=img['IndianPines100']
            return img
        elif self.name == 'PaviaU':
            # Load the image
            img = open_file(self.folder + 'PaviaU.mat')['paviaU']
            return img
        elif self.name == 'KSC':
            # Load the image
            img = open_file(self.folder + 'KSC.mat')['KSC']
            return img
        elif self.name == 'Botswana':
            # Load the image
            img = open_file(self.folder + 'Botswana.mat')['Botswana']
            return img
        elif self.name == 'Salinas':
            # Load the image
            img = open_file(self.folder + 'Salinas_corrected.mat')['salinas_corrected']
            return img
        elif self.name == 'Salinas100':
            # Load the image
            img = open_file(self.folder + 'Salinas100.mat')['output']
            return img
        elif self.name == 'PaviaC':
            # Load the image
            img = open_file(self.folder + 'Pavia.mat')['pavia']
            return img

    def Normalize(self, norm_type):
        """
        This method is used to compute the normalization based on the normalization type
        """

        if norm_type is None:
            norm_type = 'normal'

        if norm_type == 'scale':
            # this process is the same with standarization
            # process the standarization
            # this process make the data to have mean=0 and std=1
            print("norm_type is scale")
            data = self.img.reshape(np.prod(self.img.shape[:2]),
                                    np.prod(self.img.shape[2:]))  # reshape data from 145*145*200 into 21025*200
            data = preprocessing.scale(data)
            self.img = data.reshape(self.img.shape[0], self.img.shape[1], self.img.shape[2])

        elif norm_type == 'normal':
            # process with zero one normalization
            self.img = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))

        else:
            print("The normalization type you choose is not available")

        return self.img

    # def __len__(self):
    #    return len(self.indices)

    # def __getItem__(self,i): #hold to make this function, in HyperX it is used for data augmentation
    #    x,y:self.indices[i]

