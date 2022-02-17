#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:29:36 2021

@author: shah

"""

import numpy as np
from skimage import io
from PIL import Image 
import  random
import os


def ExtractPath(pathoffile):
    listofimages=[]
    for img in os.listdir(pathoffile):
        pathofimage= os.path.join(pathoffile, img)
        listofimages.append(pathofimage)
        listofimages.sort()
    return listofimages


def ReshapeData(noisyImages, groundtruthImages):
    
    
   noisyImages, groundtruthImages= np.array(noisyImages), np.array(groundtruthImages)
   noisyImages = np.reshape(noisyImages, (noisyImages.shape[0],) + tuple(noisyImages[0].shape[0:3]))
   groundtruthImages = np.reshape(groundtruthImages, (groundtruthImages.shape[0],) +  (1,) + tuple(groundtruthImages[0].shape[0:2]))
   return noisyImages, groundtruthImages    
    

def LoadImages(listofstack):
    images=[]
    for img in listofstack:
        img_arr=np.array(io.imread(img))
        images.append(img_arr)
    return images

def ImageNormalization(imgs):
    newImgs = []
    min = np.amin(imgs)
    max = np.amax(imgs)
    while len(imgs) != 0:
        img = (imgs[0] - min) * (1/(max-min))
        newImgs.append(img)
        del imgs[0]
    return newImgs

def DataRandomization(noisyImages, groundtruthImages):
    data = list(zip(noisyImages, groundtruthImages))
    random.shuffle(data)
    noisyImages, groundtruthImages = zip(*data)
    return noisyImages, groundtruthImages


def AugmentationFunction(images, angle):
    AugmentedImages=[]
    if len(images)>0:
        for i in range(len(images)):
            pImage= Image.fromarray(images[i])
            rImage= pImage.rotate(angle)
            aImage=np.array(rImage)
            AugmentedImages.append(aImage)
        return AugmentedImages