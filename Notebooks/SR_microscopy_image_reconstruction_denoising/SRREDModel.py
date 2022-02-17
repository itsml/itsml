#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:18:21 2020

@author: shah
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shah
"""
from tensorflow import keras
#import tensorflow as tf
import math 



class UNet:
    def __init__(self, height, width, channel, kernelsize, padding, stride, activation, maxpoolingfactor):

        self.height=height
        self.width = width
        self.channel= channel
        self.kernelsize = kernelsize
        self.padding = padding
        self.stride = stride
        self.activation=activation
        self.maxpooling= maxpoolingfactor
        
    def Contraction(self, image, filters):
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, strides=self.stride, activation=self.activation)(image)
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, strides=self.stride, activation=self.activation)(c)
        p = keras.layers.MaxPool2D((self.maxpooling, self.maxpooling), (self.maxpooling, self.maxpooling))(c)
        return c, p

    def Expansion(self,feature, skip, filters):
        us = keras.layers.UpSampling2D((self.maxpooling, self.maxpooling))(feature)
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, strides=self.stride, activation=self.activation)(concat)
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, strides=self.stride, activation=self.activation)(c)
        return c

    def BottleNeck(self, feature, filters):
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, strides=self.stride, activation=self.activation)(feature)
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, strides=self.stride, activation=self.activation)(c)
        return c



    def buildUNet(self):
        f =  [16,  32,  64,  128,  256, 512]
        inputs =  keras.layers.Input((self.height , self.width, self.channel))
    
        Input = inputs
        c1, p1 = self.Contraction(Input, f[0]) 
        c2, p2 = self.Contraction(p1, f[1]) 
        c3, p3 = self.Contraction(p2, f[2]) 
        c4, p4 = self.Contraction(p3, f[3]) 
        c5, p5= self.Contraction(p4, f[4]) 
        
        bn = self.BottleNeck(p5, f[5])
          
        u1 = self.Expansion(bn, c5 ,f[4]) 
        u2 = self.Expansion(u1, c4, f[3]) 
        u3 = self.Expansion(u2, c3, f[2]) 
        u4 = self.Expansion(u3, c2, f[1]) 
        u5 = self.Expansion(u4, c1, f[0]) 

        
        outputs = keras.layers.Conv2D(1, (1, 1), padding=self.padding, activation="sigmoid")(u5)
        model = keras.models.Model(Input, outputs)
        return model
    
    
class SRREDNet:
    
    def __init__(self, height, width, channel,filters, kernelsize, padding, stride, activation):
        self.height=height
        self.width = width
        self.channel= channel
        self.filters= filters
        self.kernelsize = kernelsize
        self.padding = padding
        self.stride = stride
        self.activation=activation
        
    
    def PostUpSampling(self, x, filters):
        c = keras.layers.Conv2DTranspose(filters, kernel_size=self.kernelsize, padding=self.padding,data_format='channels_first', strides=2, activation=self.activation)(x)
        return c

    def FristBlockEncoding(self, x, filters):
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding,data_format='channels_first', strides=self.stride, activation=self.activation)(x)
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, data_format='channels_first', strides=self.stride, activation=self.activation)(c)
        return c

    def Encoding(self, x, filters):
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding,data_format='channels_first', strides=self.stride, activation=self.activation)(x)
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, data_format='channels_first', strides=self.stride, activation=self.activation)(c)
        return c

    def SEncoding(self, x, filters):
    
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding,data_format='channels_first', strides=self.stride, activation=self.activation)(x)
        return c
    

    def Decoding(self, x, skip, filters):
    
        c = keras.layers.Conv2DTranspose(filters, kernel_size=self.kernelsize, padding=self.padding,data_format='channels_first', strides=self.stride, activation=self.activation)(x)
        concat = keras.layers.Add()([c, skip])
        active= keras.layers.Activation(self.activation)(concat)
        c = keras.layers.Conv2DTranspose(filters, kernel_size=self.kernelsize, padding=self.padding,data_format='channels_first', strides=self.stride, activation=self.activation)(active)
        return c


    def SDecoding(self, x, skip, filters):
        
        c = keras.layers.Conv2DTranspose(filters, kernel_size=self.kernelsize, padding=self.padding,data_format='channels_first', strides=self.stride, activation=self.activation)(x)
        c = keras.layers.Conv2DTranspose(filters, kernel_size=self.kernelsize, padding=self.padding, data_format='channels_first', strides=self.stride, activation=self.activation)(c)
    
        return c
    
    def LDecoding(self,x, filters):
        c = keras.layers.Conv2DTranspose(filters, kernel_size=self.kernelsize, padding=self.padding, data_format='channels_first', strides=self.stride, activation=self.activation)(x)
        return c

    def buildDNN(self):
        f= [32, 1]
        inputs =  keras.layers.Input((self.channel, self.height , self.width))

#    
        pinput = inputs
    
        p1 = self.FristBlockEncoding(pinput, f[0])
        p2 = self.Encoding(p1, f[0])
        p3 = self.Encoding(p2, f[0])
        p4 = self.Encoding(p3, f[0])
        p5 = self.Encoding(p4, f[0])
        p6 = self.Encoding(p5, f[0])
        p7 = self.Encoding(p6, f[0])
        p8 = self.Encoding(p7, f[0])
        p9 = self.Encoding(p8, f[0])
        p10 = self.Encoding(p9, f[0])
        p11 = self.SEncoding(p10, f[0])
    
        u1 = self.Decoding(p11, p10 ,f[0])
    
        u2 = self.Decoding(u1, p9 ,f[0])
        u3 = self.Decoding(u2, p8 ,f[0])
        u4 = self.Decoding(u3, p7 ,f[0])
        u5 = self.Decoding(u4, p6 ,f[0])
        u6 = self.Decoding(u5, p5 ,f[0])
        u7=  self.Decoding(u6,p4, f[0])
        u8 = self.Decoding(u7, p6 ,f[0])
        u9 = self.Decoding(u8, p5 ,f[0])
        u10=  self.Decoding(u9,p4, f[0])
        u11= self.LDecoding(u10, f[0])
        u12=self.PostUpSampling(u11, f[0])
        u13=self.LDecoding(u12, f[1])

    
        model = keras.models.Model(inputs, u13)
        model.summary()
        return model

    
    


