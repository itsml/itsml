#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:24:19 2020

@author: shah
"""
import tf_psnr
from tensorflow import keras
import pickle, os, numpy 
from PIL import Image
import tensorflow as tf


def numpyToPIL(numpyImage):
    return Image.fromarray(numpy.uint8(numpyImage*255))


# Wrapper function for the tensorflow psnr function. Needed for own metrics in compiling the keras model.
def psnr(y_true, y_pred):
    return tf_psnr.psnr(y_true, y_pred, 1.0)


def compileDNN(dnn, LOSS_TYPE):
    dnn.compile(optimizer=keras.optimizers.Adam()
                , loss=LOSS_TYPE
                , metrics=[LOSS_TYPE])  # ToDo Make psnr function from tf available

class SaveHistory(keras.callbacks.Callback):
    def __init__(self, outFilepath):
        self.outFilepath = outFilepath
        self.history = {}

    def on_train_begin(self, outFilepath, logs={}):
        # Dictionary of lists with metric values in order of the epochs. Keys are
        self.history = {}
        # Create output file if not already existing
        if not os.path.isfile(self.outFilepath):
            f = open(self.outFilepath, 'w')
            f.close()

    def on_epoch_end(self, epoch, logs={}):
        for key in logs:
            if key in self.history:
                self.history[key].append(logs[key])
            else:
                self.history[key] = [logs[key],]
        # Pickle losses
        with open(self.outFilepath, 'wb') as f:
            pickle.dump(self.history, f)
            
            
def safelyCreateNewDir(dirPath):
    potentialDirPath = dirPath
    dirSuffix = 1
    while True:
        try:
            os.makedirs(potentialDirPath, exist_ok=False)
            return potentialDirPath
        except OSError:
            pass
        except: # Catch any other unexpected error and re-raise it
            raise
        potentialDirPath = dirPath+'_'+str(dirSuffix)
        dirSuffix += 1
        
