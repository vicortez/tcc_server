# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:25:01 2018

@author: victor
"""
import Augmentor
import glob
import numpy as np
import cv2

def augment_dataset():
    pcar = Augmentor.Pipeline("P:\\0-My documents\\datasets\\tcc_imd_custom_dataset\\car")
    pmot = Augmentor.Pipeline("P:\\0-My documents\\datasets\\tcc_imd_custom_dataset\\motorcycle")
    pcar.random_distortion(probability=0.15, grid_width=4, grid_height=4, magnitude=6)
    pmot.random_distortion(probability=0.15, grid_width=4, grid_height=4, magnitude=6)
    pcar.flip_left_right(probability=0.75)
    pmot.flip_left_right(probability=0.75)
    pcar.crop_random(probability=0.15, percentage_area=0.85)
    pmot.crop_random(probability=0.15, percentage_area=0.85)
    pcar.rotate(probability=0.8, max_left_rotation=15, max_right_rotation=15)
    pmot.rotate(probability=0.8, max_left_rotation=15, max_right_rotation=15)
    pcar.sample(300)
    pmot.sample(300)
    
def load_car_imgs():
    return images_car
def load_mot_imgs():
    return images_mot
filenames_mot = glob.glob("P:\\0-My documents\\datasets\\tcc_imd_custom_dataset\\motorcycle\\output\\*.jpg")
filenames_car = glob.glob("P:\\0-My documents\\datasets\\tcc_imd_custom_dataset\\car\\output\\*.jpg")
#filenames.sort()
images_car = [cv2.imread(img) for img in filenames_car]
images_mot = [cv2.imread(img) for img in filenames_mot]

images_car = [cv2.resize(img, (32, 32)).astype(np.float32) for img in images_car]
images_mot = [cv2.resize(img, (32, 32)).astype(np.float32) for img in images_mot]

images_car = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_car]
images_mot = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_mot]

    