# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:23:20 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

change_celebA_data
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import os
import zipfile
from PIL import Image
import argparse
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_name', type = str, default = 'celebA')
    parser.add_argument('--crop_size', type = int, default = 128)
    parser.add_argument('--final_size', type = int, default = 64)
    
    args = parser.parse_args()
    
    config = OrderedDict([
            ('save_name', args.save_name),
            ('crop_size', args.crop_size),
            ('final_size', args.final_size)])
    
    return config
    
config = parse_args()

folder_dir = os.path.join(os.getcwd(),'celeba-dataset')
img_zip = zipfile.ZipFile(folder_dir+"/img_align_celeba.zip", 'r')
files = np.array([f for f in sorted(img_zip.NameToInfo.keys())])[1:]

def transformed_image(img_name, show=False):
    img = img_zip.open(img_name)
    img = np.array(Image.open(img), dtype=np.uint8)#[center[0]-74:center[0]+54,center[1]-64:center[1]+64,:]
    img = tf.image.crop_to_bounding_box(img,(218 - config['crop_size']) // 2,
                                          (178 - config['crop_size']) // 2, config['crop_size'], config['crop_size'])
    img = np.array(tf.image.resize(img, size=(config['final_size'], config['final_size'])), dtype=np.uint8)
    return img.flatten()

def show_transformed_image(img_name):
    img = img_zip.open(img_name)
    img_g = np.array(Image.open(img), dtype=np.uint8)#[center[0]-74:center[0]+54,center[1]-64:center[1]+64,:]
    img_c = tf.image.crop_to_bounding_box(img_g,(218 - config['crop_size']) // 2,
                                          (178 - config['crop_size']) // 2, config['crop_size'], config['crop_size'])
    img_s = np.array(tf.image.resize(img_c, size=(config['final_size'], config['final_size'])), dtype=np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(7,7))
    axs[0].imshow(img_g)
    axs[1].imshow(img_c)
    axs[2].imshow(img_s)


filename = config['save_name']+'{}.npz'.format(config['final_size'])
with zipfile.ZipFile(filename, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
    for i in range(len(files)):
        image = transformed_image(files[i])
        tmpfilename = "img_cel_{}.npy".format(i)
        np.save(tmpfilename, image)
        zf.write(tmpfilename)
        os.remove(tmpfilename) 