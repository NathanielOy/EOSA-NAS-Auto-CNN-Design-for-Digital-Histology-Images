# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 17:22:32 2020

@author: Oyelade
"""

from __future__ import division
import stain_utils as utils
import stainNorm_Reinhard
import stainNorm_Macenko
import stainNorm_Vahadane
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize
import pandas as pd 
from csv import reader

base_dir='/content/gdrive/My Drive/Paper13/' #'../../' #
base_dir_breakhis=base_dir+'Data/BreakHis/'
base_dir_beach=base_dir+'Data/Bach/'
base_dir_dest=base_dir+'Data/Preprocessed/'
height = 224
width = 224
    
def loadImages_bach(path):
    '''Put files into lists and return them as one list with all images 
     in the folder'''
    imgformat = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
    image_files = sorted([os.path.join(path, file)
                          for file in os.listdir(path)
                          if file.endswith(tuple(imgformat))])
    return image_files

def loadImages_breakhis(path):
    '''Put files into lists and return them as one list with all images 
     in the folder'''
    folders=['fold2/train', #'fold1/train', 'fold3/train', 'fold4/train', 'fold5/train',
             'fold2/test' #,'fold1/test', 'fold3/test', 'fold4/test', 'fold5/test'
            ]
    for fldr in folders:
        mdir=path+fldr+'/40X/'
        imgformat = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
        image_files = sorted([os.path.join(mdir, file)
                              for file in os.listdir(mdir)
                              if file.endswith(tuple(imgformat))])
    return image_files

# Display two images
def display(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()


# Display two images
def display2(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), cv2.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), cv2.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()
# Display one image
def display_one(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.show()

def denoise_image(imgs): # remove noise from all images
    no_noise = []
    for img in imgs:
        img, label=img
        blur = cv2.GaussianBlur(img, (5, 5), 0)#.astype('uint8') # Remove noise using Gaussian Blur
        #kernel = np.ones((3, 3), np.uint8)# Further noise removal (Morphology)
        #opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel, iterations=2)
        blur=blur, label
        no_noise.append(blur)
    return no_noise
        
def resize_image(img_data): # Resizing images
    dim = (width, height)    
    res_img = []
    for img in img_data:
        img, label=img
        res = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        res=res, label
        res_img.append(res)        
    return res_img

def reinhard_normalizer(img, target):
    n=stainNorm_Reinhard.Normalizer()
    n.fit(img)
    out = n.transform(img)
    #out = n.hematoxylin(img)
    #normalized=utils.build_stack((img))
    #utils.patch_grid(normalized,width=3,save_name='Reinhard.pdf')
    return out

def macenko_normalizer():
    n=stainNorm_Macenko.Normalizer()
    n.fit(i1)
    normalized=utils.build_stack((i1,n.transform(i2),n.transform(i3),n.transform(i4),n.transform(i5),n.transform(i6)))
    utils.patch_grid(normalized,width=3,save_name='./Macenko.pdf')
    return normalized

def vahadane_normalizer():
    hemo=utils.build_stack((n.hematoxylin(i1),n.hematoxylin(i2),n.hematoxylin(i3),n.hematoxylin(i4),n.hematoxylin(i5),n.hematoxylin(i6)))
    utils.patch_grid(hemo,width=3,save_name='./Macenko_hemo.pdf')
    
    n=stainNorm_Vahadane.Normalizer()
    n.fit(i1)
    normalized=utils.build_stack((i1,n.transform(i2),n.transform(i3),n.transform(i4),n.transform(i5),n.transform(i6)))
    utils.patch_grid(normalized,width=3,save_name='./Vahadane.pdf')
    return normalized

def vahadane_normalizer2():  
    hemo=utils.build_stack((n.hematoxylin(i1),n.hematoxylin(i2),n.hematoxylin(i3),n.hematoxylin(i4),n.hematoxylin(i5),n.hematoxylin(i6)))
    utils.patch_grid(hemo,width=3,save_name='./Vahadane_hemo.pdf')
    
    n = stainNorm_Vahadane.Normalizer()
    n.fit(i1)
    utils.show_colors(n.target_stains())
    plt.savefig('./stains.pdf')
    return normalized

def standardize_brightness(img):
    p = np.percentile(img, 90)
    return np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)

def normalize_images(imgs, target):
    norm_img=[]
    for img in imgs:
        img, label=img
        #img=standardize_brightness(img)
        img=reinhard_normalizer(img, target)
        img=img, label
        norm_img.append(img)     
    return norm_img

def write_2_file(imgs, base_dir_dest):
    i=0
    for img in imgs:
        img, label=img
        path=base_dir_dest+label+'_'+str(i)+'.png'
        cv2.imwrite(path, img)
        i=i+1
        
def get_breakhis_labels(imgs):
    img_labels=[]
    for img in imgs:
        label=img.split('_')[-1]
        label=label.split('-')[0]
        img_labels.append((img, label))                      
    return img_labels

def get_bach_labels(imgs):
    img_labels=[]
    for img in imgs:
        label=img.split('/')[-1]
        label=label[:2]
        for s in label:
            if s.isdigit():
                label=label[0]                
        img_labels.append((img, label))                      
    return img_labels

def processing(data, datype, base_dir_dest):  
    # Reading all images to work
    if datype=='bach':
        target=base_dir_beach+'n100.jpg'
        data=get_bach_labels(data)
    else:
        target=base_dir_dest+'fold2/train/40X/SOB_M_PC-15-190EF-40-019.png'
        data=get_breakhis_labels(data)
    
    target=base_dir_dest+'processed/IS.png'
    target = cv2.imread(target, cv2.COLOR_BGR2RGB)
    #print(target)
    img_label=[]
    i=0
    for im in data:
        image, label=im
        label=label.strip().upper()
        img = cv2.imread(image, cv2.COLOR_BGR2RGB)  #IMREAD_GRAYSCALE, COLOR_BGR2RGB
        img_label.append((img, label))           
    
    img, _=img_label[1]
    original=img
    try:
        print('Original size',img[0].shape)
    except AttributeError:
        print("shape not found")
        
    denoised_images = denoise_image(img_label)
    img, _=denoised_images[1]
    display(original, img, 'Original', 'Blured: No Noise')
    
    norm_images = normalize_images(denoised_images, target)
    img, _=norm_images[1]
    display(original, img, 'Original', 'Normalized images')
    
    res_img=resize_image(norm_images) 
    img, _=res_img[1]
    display(original, img, 'Original', 'Resized images')
    
    write_2_file(res_img, base_dir_dest)
    

def main():
    global base_dir_breakhis, base_dir_beach, base_dir_dest
    datype='bach'
    data=loadImages_bach(base_dir_beach)
    processing(data, datype, base_dir_dest)
    datype='breakhis'
    data=loadImages_breakhis(base_dir_breakhis)
    processing(data, datype, base_dir_dest)
    
main()    




'''
def segement_images(original, imgs):
    segmented_imgs = []
    sure_foreground_area_imgs = []
    real_process_imgs = []
    for i in range(len(imgs)):
        ret, seg = cv2.threshold(imgs[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)# Further noise removal (Morphology)
        opening = cv2.morphologyEx(seg, cv2.MORPH_OPEN, kernel, iterations=2)    
        sure_bg = cv2.dilate(opening, kernel, iterations=3)# sure background area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)# Finding sure foreground area
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg) # Finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)    
        ret, markers = cv2.connectedComponents(sure_fg)# Marker labelling
        markers = markers + 1# Add one to all labels so that sure background is not 0, but 1    
        markers[unknown == 255] = 0# Now, mark the region of unknown with zero        
        sure_foreground_area_imgs.append(sure_bg)        
        segmented_imgs.append(seg)
        real_process_imgs.append(imgs[i])
    
    # Displaying segmented images
    thresh=segmented_imgs[1]
    display(original, thresh, 'Original', 'Segmented')

    #Displaying segmented back ground
    sure_bg=sure_foreground_area_imgs[1]
    display(original, sure_bg, 'Original', 'Segmented Background')
    
    #Code below only works for input 3 channels (RGB images)
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    # Displaying markers on the image
    display(original, markers, 'Original', 'Marked')
    
    return segmented_imgs, sure_foreground_area_imgs, real_process_imgs
'''