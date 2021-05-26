#!/usr/bin/env python
#coding:utf-8

""" This script creates the 5 folds used in the experiments reported in [1].

    It ASSUMES:
        - that the path is relative to the current directory.
        - that BreaKHis_v1.tar.gz was decompressed in current directory.

    It REQUIRES:
        - text files dsfold1.txt, ... dsfold5.txt located in current directory.

    ****Approximately 20 GB of disk space will be allocated for fold1,... fold5 directories.


    -------
    [1] Spanhol, F.A.; Oliveira, L.S.; Petitjean, C.; Heutte, L. "A Dataset for Breast Cancer Histopathological Image Classification". Biomedical Engineering, IEEE Transactions on. Year: 2015, DOI: 10.1109/TBME.2015.2496264

"""
__author__ = "Fabio Alexandre Spanhol"
__email__ = "faspanhol@gmail.com"


import sys
import os
import shutil
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import cv2
plt.style.use('seaborn-white')
from matplotlib.ticker import ScalarFormatter

base_dir='../../Data/BreakHis/'
base_dir_beach='../../Data/Bach/'

# -----------------------------------------------------------------------------
def get_title_index(i):
    idx='abcdefghijklmnopqrstuvwxyz'
    idx='('+idx[i]+')'
    return idx

def get_class_breakhis(i, imgclass, imgtype):
    img_dict={ 'A': 'Adenosis', 'F': 'Fibroadenoma','PT':'Phyllodes tumor', 'TA': 'Tubular adenona', 
              'DC': 'Carcinoma', 'LC':'Lobular carcinoma', 'MC': 'Mucinous carcinoma', 'PC': 'Papillary carcinoma'}
    img_cls=get_title_index(i)+' '+imgclass+':'+img_dict[imgtype]
    return img_cls

def show_samples_breakhis():
    #'SOB_B_F-14-14134-40-001', 'SOB_B_PT-14-22704-40-012', 'SOB_B_TA-14-13200-40-001', 'SOB_M_LC-14-12204-40-017',
    #'SOB_B_F-14-14134-100-001', 'SOB_B_PT-14-22704-100-012', 'SOB_B_TA-14-13200-100-001','SOB_M_LC-14-12204-100-031',
    #'SOB_B_F-14-14134-200-001', 'SOB_B_PT-14-22704-200-012', 'SOB_B_TA-14-13200-200-001', 'SOB_M_LC-14-12204-200-031',
    #'SOB_B_F-14-14134-400-001', 'SOB_B_PT-14-22704-400-012', 'SOB_B_TA-14-13200-400-001', 'SOB_M_LC-14-12204-400-031',
    imgs=['SOB_B_A-14-22549AB-40-001','SOB_M_DC-14-2773-40-007', 'SOB_M_MC-14-13418DE-40-007', 'SOB_M_PC-14-9146-40-009',
          'SOB_B_A-14-22549AB-100-001','SOB_M_DC-14-2773-100-007', 'SOB_M_MC-14-13418DE-100-007', 'SOB_M_PC-14-9146-100-009',
          'SOB_B_A-14-22549AB-200-001','SOB_M_DC-14-2773-200-007', 'SOB_M_MC-14-13418DE-200-007', 'SOB_M_PC-14-9146-200-009',
          'SOB_B_A-14-22549AB-400-001','SOB_M_DC-14-2773-400-007', 'SOB_M_MC-14-13418DE-400-007', 'SOB_M_PC-14-9146-400-009',]
    imgs_per_layer=4
    sub_dir=['40X/', '100X/', '200X/', '400X/']
    img_dir=base_dir+'fold1/train/'
    n=0
    cols=imgs_per_layer
    rows=4            
    fig, axs = plt.subplots(rows, cols, figsize=(12,12))
    fig.subplots_adjust(wspace=0.0, hspace=-0.6)
    
    i=0
    k=0
    while k < len(imgs):        
        j=0
        for img in imgs[k:(k+imgs_per_layer)]:
            img_label=img.split('_')
            img_label2=img_label[2].split('-')
            if k ==0:
                axs[i, j].set_title(get_class_breakhis(n, img_label[1], img_label2[0]))
            n=n+1
            img= img_dir+sub_dir[i]+img+'.png'
            imgp=cv2.imread(img)
            axs[i, j].imshow(imgp)        
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)
            axs[i, j].spines['bottom'].set_visible(False)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            j=j+1
        i=i+1
        k=(k+imgs_per_layer)

def get_class_beach(i, imgclass):
    #img_dict={ 'n': 'Normal', 'b': 'Benign','is':'In Situ Carcinoma', 'iv': 'Invasive Carcinoma'}
    img_cls=get_title_index(i)
    return img_cls

def show_samples_beach():
    imgs=['n036', 'b002', 'is006', 'iv026','n036', 'b002', 'is006', 'iv026']
    imgs_per_layer=4
    img_dir=base_dir_beach
    n=0
    cols=imgs_per_layer
    rows=2
    fig, axs = plt.subplots(rows, cols, figsize=(12,12))
    fig.subplots_adjust(wspace=0.0, hspace=-0.6)
    
    j=0
    i=0
    for img in imgs:
        img_label=img[:2]
        if n == 4:
            i=1
            j=0
        axs[i, j].set_title(get_class_beach(n, img_label))
        img= img_dir+img+'.jpg'
        imgp=cv2.imread(img)
        axs[i, j].imshow(imgp)        
        axs[i, j].spines['right'].set_visible(False)
        axs[i, j].spines['top'].set_visible(False)
        axs[i, j].spines['left'].set_visible(False)
        axs[i, j].spines['bottom'].set_visible(False)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        j=j+1
        n=n+1
        
    
def create_folds_from_ds(dst_path='../../Data/BreakHis/', folds=(1,2,3,4,5)):
    """Creates a structure of directories containing images
        selected from BreaKHis_v1 dataset.
    """
    root_dir = base_dir+'BreaKHis_v1/histology_slides/breast'
    srcfiles = {'DC': '%s/malignant/SOB/ductal_carcinoma/%s/%sX/%s',
                'LC': '%s/malignant/SOB/lobular_carcinoma/%s/%sX/%s',
                'MC': '%s/malignant/SOB/mucinous_carcinoma/%s/%sX/%s',
                'PC': '%s/malignant/SOB/papillary_carcinoma/%s/%sX/%s',
                'A': '%s/benign/SOB/adenosis/%s/%sX/%s',
                'F': '%s/benign/SOB/fibroadenoma/%s/%sX/%s',
                'PT': '%s/benign/SOB/phyllodes_tumor/%s/%sX/%s',
                'TA': '%s/benign/SOB/tubular_adenoma/%s/%sX/%s'}

    for nfold in folds:
        # directory for nth-fold
        dst_dir = dst_path + '/fold%s' % nfold
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        dst_dir = dst_dir + '/%s'

        # image list
        db = open('dsfold%s.txt' % nfold)

        for row in db.readlines():
            columns = row.split('|')
            imgname = columns[0]
            mag = columns[1]  # 40, 100, 200, or 400
            grp = columns[3].strip()  # train or test

            dst_subdir = dst_dir % grp
            if not os.path.exists(dst_subdir):
                os.mkdir(dst_subdir)

            dst_subdir = dst_subdir + '/%sX/' % mag
            if not os.path.exists(dst_subdir):
                os.mkdir(dst_subdir)

            tumor = imgname.split('-')[0].split('_')[-1]
            srcfile = srcfiles[tumor]

            s = imgname.split('-')
            sub = s[0] + '_' + s[1] + '-' + s[2]

            srcfile = srcfile % (root_dir, sub, mag, imgname)

            dstfile = dst_subdir + imgname

            print ("Copying from [%s] to [%s]" % (srcfile, dstfile))
            shutil.copy(srcfile, dstfile)
        print ('\n\n\t\tFold #%d finished.\n' % nfold)
    db.close()
    print ("\nProcess completed.")
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    #create_folds_from_ds()
    #show_samples_breakhis()
    show_samples_beach()
