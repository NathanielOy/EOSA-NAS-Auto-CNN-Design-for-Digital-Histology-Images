# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:58:57 2021

@author: Oyelade
"""
import tensorflow as tf
from keras import backend as K
from numpy import array
from keras.utils import to_categorical
from time import time
import os
import numpy as np
import cv2
import random

class EOSANASSearchStrategy(object):
    
    def __init__(self, search_strategy_paras=None, search_paras=None, cnn_configs_paras=None):
        self.model_config = search_strategy_paras['model_config'] 
        self.log=search_strategy_paras['log_mode']
        self.num_classes=search_paras["num_classes"]
        self.classes=search_paras["classes"]
        self.train_result=search_strategy_paras['train_result'] 
        self.test_result=search_strategy_paras['test_result'] 
        self.train_buffer_result=[] 
        self.test_buffer_result=[] 
        self.input_dataset=search_strategy_paras['input_dataset']
        self.IMG_WIDTH=cnn_configs_paras["width"]
        self.IMG_HEIGHT=cnn_configs_paras["height"]
        self.history=[]
    
    def _optimize__(self):
        pass
    
    def _show_summary__(self):
        for i in range(len(self.model_config)):
            _, _, model_summary, _, _=self.model_config[i]
            print('Summary of Solution'+str(i))
            print(model_summary)
            print('--------------------------------------------------------------------------------------------')
        
    def _training_all__(self):
        #print('Train: Result ')
        #print('--------------------------------------------------------------------------------------------')
        #print('| Solution No  | time_total_train   |   loss_train   |   accuracy_train   |   batch_mode  |')
        #print('--------------------------------------------------------------------------------------------')
        
        evaluated_solutions=[]
        for i in range(len(self.model_config)):
            x_train, y_train, x_test, y_test, batch_size=self._processing_train_input__(self.model_config[i])
            model_summary, batch_mode, time_total_train, loss_train, accuracy_train, val_loss_train, val_accuracy_train=self._train__(self.model_config[i], batch_size=batch_size, x_train=x_train, y_train=y_train)
            self.train_buffer_result.append((batch_mode, time_total_train, loss_train, accuracy_train, val_loss_train, val_accuracy_train))
            #print('| Train: Sol'+str(i)+' | '+str(time_total_train)+' |   '+str(loss_train)+' | '+str(accuracy_train)+' | '+str(batch_mode)+' |')
            time_predict, avg_pred, y_test, y_pred=self._predict__(self.model_config[i], x_test=x_test, y_test=y_test)
            self.test_buffer_result.append((i, time_predict, avg_pred))
            #print('| Predict: Sol'+str(i)+' | '+str(time_predict)+' |   '+str(avg_pred)+' | ')
            raw_sol, _, _, _, _,=self.model_config[i]
            evaluated_solutions.append((raw_sol, loss_train, accuracy_train, val_loss_train, val_accuracy_train, time_total_train, time_predict, avg_pred, y_test, y_pred))            
        return evaluated_solutions
    
    def make_inference__(model, single_img_file, height, width):
        img = cv2.imread(single_img_file)
        orig = img.copy() # save for plotting later on 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # gray scaling 
        img = np.asarray(img)              # convert to array 
        img = cv2.resize(img, (height, width))   # resize to target shape 
        img = cv2.bitwise_not(img)         # [optional] my input was white bg, I turned it to black - {bitwise_not} turns 1's into 0's and 0's into 1's
        img = img / 255                    # normalize 
        #img = img.reshape(1, 784)          # reshaping 
        pred = model.predict(img)
        plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        plt.title(np.argmax(pred, axis=1))
        plt.show()
        return img 

    def _train__(self, model_config=None, batch_size=None, x_train=None, y_train=None):
        raw_sol, model, model_summary, batch_mode, epoch=model_config        
        time_total_train = time()
        batch_size=32
        half=int(len(x_train)/2)
        test_data=x_train[0:half], y_train[0:half]
        test_records=len(test_data[0])
        ml = model.fit(x_train, y_train, epochs=epoch, 
                       batch_size=batch_size, verbose=self.log, 
                       initial_epoch=0,
                       validation_data=test_data, 
                       validation_steps=test_records//batch_size,
                       workers=0)
        time_total_train = round(time() - time_total_train, 4)
        self.history.append(ml.history)
        loss_train = ml.history["loss"]
        accuracy_train = ml.history["accuracy"]
        val_loss_train = ml.history["val_loss"]
        val_accuracy_train = ml.history["val_accuracy"]        
        return model_summary, batch_mode, time_total_train, loss_train, accuracy_train, val_loss_train, val_accuracy_train
    
    def _predict__(self, model_config=None, x_test=None, y_test=None):
        raw_sol, model, model_summary, batch_mode, epoch=model_config        
        time_predict = time()
        y_pred = model.predict(x_test)
        avg_pred=tf.keras.losses.categorical_crossentropy(y_test, y_pred)
        avg_pred=avg_pred.numpy()
        time_predict = round(time() - time_predict, 8)
        return time_predict, avg_pred, y_test, y_pred
    
    def _processing_train_input__(self, config=None):
        _,_, _, batch_mode, _=config
        nFiles=len(os.listdir(self.input_dataset))
        if K.image_data_format() == 'channels_first':
            dim=(3, self.IMG_WIDTH, self.IMG_HEIGHT)
            img_data_array = np.empty((nFiles, 3, self.IMG_WIDTH, self.IMG_HEIGHT))
        else:
            dim=(self.IMG_WIDTH, self.IMG_HEIGHT, 3)
            img_data_array = np.empty((nFiles, self.IMG_WIDTH, self.IMG_HEIGHT, 3))
            
        class_name = np.empty((nFiles, ))
        n=0
        for file in os.listdir(self.input_dataset):
            image_path= os.path.join(self.input_dataset, file)            
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB) #image= np.array(Image.open(image_path))
            image=cv2.resize(image, (self.IMG_WIDTH, self.IMG_HEIGHT),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image=image.reshape(dim)
            image /= 255 
            # one hot encode
            lbl=file.split('_')[0]
            label = self.classes[lbl] #, self.num_classes)
            img_data_array[n, :, :, :] = image
            class_name[n] = label
            n=n+1
        
        
        class_name=array(class_name)
        class_name = to_categorical(class_name, self.num_classes)
        
        if batch_mode == 0: #random mode=0  randomly choose the size to use
            batch_size=random.randint(int(n/3), n-1)
        elif batch_mode == 1: #batch mode=1 chooses all samples to use
            batch_size=n-1
        else: #batch_mode ==3, mini-batch mode 
            batch_size=256
        
        x_train, y_train=img_data_array[:batch_size], class_name[:batch_size]
        twenty_percent=int((len(x_train) * 20)/100)
        x_test, y_test=x_train[:twenty_percent], y_train[:twenty_percent]       
        
        # summarize pixel values
        #print('Train', x_train.min(), x_train.max(), x_train.mean(), x_train.std())
        #print('Test', x_test.min(), x_test.max(), x_test.mean(), x_test.std())
        return x_train, y_train, x_test, y_test, batch_size
    
    
    