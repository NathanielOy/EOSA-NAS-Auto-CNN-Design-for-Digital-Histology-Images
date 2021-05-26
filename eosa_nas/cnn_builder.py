# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 15:19:21 2021

@author: Oyelade
"""
from __future__ import print_function
from keras import backend
import tensorflow as tf
from tensorflow.keras.models import Sequential
#from keras.models import Sequential
#from pool_helper import PoolHelper
from tensorflow.python.keras.regularizers import l1, l2, L1L2
from tensorflow import keras
#from lrn import LRN
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
#from keras.layers import LSTM, GRU, Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Concatenate, Reshape, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D, ZeroPadding1D, AveragePooling1D
from keras import backend as K
#K.set_image_dim_ordering('th')

'''
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
'''
from tensorflow.python.keras.models import Model
#from tensorflow.python.keras.optimizers import SGD, Adam, Adadelta, RMSprop, Adagrad, Adamax
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, RMSprop, Adagrad, Adamax
from keras.callbacks import Callback
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
import os
import pandas as pd
from functools import partial
from keras.layers import LeakyReLU, PReLU
#from keras.layers.advanced_activations import LeakyReLU, PReLU

class CNNAutoBuilder(object):
    REGULARIZER_RATES=0.0002
       
    def __init__(self, cnn_configs_paras=None, solutions=None, num_classes=None, pos_ids=None):
        #self.filename = "CNN-Sol-" + cnn_configs_paras["solution_number"]
        self.num_classes = num_classes
        self.IMG_WIDTH=cnn_configs_paras["width"]
        self.IMG_HEIGHT=cnn_configs_paras["height"]
        self.solutions=solutions
        self.solutions_models=[]
        self.ID_CONVO=pos_ids['ID_CONVO']
        self.ID_ACTIVATION_FUNC=pos_ids['ID_ACTIVATION_FUNC']
        self.ID_NUMBER_OF_KERNEL=pos_ids['ID_NUMBER_OF_KERNEL']
        self.ID_FILTER_SIZE=pos_ids['ID_FILTER_SIZE']
        self.ID_POOL_SIZE=pos_ids['ID_POOL_SIZE']
        self.ID_POOL_TYPE=pos_ids['ID_POOL_TYPE']
        self.ID_REGULARIZER=pos_ids['ID_REGULARIZER']
        self.ID_FULLY_DENSE_ACTIVATION_FUNC=pos_ids['ID_FULLY_DENSE_ACTIVATION_FUNC']
        self.ID_FULLY_DENSE_DROPOUT=pos_ids['ID_FULLY_DENSE_DROPOUT']
        self.ID_FULLY_DENSE_REGULARIZER=pos_ids['ID_FULLY_DENSE_REGULARIZER']
        self.ID_POS_LEARNING_RATE=pos_ids['ID_POS_LEARNING_RATE']
        self.ID_POS_OPTIMIZER=pos_ids['ID_POS_OPTIMIZER']
        self.ID_POS_BATCH_MODE=pos_ids['ID_POS_BATCH_MODE'] 
        self.ID_POS_EPOCH=pos_ids['ID_POS_EPOCH']
    
    def _build_architecture__(self):        
        for s in self.solutions:
            model = Sequential()
            gb, iz, cb, fcb, lfb=s
            input_pad=self._cnn_input__(model_type=0,is_zeropad=iz)
            if iz:
                model.add(input_pad)
            nb, blks=cb
            n=0
            for convo_block in range(nb):
                blk=blks[convo_block]
                for num_conv_operation in range(blk[self.ID_CONVO]):
                    convo=self._2Dconvolution__(convo_number=n, cf=blk[self.ID_FILTER_SIZE],
                                          ck=blk[self.ID_NUMBER_OF_KERNEL], 
                                          activation=blk[self.ID_ACTIVATION_FUNC],
                                          regularizer=blk[self.ID_REGULARIZER],
                                          is_first=True if n==0 else False,
                                          input_pad=input_pad if n==0 else None,is_zeropad=iz)
                    model.add(convo)
                    n=n+1
                pool=self._2Dpool__(pool_number=n, cps=blk[self.ID_POOL_SIZE], cpt=blk[self.ID_POOL_TYPE])   
                model.add(pool)
                n=n+1
            #full connected layer
            fnb, fb=fcb
            for fully_dense_block in range(fnb):
                model.add(self._flaten__())
                
            faf=fb[self.ID_FULLY_DENSE_ACTIVATION_FUNC]
            fd=fb[self.ID_FULLY_DENSE_DROPOUT]
            fr=fb[self.ID_FULLY_DENSE_REGULARIZER]
            model=self._fully_dense__(dense_number=n, activation=faf, dropout=fd, regularizer=fr, model_type=0, input_pad=input_pad, model=model)            
            summary=self._architecture_summary__(model)
            model=self._compile_model__(model, optimizer=gb[self.ID_POS_OPTIMIZER], learning_rate=gb[self.ID_POS_LEARNING_RATE], lossfunc=lfb)
            mdl_config=(s, model, summary, gb[self.ID_POS_BATCH_MODE], gb[self.ID_POS_EPOCH])
            self.solutions_models.append(mdl_config)
        return self.solutions_models
    
    def _build_architecture2__(self):
        #self._config__()
        for s in self.solutions:
            gb, iz, cb, fcb, lfb=s
            input_pad=self._cnn_input__(model_type=1,is_zeropad=iz)
            nb, blks=cb
            n=0
            #convolutional layers
            for convo_block in range(nb):
                blk=blks[convo_block]
                for num_conv_operation in range(blk[self.ID_CONVO]):
                    convo=self._2Dconvolution__(convo_number=n, cf=blk[self.ID_FILTER_SIZE],
                                          ck=blk[self.ID_NUMBER_OF_KERNEL], 
                                          activation=blk[self.ID_ACTIVATION_FUNC],
                                          regularizer=blk[self.ID_REGULARIZER],
                                          is_first=True if n==0 else False,
                                          input_pad=input_pad if n==0 else None,is_zeropad=iz)(input_pad)
                    input_pad=convo
                    n=n+1
                input_pad=self._2Dpool__(pool_number=n, cps=blk[self.ID_POOL_SIZE], cpt=blk[self.ID_POOL_TYPE])(input_pad)            
                n=n+1
            #full connected layer
            fnb, fb=fcb
            for fully_dense_block in range(fnb):
                input_pad=self._flaten__()(input_pad)
                
            faf=fb[self.ID_FULLY_DENSE_ACTIVATION_FUNC]
            fd=fb[self.ID_FULLY_DENSE_DROPOUT]
            fr=fb[self.ID_FULLY_DENSE_REGULARIZER]
            input_pad=self._fully_dense__(dense_number=n, activation=faf, dropout=fd, regularizer=fr, model_type=1, input_pad=input_pad, model=None)
            model = Model(inputs=input, outputs=[input_pad])
            summary=self._architecture_summary__(model)
            model=self._compile_model__(model, optimizer=gb[self.ID_POS_OPTIMIZER], learning_rate=gb[self.ID_POS_LEARNING_RATE], lossfunc=lfb)
            self.solutions_models.append(s, model, summary, gb[self.ID_POS_BATCH_MODE], gb[self.ID_POS_EPOCH])
        return self.solutions_models
                
    def _cnn_input__(self, model_type=0, is_zeropad=None):
        channel='channels_first'
        if model_type==0:
            inputs=(3, self.IMG_WIDTH, self.IMG_HEIGHT) if K.image_data_format() ==channel  else (self.IMG_WIDTH, self.IMG_HEIGHT, 3)
            inputs =ZeroPadding2D((1,1), 
                                  input_shape=
                                  (3, self.IMG_WIDTH, self.IMG_HEIGHT)
                                  if K.image_data_format() == channel else (self.IMG_WIDTH, self.IMG_HEIGHT, 3)) if is_zeropad else inputs
        else:
            inputs=(3, self.IMG_WIDTH, self.IMG_HEIGHT) if K.image_data_format() == channel else (self.IMG_WIDTH, self.IMG_HEIGHT, 3)
            inputs = ZeroPadding2D(padding=(1, 1))(inputs) if is_zeropad else inputs
        return inputs
        
    def _2Dconvolution__(self, convo_number=None, cf=None, ck=None, activation=None, regularizer=None, is_first=False, input_pad=None,is_zeropad=False):
        regularizer=self._get_regularizer__()         
        if is_first and not(is_zeropad):
            convo2d=Conv2D(int(ck), (int(cf),int(cf)), strides=(1,1), padding='same', 
                           activation=self._get_activation_function__(func=activation), input_shape=input_pad,  
                       name='conv_'+str(convo_number),   kernel_regularizer=regularizer) 
        else:
            convo2d=Conv2D(int(ck), (int(cf),int(cf)), strides=(1,1), padding='same', 
                           activation=self._get_activation_function__(func=activation),
                           name='conv_'+str(convo_number),
                       kernel_regularizer=regularizer) 
        return convo2d
    
    def _2Dpool__(self, pool_number=None, cps=None, cpt=None):
        if cpt=="Max": #data_format="channels_first",
            pool = MaxPooling2D(pool_size=(int(cps), int(cps)), strides=(2,2), padding='same',  name='pool_'+str(pool_number))        
        else:
            pool = AveragePooling2D(pool_size=(int(cps), int(cps)), strides=(2,2), padding='same', name='pool_'+str(pool_number))
        return pool
    
    def _flaten__(self):
        return Flatten()
    
    def _architecture_summary__(self, model=None):
        return model.summary()
        
    def _fully_dense__(self, dense_number=1, activation=None, dropout=None, regularizer=None, model_type=0, input_pad=None, model=None):
        regularizer=self._get_regularizer__(regularizer)         
        if model_type==0:
            model.add(Dropout(rate=float(dropout)))     
            model.add(Dense(self.num_classes, name='loss_classifier_'+str(dense_number), kernel_regularizer=regularizer))
            model.add(Activation(activation, name='class_prob'))
            return model            
        else:
            input_pad=Dropout(rate=float(dropout))(input_pad)
            input_pad=Dense(self.num_classes, name='loss_classifier_'+str(dense_number), kernel_regularizer=regularizer)(input_pad)
            input_pad=Activation(activation, name='class_prob')(input_pad)
            return input_pad
    
    def _get_regularizer__(self, regularizer=None):
        if regularizer=="L1" :
            regularizer=l1(self.REGULARIZER_RATES)
        elif regularizer=="L2":
            regularizer=l2(self.REGULARIZER_RATES)
        #elif regularizer=="L1L2"
        #    regularizer=L1L2(self.REGULARIZER_RATES)
        else:
            regularizer=None
        return regularizer
    
    def _get_activation_function__(self, func=None):
        if func=='leakyrelu':
            return tf.keras.layers.LeakyReLU(alpha=0.1)
        elif func=='parametricrelu':
            return tf.keras.layers.PReLU() #alpha=0.1
        else:
            return func
    
    def _config__(self):
        if keras.backend.backend() == 'tensorflow':
            from tensorflow.python.keras import backend as K
            import tensorflow as tf
            tf.compat.v1.disable_eager_execution()
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()
            from tensorflow.python.keras.utils.conv_utils import convert_kernel
            from keras.backend.tensorflow_backend import set_session, clear_session, get_session
            NUM_PARALLEL_EXEC_UNITS=2
            config =  tf.compat.v1.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.333
            session = tf.compat.v1.Session(config=config) #tf.Session(config=config)
            K.set_session(session)            
            # Using the Winograd non-fused algorithms provides a small performance boost.
            os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            #To not use GPU, a good solution is to not allow the environment to see any GPUs by setting the environmental variable CUDA_VISIBLE_DEVICES.
            os.environ["CUDA_VISIBLE_DEVICES"]="1"
            os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
            os.environ["KMP_BLOCKTIME"] = "30"
            os.environ["KMP_SETTINGS"] = "1"
            os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    
    def _compile_model__(self, model=None, optimizer=None, learning_rate=None, lossfunc=None):
        momentum=0.9 #0.0,  0.5,  0.9,  0.99
        if optimizer == 'SGD':
            optim = SGD(lr=learning_rate, decay=1e-6, momentum=momentum, nesterov=False)
        elif optimizer == 'RMSprop':
            optim = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,rho=0.9,momentum=momentum,epsilon=1e-07, centered=False, name="RMSprop")
        elif optimizer == 'Adagrad':
            optim = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
        elif optimizer == 'Adam': #adam
            optim=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
        elif optimizer == 'Adadelta': #Adadelta
            optim=Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-07, name="Adadelta")
        elif optimizer == 'Adamax': #Adamax
            optim=Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07) #, name="Adamax"
        elif optimizer == 'Momentum': #Momentum
            optim=SGD(lr=learning_rate, decay=1e-6, momentum=momentum, nesterov=False)
        else: #optimizer == 'Nestrov': #Nestrov
            optim= SGD(lr=learning_rate, decay=1e-6, momentum=0.0, nesterov=True)
        
        model.compile(optimizer=optim, loss=lossfunc, metrics=['accuracy']) #'categorical_crossentropy'        
        return model
    
    def _top1_architecture__(self,x_train, y_train,x_test,y_test, optimizer,learning_rate,epoch):        
        model = Sequential()
        num_classes=12
        activation='leakyrelu'
        model.add(ZeroPadding2D((1,1), input_shape=(224, 224, 3)))
        model.add(Conv2D(32, (9,9), strides=(1,1), padding='same',
                              activation='relu', #input_shape=(244, 244, 3), 
                              name='conv1_1/3x3_s1', kernel_regularizer=l2(0.0002)
                  ))
        model.add(AveragePooling2D(pool_size=(2,2), strides=(1, 1), padding='same', name='pool1/2x2_s1'))
        
        
        model.add(Conv2D(64, (9,9), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv2_1/3x3_reduce', kernel_regularizer=l1(0.0002)))
        model.add(Conv2D(64, (9,9), strides=(1, 1), padding='same', 
                              activation=partial(tf.nn.leaky_relu, alpha=0.01), 
                              name='conv2_2/3x3_reduce', kernel_regularizer=l1(0.0002)))
        model.add(Conv2D(64, (9,9), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv2_3/3x3_reduce', kernel_regularizer=l1(0.0002)))
        model.add(AveragePooling2D(pool_size=(2,2), strides=(1, 1), padding='same', name='pool2/2x2_s1'))
                  
        model.add(Flatten())
        model.add(Flatten())
        model.add(Dropout(rate=0.48))
        model.add(Dense(num_classes, name='loss3/classifier', #kernel_regularizer=l2(0.0002)
                        ))
        model.add(Activation('softmax', name='prob'))
        
        model.summary()
        model=self._compile_model__(model=model, optimizer=optimizer, learning_rate=learning_rate, lossfunc='categorical_crossentropy')
        half=int(len(x_train)/2)
        test_data=x_train[0:half], y_train[0:half]
        batch_size=32
        test_records=len(test_data[0])
        ml = model.fit(x_train, y_train, epochs=epoch, 
                       batch_size=batch_size, verbose=1, 
                       initial_epoch=0,
                       validation_data=test_data, 
                       validation_steps=test_records//batch_size,
                       workers=0)
       
        y_pred = model.predict(x_test)
        avg_pred=tf.keras.losses.categorical_crossentropy(y_test, y_pred)
        return ml, y_pred, y_test,avg_pred
    
    
    def _top2_architecture__(self,x_train, y_train,x_test,y_test, optimizer,learning_rate,epoch):        
        model = Sequential()
        num_classes=12
        activation='leakyrelu'
        model.add(ZeroPadding2D((1,1), input_shape=(224, 224, 3)))
        model.add(Conv2D(32, (3,3), strides=(1,1), padding='same',
                              activation='relu', #input_shape=(244, 244, 3), 
                              name='conv1_1/3x3_s1', kernel_regularizer=l2(0.0002)
                  ))
        model.add(Conv2D(32, (3,3), strides=(1,1), padding='same',
                              activation='relu', #input_shape=(244, 244, 3), 
                              name='conv1_2/3x3_s1', 
                  ))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding='same', name='pool1/2x2_s1'))
        
        
        model.add(Conv2D(64, (3,3), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv2_1/3x3_reduce'))
        model.add(Conv2D(64, (3,3), strides=(1, 1), padding='same', 
                              activation=partial(tf.nn.leaky_relu, alpha=0.01), 
                              name='conv2_2/3x3_reduce'))
        model.add(Conv2D(64, (3,3), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv2_3/3x3_reduce'))
        model.add(Conv2D(64, (3,3), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv2_4/3x3_reduce'))
        model.add(AveragePooling2D(pool_size=(2,2), strides=(1, 1), padding='same', name='pool2/2x2_s1'))
        
        
        model.add(Conv2D(128, (3,3), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv3_1/3x3_reduce'))
        model.add(Conv2D(128, (3,3), strides=(1, 1), padding='same', 
                              activation=partial(tf.nn.leaky_relu, alpha=0.01), 
                              name='conv3_2/3x3_reduce'))
        model.add(Conv2D(128, (3,3), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv3_3/3x3_reduce'))
        model.add(Conv2D(128, (3,3), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv3_4/3x3_reduce'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding='same', name='pool3/2x2_s1'))
        
        model.add(Flatten())
        model.add(Flatten())
        model.add(Flatten())
        model.add(Dropout(rate=0.45))
        model.add(Dense(num_classes, name='loss3/classifier', kernel_regularizer=l2(0.0002)
                        ))
        model.add(Activation('softmax', name='prob'))
        
        model.summary()
        model=self._compile_model__(model=model, optimizer=optimizer, learning_rate=learning_rate, lossfunc='categorical_crossentropy')
        half=int(len(x_train)/2)
        test_data=x_train[0:half], y_train[0:half]
        batch_size=32
        test_records=len(test_data[0])
        ml = model.fit(x_train, y_train, epochs=epoch, 
                       batch_size=batch_size, verbose=1, 
                       initial_epoch=0,
                       validation_data=test_data, 
                       validation_steps=test_records//batch_size,
                       workers=0)
       
        y_pred = model.predict(x_test)
        avg_pred=tf.keras.losses.categorical_crossentropy(y_test, y_pred)
        return ml, y_pred, y_test,avg_pred
    
    
    def _top3_architecture__(self,x_train, y_train,x_test,y_test, optimizer,learning_rate,epoch):        
        model = Sequential()
        num_classes=12
        activation='leakyrelu'
        model.add(ZeroPadding2D((1,1), input_shape=(224, 224, 3)))
        model.add(Conv2D(32, (9,9), strides=(1,1), padding='same',
                              activation='relu', #input_shape=(244, 244, 3), 
                              name='conv1_1/3x3_s1', kernel_regularizer=l2(0.0002)
                  ))
        model.add(AveragePooling2D(pool_size=(2,2), strides=(1, 1), padding='same', name='pool1/2x2_s1'))
        
        
        model.add(Conv2D(64, (9,9), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv2_1/3x3_reduce', kernel_regularizer=l1(0.0002)))
        model.add(Conv2D(64, (9,9), strides=(1, 1), padding='same', 
                              activation=partial(tf.nn.leaky_relu, alpha=0.01), 
                              name='conv2_2/3x3_reduce', kernel_regularizer=l1(0.0002)))
        model.add(Conv2D(64, (9,9), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv2_3/3x3_reduce', kernel_regularizer=l1(0.0002)))
        model.add(AveragePooling2D(pool_size=(2,2), strides=(1, 1), padding='same', name='pool2/2x2_s1'))
                  
        model.add(Flatten())
        model.add(Flatten())
        model.add(Dropout(rate=0.45))
        model.add(Dense(num_classes, name='loss3/classifier', #kernel_regularizer=l2(0.0002)
                        ))
        model.add(Activation('softmax', name='prob'))
        
        model.summary()
        model=self._compile_model__(model=model, optimizer=optimizer, learning_rate=learning_rate, lossfunc='categorical_crossentropy')
        half=int(len(x_train)/2)
        test_data=x_train[0:half], y_train[0:half]
        batch_size=32
        test_records=len(test_data[0])
        ml = model.fit(x_train, y_train, epochs=epoch, 
                       batch_size=batch_size, verbose=1, 
                       initial_epoch=0,
                       validation_data=test_data, 
                       validation_steps=test_records//batch_size,
                       workers=0)
       
        y_pred = model.predict(x_test)
        avg_pred=tf.keras.losses.categorical_crossentropy(y_test, y_pred)
        return ml, y_pred, y_test,avg_pred
    
    def _top4_architecture__(self,x_train, y_train,x_test,y_test, optimizer,learning_rate,epoch):        
        model = Sequential()
        num_classes=12
        activation='leakyrelu'
        model.add(ZeroPadding2D((1,1), input_shape=(224, 224, 3)))
        model.add(Conv2D(32, (3,3), strides=(1,1), padding='same',
                              activation='relu', #input_shape=(244, 244, 3), 
                              name='conv1_1/3x3_s1', kernel_regularizer=l2(0.0002)
                  ))
        model.add(Conv2D(32, (3,3), strides=(1,1), padding='same',
                              activation='relu', #input_shape=(244, 244, 3), 
                              name='conv1_2/3x3_s1', kernel_regularizer=l2(0.0002)
                  ))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding='same', name='pool1/2x2_s1'))
        
        
        model.add(Conv2D(64, (3,3), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv2_1/3x3_reduce', kernel_regularizer=l1(0.0002)))
        model.add(Conv2D(64, (3,3), strides=(1, 1), padding='same', 
                              activation=partial(tf.nn.leaky_relu, alpha=0.01), 
                              name='conv2_2/3x3_reduce', kernel_regularizer=l1(0.0002)))
        model.add(Conv2D(64, (3,3), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv2_3/3x3_reduce', kernel_regularizer=l1(0.0002)))
        model.add(Conv2D(64, (3,3), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv2_4/3x3_reduce', kernel_regularizer=l1(0.0002)))
        model.add(AveragePooling2D(pool_size=(2,2), strides=(1, 1), padding='same', name='pool2/2x2_s1'))
        
        
        model.add(Conv2D(128, (3,3), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv3_1/3x3_reduce'))
        model.add(Conv2D(128, (3,3), strides=(1, 1), padding='same', 
                              activation=partial(tf.nn.leaky_relu, alpha=0.01), 
                              name='conv3_2/3x3_reduce'))
        model.add(Conv2D(128, (3,3), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv3_3/3x3_reduce'))
        model.add(Conv2D(128, (3,3), strides=(1, 1), padding='same', 
                              activation='relu', 
                              name='conv3_4/3x3_reduce'))
        model.add(AveragePooling2D(pool_size=(2,2), strides=(1, 1), padding='same', name='pool3/2x2_s1'))
        
        model.add(Flatten())
        model.add(Flatten())
        model.add(Flatten())
        model.add(Dropout(rate=0.50))
        model.add(Dense(num_classes, name='loss3/classifier', kernel_regularizer=l2(0.0002)
                        ))
        model.add(Activation('softmax', name='prob'))
        
        model.summary()
        model=self._compile_model__(model=model, optimizer=optimizer, learning_rate=learning_rate, lossfunc='categorical_crossentropy')
        half=int(len(x_train)/2)
        test_data=x_train[0:half], y_train[0:half]
        batch_size=32
        test_records=len(test_data[0])
        ml = model.fit(x_train, y_train, epochs=epoch, 
                       batch_size=batch_size, verbose=1, 
                       initial_epoch=0,
                       validation_data=test_data, 
                       validation_steps=test_records//batch_size,
                       workers=0)
       
        y_pred = model.predict(x_test)
        avg_pred=tf.keras.losses.categorical_crossentropy(y_test, y_pred)
        return ml, y_pred, y_test,avg_pred
    
    def _top5_architecture__(self,x_train, y_train,x_test,y_test, optimizer,learning_rate,epoch):        
        model = Sequential()
        num_classes=12
        model.add(ZeroPadding2D((1,1), input_shape=(224, 224, 3)))
        model.add(Conv2D(16, (2,2), strides=(1,1), padding='same',
                              activation='relu', #input_shape=(244, 244, 3), 
                              name='conv1_1/3x3_s1', #kernel_regularizer=l2(0.0002)
                  ))
        model.add(Conv2D(16, (2,2), strides=(1,1), padding='same', 
                              activation='relu', 
                              name='conv1_2/3x3_s1', #kernel_regularizer=l2(0.0002)
                  ))
        model.add(Conv2D(16, (2,2), strides=(1,1), padding='same', 
                              activation='relu', 
                              name='conv1_3/3x3_s1', #kernel_regularizer=l2(0.0002)
                  ))
        model.add(AveragePooling2D(pool_size=(3,3), strides=(1, 1), padding='same', name='pool1/2x2_s1'))
        
        model.add(Flatten())
        model.add(Flatten())
        model.add(Dropout(rate=0.47))
        model.add(Dense(num_classes, name='loss3/classifier', #kernel_regularizer=l2(0.0002)
                        ))
        model.add(Activation('softmax', name='prob'))
        
        model.summary()
        model=self._compile_model__(model=model, optimizer=optimizer, learning_rate=learning_rate, lossfunc='categorical_crossentropy')
        half=int(len(x_train)/2)
        test_data=x_train[0:half], y_train[0:half]
        batch_size=32
        test_records=len(test_data[0])
        ml = model.fit(x_train, y_train, epochs=epoch, 
                       batch_size=batch_size, verbose=1, 
                       initial_epoch=0,
                       validation_data=test_data, 
                       validation_steps=test_records//batch_size,
                       workers=0)
       
        y_pred = model.predict(x_test)
        avg_pred=tf.keras.losses.categorical_crossentropy(y_test, y_pred)
        return ml, y_pred, y_test,avg_pred
    
    '''
    def _get_average_error__(self, individual=None, X_data=None, y_data=None):
        t1 = time()
        weights = [rand(*w.shape) for w in self.model_rnn.get_weights()]
        ws = []
        cur_point = 0
        for wei in weights:
            ws.append(reshape(individual[cur_point:cur_point + len(wei.reshape(-1))], wei.shape))
            cur_point += len(wei.reshape(-1))

        self.model_rnn.set_weights(ws)
        y_pred = self.model_rnn.predict(X_data)
        # print("GAE time: {}".format(time() - t1))

        # return [mean_squared_error(y_pred, y_data), mean_absolute_error(y_pred, y_data)]
        return tf.keras.losses.categorical_crossentropy(y_pred, y_data)

    def _objective_function__(self, solution=None):
        weights = [rand(*w.shape) for w in self.model_rnn.get_weights()]
        ws = []
        cur_point = 0
        for wei in weights:
            ws.append(reshape(solution[cur_point:cur_point + len(wei.reshape(-1))], wei.shape))
            cur_point += len(wei.reshape(-1))

        self.model_rnn.set_weights(ws)
        y_pred = self.model_rnn.predict(self.X_train)
        return tf.keras.losses.categorical_crossentropy(y_pred, self.y_train)
    '''