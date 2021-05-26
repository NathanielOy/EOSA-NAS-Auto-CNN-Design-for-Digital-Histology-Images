# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:50:44 2021

@author: Oyelade
"""

import numpy as np
import random
from utils.FunctionUtil import *

"""
Constants and values describing rates and variables
"""
'''
Settings from the paper
--------------------------------------------------------------------------------------------
 Notation       Definition                                                     Range of Value
--------------------------------------------------------------------------------------------
    π     Recruitment rate of susceptible human individuals                          Variable
    ŋ    Decay rate of Ebola virus in the environment                               (0, )
    α    Rate of hospitalization of infected individuals                               (0, 1)
        Disease-induced death rate of human individuals                               [0.4, 0.9]
    β1    Contact rate of infectious human individuals                               Variable
    β2    Contact rate of pathogen individuals/environment                           Variable
    β3    Contact rate of deceased human individuals                                   Variable
    β4    Contact rate of recovered human individuals                                   Variable
        Recovery rate of human individuals                                           (0, 1)
        Natural death rate of human individuals                                       (0, 1)
        Rate of  burial of deceased human individuals                               (0, 1)
        Rate of vaccination of individuals                                           (0, 1)
        Rate of response to hospital treatment                                       (0, 1)
        Rate response to vaccination                                               (0, 1)
'''
π=0.1 #Recruitment rate of susceptible human individuals
ŋ=np.random.rand() #Decay rate of Ebola virus in the environment
α=np.random.rand() #Rate of hospitalization of infected individuals
dis=random.uniform(0.4, 0.9)#Disease-induced death rate of human individuals
β_1=0.1#Contact rate of infectious human individuals
β_2=0.1#Contact rate of pathogen individuals/environment
β_3=0.1#Contact rate of deceased human individuals
β_4=0.1#Contact rate of recovered human individuals
rr=np.random.rand() #Recovery rate of human individuals
dr=np.random.rand() #Natural death rate of human individuals
br=np.random.rand() #Rate of  burial of deceased human individuals
vr=np.random.rand() #Rate of vaccination of individuals
hr=np.random.rand() #Rate of response to hospital treatment
vrr=np.random.rand() #Rate response to vaccination
qrr=np.random.rand()	#Rate of quarantine of infected individuals

save_results_dir='./results' #'/content/gdrive/MyDrive/Paper13/code/results/'  #
train_result='train/'
test_result='test/'
input_dataset='../Data/Preprocessed/' #'/content/gdrive/MyDrive/Paper13/Data/Preprocessed/' #
num_classes=12 #len(classes)
log_mode=1 #verbose=0 will show you nothing (silent), verbose=1 will show you an animated progress bar, verbose=2 will just mention the number of epoch
number_of_runs=1
cnn_epoch=3 #
eosa_epoch=500
problem_size=50

modelrates = {
    "recruitment_rate": π,
    "decay_rate": ŋ,
    "hospitalization_rate": α,
    "disease_induced_death_rate": dis,
    "contact_rate_infectious": β_1,
    "contact_rate_pathogen": β_2,
    "contact_rate_deceased": β_3,
    "contact_rate_recovered": β_4,
    "recovery_rate": rr,
    "natural_death_rate": dr,
    "burial_rate": br,
    "vacination_rate": vr,
    "hospital_treatment_rate": hr,
    "vaccination_response_rate": vrr,
    "quarantine_rate": qrr
}

blocks={"gb":0, "cb":1, "fb":2, "lb":3}
subblocks={
   "Gb": 1, "Gl":2, "Go":3, "Ge":4, "IZ":5, "CL":6, "CC": 7, "CAF":8, 
   "CK":9, "CF":10, "CPS":11, "CPT":12, "CR":13, "FL":14, "FAF":15, 
   "FD":16, "FR":17, "LFL":0
}
learning_rates={#0:1e-00, 1:1e-01, 
                0:1e-02, 1:1e-03, 2:1e-04, 3:1e-05, 4:1e-06, 
                #5:1e-07, 6:1e-08, 9:5e-00, 10:5e-01, 
                5:5e-02, 6:5e-03, 7:5e-04, 8:5e-05, 
                #11:5e-06, 12:5e-07, 13:5e-08
                }
optimizers={0:"SGD", 
            1:"Adam", 
            2:"RMSprop", 
            3:"Adagrad", 
            4:"Nestrov", 
            5:"Adadelta", 
            6:"Adamax", 
            7:"Momentum", 
            #8:" Nestrov Accelerated Gradient"
          }
activations={0:"relu", 1:"relu"}#, 2:"parametricrelu"} leakyrelu
pooling_operations={0:"Max", 1:"Avg"}
regularizers={0:"None", 1:"L1", 2:"L2"}#, 2:"L1L2"}
factivations={0:"softmax", 1:"softmax"}#, 2:"linear"}
loss_functions={0: 'categorical_crossentropy', #1: 'binary_crossentropy', 
                1: 'categorical_crossentropy' #sparse_categorical_crossentropy', 
                #2: 'kl_divergence'
                }

bound_labels={"lfb":0, "fl":1, "faf":2, "fd":3, "fr":4, "cc":5, "caf":6, "ck":7, "cf":8, "cps":9, "cpt":10, "cr":11, "gl":12, "go":13}
upper_lower_bounds={'GB_MIN':0, 'GB_MAX':2, 
                    'GL_MIN':0, 'GL_MAX':8, 
                    'GO_MIN':0, 'GO_MAX':7, 
                    'GE_MIN':1, 'GE_MAX':2,
                    'IZ_MIN':0, 'IZ_MAX':1, 
                    'CL_MIN':2, 'CL_MAX':6, 
                    'CC_MIN':0, 'CC_MAX':2, 
                    'CAF_MIN':0, 'CAF_MAX':1,
                    'CK_MIN':3, 'CK_MAX':10, 
                    'CF_MIN':0, 'CF_MAX':10, 
                    'CPS_MIN':0, 'CPS_MAX':2, 
                    'CPT_MIN':0, 'CPT_MAX':1, 
                    'CR_MIN':0, 'CR_MAX':1, 
                    'FL_MIN':0, 'FL_MAX':1, 
                    'FAF_MIN':0, 'FAF_MAX':1,
                    'FD_MIN':2.0, 'FD_MAX':2.5, 
                    'FR_MIN':0, 'FR_MAX':1, 'LFL_MIN':0, 'LFL_MAX':1}

pos_ids={'ID_CONVO':0, 'ID_ACTIVATION_FUNC':1, 'ID_NUMBER_OF_KERNEL':2, 'ID_FILTER_SIZE':3, 'ID_POOL_SIZE':4, 'ID_POOL_TYPE':5,
         'ID_REGULARIZER':6, 'ID_FULLY_DENSE_ACTIVATION_FUNC':0, 'ID_FULLY_DENSE_DROPOUT':1, 'ID_FULLY_DENSE_REGULARIZER':2,
         'ID_POS_LEARNING_RATE':1, 'ID_POS_OPTIMIZER':2, 'ID_POS_BATCH_MODE':0, 'ID_POS_EPOCH':3}

cnn_configs={"width": 224, "height":224}
classes={"N":0, #normal (BACH dataset) 
         "B":1, #benign (BACH dataset) 
         "IS":2, #in situ carcinoma (BACH dataset)
         "IV":3, #invasive carcinoma, (BACH dataset)
         "A":4, #adenosis as benign (BreakHis dataset)
         "F":5, #fibroadenoma as benign (BreakHis dataset)
         "PT":6, #phyllodes tumor as benign (BreakHis dataset)
         "TA":7, #tubular adenona as benign (BreakHis dataset)
         "DC":8, #carcinoma as malignant (BreakHis dataset)
         "LC":9, #lobular carcinoma as malignant (BreakHis dataset)
         "MC":10, #mucinous carcinoma as malignant (BreakHis dataset)
         "PC":11 #papillary carcinoma as malignant (BreakHis dataset)
        }
scatter_resolution=0.25 #resolution of function meshgrid, default: 0.25
scatter_sleep_time=0.1  #animate sleep time, lower values increase animation speed, default: 0.1')

ieee_func_domain_ranges = { #represents the constrains of all benchmar functions used in the experimentation
    "CEC_1":     ([-100, 100], CEC_1, 'CEC_F1', 'CEC_1', 0),
    "CEC_2":     ([-100, 100], CEC_2, 'CEC_F2',  'CEC_2', 0),   
    "CEC_3":     ([-100, 100], CEC_3, 'CEC_F3',  'CEC_3', 0), 
    "CEC_4":     ([-100, 100],  CEC_4, 'CEC_F4',  'CEC_4', 0),  
    "CEC_5":     ([-100, 100],  CEC_5, 'CEC_F5',  'CEC_5', 0),  
    "CEC_6":     ([-100, 100],  CEC_6, 'CEC_F6',  'CEC_6', 0),  
    "CEC_7":     ([-100, 100],  CEC_7, 'CEC_F7',  'CEC_7', 0),  
    "CEC_8":     ([-100, 100],  CEC_8, 'CEC_F8',  'CEC_8', 0),  
    "CEC_9":     ([-100, 100],  CEC_9, 'CEC_F9',  'CEC_9', 0),  
    "CEC_10":     ([-100, 100],  CEC_10, 'CEC_F10',  'CEC_10', 0),  
    "CEC_11":     ([-100, 100],  CEC_11, 'CEC_F11',  'CEC_11', 0),  
    "CEC_12":     ([-100, 100],  CEC_12, 'CEC_F12',  'CEC_12', 0),  
    "CEC_13":     ([-100, 100],  CEC_13, 'CEC_F13',  'CEC_13', 0),  
    "CEC_14":     ([-100, 100],  CEC_14, 'CEC_F14',  'CEC_14', 0),  
    "C1":     ([-100, 100],  C1, 'C_F1',  'C1', 0),  
    "C2":     ([-100, 100],  C2, 'C_F2',  'C2', 0),  
    "C3":     ([-100, 100],  C3, 'C_F3',  'C3', 0),  
    "C4":     ([-100, 100],  C4, 'C_F4',  'C4', 0),  
    "C5":     ([-100, 100],  C5, 'C_F5',  'C5', 0),  
    "C6":     ([-100, 100],  C6, 'C_F6',  'C6', 0),  
    "C7":     ([-100, 100],  C7, 'C_F7',  'C7', 0),  
    "C8":     ([-100, 100],  C8, 'C_F8',  'C8', 0),  
    "C9":     ([-100, 100],  C9, 'C_F9',  'C9', 0),  
    "C10":     ([-100, 100], C10, 'C_F10', 'C10', 0),  
    "C11":     ([-100, 100],  C11, 'C_F11',  'C11', 0),  
    "C12":     ([-100, 100],  C12, 'C_F12',  'C12', 0),  
    "C13":     ([-100, 100],  C13, 'C_F13',  'C13', 0),  
    "C14":     ([-100, 100],  C14, 'C_F14',  'C14', 0),  
    "C15":     ([-100, 100],  C15, 'C_F15',  'C15', 0),  
    "C16":     ([-100, 100],  C16, 'C_F16',  'C16', 0),  
    "C17":     ([-100, 100],  C17, 'C_F17',  'C17', 0),  
    "C18":     ([-100, 100],  C18, 'C_F18',  'C18', 0),  
    "C19":     ([-100, 100],  C19, 'C_F19',  'C19', 0),  
    "C20":     ([-100, 100], C20, 'C_F20', 'C20', 0),
    "C21":     ([-100, 100],  C21, 'C_F21',  'C21', 0),  
    "C22":     ([-100, 100],  C22, 'C_F22',  'C22', 0),  
    "C23":     ([-100, 100],  C23, 'C_F23',  'C23', 0),  
    "C24":     ([-100, 100],  C24, 'C_F24',  'C24', 0),  
    "C25":     ([-100, 100],  C25, 'C_F25',  'C25', 0),  
    "C26":     ([-100, 100],  C26, 'C_F26',  'C26', 0),  
    "C27":     ([-100, 100],  C27, 'C_F27',  'C27', 0),  
    "C28":     ([-100, 100],  C28, 'C_F28',  'C28', 0),  
    "C29":     ([-100, 100],  C29, 'C_F29',  'C29', 0),  
    "C30":     ([-100, 100], C30, 'C_F30',   'C30', 0), 
    }

'''
L1/L2
Convo block
'''