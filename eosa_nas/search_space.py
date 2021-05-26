# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:33:46 2021

@author: Oyelade
"""

import numpy as np
import random
from copy import deepcopy


class EOSANASSearchSpace(object):
    
    def __init__(self, serch_space_paras=None,upper_lower_bounds=None, mode='init', bounds=None):
        self.blocks = serch_space_paras["blocks"]
        self.subblocks = serch_space_paras["subblocks"]
        self.problem_size = serch_space_paras["problem_size"]
        self.optimizers = serch_space_paras["optimizers"]
        self.activations=serch_space_paras["activations"]
        self.learning_rates=serch_space_paras["learning_rates"]
        self.pooling_operations=serch_space_paras["pooling_operations"]
        self.regularizers=serch_space_paras["regularizers"]
        self.factivations=serch_space_paras["factivations"]
        self.loss_functions=serch_space_paras["loss_functions"]
        self.cnn_epoch=serch_space_paras["cnn_epoch"]        
        self.upper_lower_bounds=upper_lower_bounds
        self.solutions = []
        self.mode=mode
        self.a_scatter=None
        self.gl, self.go=bounds['gl'], bounds['go']
        self.lfb, self.fl, self.faf, self.fd=bounds['lfb'], bounds['fl'], bounds['faf'], bounds['fd']
        self.fr, self.cc, self.caf, self.ck=bounds['fr'], bounds['cc'], bounds['caf'], bounds['ck']
        self.cf, self.cps, self.cpt, self.cr=bounds['cf'], bounds['cps'], bounds['cpt'], bounds['cr']
        
    def _generate_general_block__(self):
        #batch size mode 
        gbn=random.randint(self.upper_lower_bounds['GB_MIN'],   self.upper_lower_bounds['GB_MAX'])
        gb=(2**gbn) - 1
        #learning rates 
        one_or_five=[1, 5]
        one_or_five_index=random.randint(0,   1)
        gln=random.randint(self.upper_lower_bounds['GL_MIN'],   self.upper_lower_bounds['GL_MAX'])
        #gl=one_or_five[one_or_five_index] * (10 ** -gln)
        gl=self.learning_rates[gln]
        #optimizer
        gon=random.randint(self.upper_lower_bounds['GO_MIN'],   self.upper_lower_bounds['GO_MAX'])
        go=self.optimizers[gon]
        #nu,ber of epoch
        gen=random.randint(self.upper_lower_bounds['GE_MIN'],   self.upper_lower_bounds['GE_MAX'])
        ge=self.cnn_epoch        
        return [gb, gl, go, ge]
    
    def _generate_inputzeropad_block__(self):
        zeropad_or_not=random.randint(0,   1)
        return True #if zeropad_or_not else False
    
    def _generate_convo_block__(self):
        #Number of convolutional blocks
        cln=random.randint(self.upper_lower_bounds['CL_MIN'],   self.upper_lower_bounds['CL_MAX'])
        cl=1+cln #(2*cln) - 1
        convo_blocks=[]
        ckn=0
        for i in range(cl):
            ckn, cc, caf, ck, cf, cps, cpt, cr=self._generate_sub_convo_block__(ckn, i)
            blk=[cc, caf, ck, cf, cps, cpt, cr]
            convo_blocks.append(blk)
        return (cl, convo_blocks)
    
    def _generate_sub_convo_block__(self, ckn, i):
        #Number of convolutional operation per block
        ccn=random.randint(self.upper_lower_bounds['CC_MIN'],   self.upper_lower_bounds['CC_MAX'])
        cc=3 - ccn
        #Select activation function
        cafn=random.randint(self.upper_lower_bounds['CAF_MIN'],   self.upper_lower_bounds['CAF_MAX'])
        caf=self.activations[cafn]
        #Number of kernel
        if i==0:
            limit_max=5 #self.upper_lower_bounds['CK_MAX']
            ckn=random.randint(self.upper_lower_bounds['CK_MIN'], limit_max)
        else:
            ckn=ckn+1
        ck=(2**ckn)
        #filter size
        cfn=1
        while cfn%2 != 0:
            cfn=random.randint(self.upper_lower_bounds['CF_MIN'], self.upper_lower_bounds['CF_MAX']) 
        cf=2+(cfn-1)
        
        #Pooling size
        cpsn=1
        while cpsn!=0 or cpsn%2 != 0:
            cpsn=random.randint(self.upper_lower_bounds['CPS_MIN'],  self.upper_lower_bounds['CPS_MAX'])
        cps=2+cpsn
        #Pooling operation type
        cptn=random.randint(self.upper_lower_bounds['CPT_MIN'],  self.upper_lower_bounds['CPT_MAX'])
        cpt=self.pooling_operations[cptn]
        crn=random.randint(self.upper_lower_bounds['CR_MIN'],   self.upper_lower_bounds['CR_MAX'])
        cr=self.regularizers[crn]
        return ckn, cc, caf, ck, cf, cps, cpt, cr
    
    def _augmented_sub_convo_block__(self, ckn):
        ckn, cc, caf, ck, cf, cps, cpt, cr=self._generate_sub_convo_block__(ckn, 1)
        isalive=1
        caf=list(self.activations.keys())[list(self.activations.values()).index(caf)]
        cpt=list(self.pooling_operations.keys())[list(self.pooling_operations.values()).index(cpt)]
        cr=list(self.regularizers.keys())[list(self.regularizers.values()).index(cr)]
        return ckn, [isalive, cc, caf, ck, cf, cps, cpt, cr]
    
    def _generate_fullyconnected_block__(self):
        #Number of fully connected layers
        fln=random.randint(self.upper_lower_bounds['FL_MIN'],   self.upper_lower_bounds['FL_MAX'])
        fl=1+fln
        #activation functions for fully connected layers
        fafn=random.randint(self.upper_lower_bounds['FAF_MIN'],  self.upper_lower_bounds['FAF_MAX'])
        faf=self.factivations[fafn]
        #Dropout rate for fully connected layers
        fdn=round(random.uniform(self.upper_lower_bounds['FD_MIN'], self.upper_lower_bounds['FD_MAX']), 1)
        #fdn=random.randint(self.upper_lower_bounds['FD_MIN'], self.upper_lower_bounds['FD_MAX'])
        fd=1/fdn
        #Regularizers for fully connected layers
        frn=random.randint(self.upper_lower_bounds['FR_MIN'], self.upper_lower_bounds['FR_MAX'])
        fr=self.regularizers[frn]
        return (fl,[faf, fd, fr])
    
    def _generate_lossfunction_block__(self):
        #Regularizers for fully connected layers
        lfln=random.randint(self.upper_lower_bounds['LFL_MIN'], self.upper_lower_bounds['LFL_MAX'])
        lfl=self.loss_functions[lfln]
        return lfl
    
    def _create_solution__(self, minmax=1):         
        gb=self._generate_general_block__()
        iz=self._generate_inputzeropad_block__()
        cb=self._generate_convo_block__()
        fcb=self._generate_fullyconnected_block__()
        lfb=self._generate_lossfunction_block__()        
        solution=(gb, iz, cb, fcb, lfb)
        return solution    
    
    def _generate_solutions__(self):
        for i in range(self.problem_size):
            self.solutions.append(self._create_solution__(minmax=1))
        return self.solutions
    
    def _regen_convolayers__(self):
        return 1
    
    def _check_fd_within_values__(self, optimize_index, params):
        lb, ub=self._get_upper_lower_bounds__(params)
        if not (optimize_index <= 1/ub and optimize_index >=1/lb):             
            fdn=round(random.uniform(lb, ub), 1)
            optimize_index=1/fdn
        return optimize_index
    
    def _check_within_bounds__(self, optimize_index, params):        
        lb, ub=self._get_upper_lower_bounds__(params)
        if not (optimize_index <= ub and optimize_index >=lb):            
            optimize_index=random.randint(lb, ub) #ensure optimize_index sits within the upper (ub) and lower (lb) bounds           
        return optimize_index
    
    def _get_upper_lower_bounds__(self, params=None):
        if params==self.lfb: 
            return self.upper_lower_bounds['LFL_MIN'], self.upper_lower_bounds['LFL_MAX']
        elif params==self.fl:
            return self.upper_lower_bounds['FL_MIN'],   self.upper_lower_bounds['FL_MAX']
        elif params==self.faf:
            return self.upper_lower_bounds['FAF_MIN'],  self.upper_lower_bounds['FAF_MAX']
        elif params==self.fd:
            return self.upper_lower_bounds['FD_MIN'], self.upper_lower_bounds['FD_MAX']
        elif params==self.fr:
            return self.upper_lower_bounds['FR_MIN'], self.upper_lower_bounds['FR_MAX']
        elif params==self.cc:
            return self.upper_lower_bounds['CC_MIN'],   self.upper_lower_bounds['CC_MAX']
        elif params==self.caf:
            return self.upper_lower_bounds['CAF_MIN'],   self.upper_lower_bounds['CAF_MAX']
        elif params==self.ck:
            return self.upper_lower_bounds['CK_MIN'], self.upper_lower_bounds['CK_MAX']
        elif params==self.cf:
            return self.upper_lower_bounds['CF_MIN'], self.upper_lower_bounds['CF_MAX']
        elif params==self.cps:
            return self.upper_lower_bounds['CPS_MIN'],  self.upper_lower_bounds['CPS_MAX']
        elif params==self.cpt:
            return self.upper_lower_bounds['CPT_MIN'],  self.upper_lower_bounds['CPT_MAX']
        elif params==self.cr:
            return self.upper_lower_bounds['CR_MIN'],   self.upper_lower_bounds['CR_MAX']
        elif params==self.gl:
            return self.upper_lower_bounds['GL_MIN'],   self.upper_lower_bounds['GL_MAX']
        elif params==self.go:
            return self.upper_lower_bounds['GO_MIN'],   self.upper_lower_bounds['GO_MAX']