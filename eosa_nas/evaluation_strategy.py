# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:59:20 2021

@author: Oyelade
"""
import numpy as np
from copy import deepcopy
from utils.Ucomp import cmp_to_key, ucompare, ureverse
import random
from eosa_nas.search_space import EOSANASSearchSpace
from eosa_nas.search_strategy import EOSANASSearchStrategy
from eosa_nas.cnn_builder import CNNAutoBuilder
from utils.Settings import *
import math

class EOSANASEvaluationStrategy(object):
    """ This is root of all Algorithms """
    ID_MIN_PROBLEM = 0
    ID_MAX_PROBLEM = -1

    def __init__(self, eval_strategy_paras = None, evaluated_solutions=None, pos_ids=None, search_paras=None, cnn_configs=None, num_classes=None, bounds=None):
        self.problem_size = eval_strategy_paras["problem_size"]
        self.num_classes=search_paras["num_classes"]
        self.classes=search_paras["classes"]
        #self.domain_range = eval_strategy_paras["domain_range"]
        #self.print_train = eval_strategy_paras["print_train"]
        self.cnn_configs=cnn_configs
        self.objective_func = eval_strategy_paras["objective_func"]
        self.solution, self.fit = None, []
        self.search_paras=search_paras
        self.evaluated_solutions=evaluated_solutions
        self.pos_ids=pos_ids
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
        self.solutions_meta_data=[]
        self.sols_config_and_perfromances=[]
        self.gl, self.go=bounds['gl'], bounds['go']
        self.lfb, self.fl, self.faf, self.fd=bounds['lfb'], bounds['fl'], bounds['faf'], bounds['fd']
        self.fr, self.cc, self.caf, self.ck=bounds['fr'], bounds['cc'], bounds['caf'], bounds['ck']
        self.cf, self.cps, self.cpt, self.cr=bounds['cf'], bounds['cps'], bounds['cpt'], bounds['cr']
                
    def _formalize__(self, raw_solutions=None): 
        gbi, iz, cb, fcb, lfb=raw_solutions
        gb=gbi[self.ID_POS_BATCH_MODE]
        gl= list(self.search_paras['learning_rates'].keys())[list(self.search_paras['learning_rates'].values()).index(gbi[self.ID_POS_LEARNING_RATE])] 
        go=list(self.search_paras['optimizers'].keys())[list(self.search_paras['optimizers'].values()).index(gbi[self.ID_POS_OPTIMIZER])]
        ge=gbi[self.ID_POS_EPOCH]
        
        fl, ff=fcb
        faf=list(self.search_paras['factivations'].keys())[list(self.search_paras['factivations'].values()).index(ff[self.ID_FULLY_DENSE_ACTIVATION_FUNC])]
        fd=ff[self.ID_FULLY_DENSE_DROPOUT] 
        fr=list(self.search_paras['regularizers'].keys())[list(self.search_paras['regularizers'].values()).index(ff[self.ID_FULLY_DENSE_REGULARIZER])]
        
        z=1 if iz else 0
        l=list(self.search_paras['loss_functions'].keys())[list(self.search_paras['loss_functions'].values()).index(lfb)]
        
        #Our formalism wil assume a 8xn shape of np array
        np_form=[]
        #row-1 (minus 1) will keep info the EOSA algorithm can't mutate e.g [gb, ge, l, current_num_convo_blk]
        num_convo_blk, convo_blks=cb
        row_minus_1=[gb, ge, l, num_convo_blk] #hence, we will not pass row_minus_1 into solution structure
        
        #row0, will hold: LR, optimizer, zeropad, fl, faf, fd, fr, num_convo_blk        
        row0=np.array([gl, go, z, fl, faf, fd, fr, num_convo_blk])
        np_form.append(row0)
        
        for cv in convo_blks:
            cc=cv[self.ID_CONVO]
            caf=list(self.search_paras['activations'].keys())[list(self.search_paras['activations'].values()).index(cv[self.ID_ACTIVATION_FUNC])]
            ck=cv[self.ID_NUMBER_OF_KERNEL]
            cf=cv[self.ID_FILTER_SIZE]
            cps=cv[self.ID_POOL_SIZE]
            cpt=list(self.search_paras['pooling_operations'].keys())[list(self.search_paras['pooling_operations'].values()).index(cv[self.ID_POOL_TYPE])]
            cr=list(self.search_paras['regularizers'].keys())[list(self.search_paras['regularizers'].values()).index(cv[self.ID_REGULARIZER])]
            #row1-row_n will hold: [is_alive (e.g use to drop block if num_convo_blk reduces), cc, caf, ck, cf, cps, cpt, cr] per-block
            is_alive=1
            row1_row_n=np.array([is_alive, cc, caf, ck, cf, cps, cpt, cr]) 
            np_form.append(row1_row_n)
        
        np_form=np.array(np_form)
        #print('Formalized: '+str(np_form))
        return row_minus_1, np_form
    
    def _unformalize__(self, ssp=None, raw_solutions=None): 
        row_minus_1, row0_n=raw_solutions
        #row-1 (minus 1) will keep info the EOSA algorithm can't mutate e.g [gb, ge, l]
        gb=row_minus_1[0]
        ge=row_minus_1[1]
        lossfunc=self.search_paras['loss_functions']
        lfb=lossfunc[row_minus_1[2]]
        previous_num_convo_blk=row_minus_1[3]
        
        #Remove all traces of -negative values from our np array, since CNN builder does not expect any params to be -ve
        row0_n=np.abs(row0_n)
        
        #row0, will hold: LR, optimizer, zeropad, faf, fd, fr, num_convo_blk
        row0=row0_n[0]
        gbllist=self.search_paras['learning_rates']
        gl=gbllist[ssp._check_within_bounds__(int(row0[0]), self.gl)]
        gbolist=self.search_paras['optimizers']
        go=gbolist[ssp._check_within_bounds__(int(row0[1]), self.go)]
        gbi=[gb, gl, go, ge]
        iz=True #if row0[2]==1 else False
                
        fl=int(row0[3]) if  int(row0[3])>=1 else 1
        faflist=self.search_paras['factivations']
        faf=faflist[ssp._check_within_bounds__(int(row0[4]), self.faf)]
        fd=ssp._check_fd_within_values__(row0[5], self.fd)
        frlist=self.search_paras['regularizers']
        fr=frlist[ssp._check_within_bounds__(int(row0[6]), self.fr)]
        fcb=(fl, [faf, fd, fr])
        
        new_num_convo_blk=ssp._regen_convolayers__() if math.ceil(int(row0[7])) < 1 else math.ceil(int(row0[7]))
        tmp=[]
        if previous_num_convo_blk < new_num_convo_blk: #add new convo-blk to CNN config
            diff=new_num_convo_blk-previous_num_convo_blk            
            for i in range(diff):
                ckn=previous_num_convo_blk+i
                #generate new blocks to suffice for the increment
                ckn, convo_blk = ssp._augmented_sub_convo_block__(ckn)
                row0_n = np.vstack([row0_n, np.array(convo_blk)])
                tmp=row0_n
                
        if previous_num_convo_blk > new_num_convo_blk: #eliminate some convo-blk from CNN config
            #select  blocks to eliminate for balance
            diff=previous_num_convo_blk-new_num_convo_blk
            #We allow the last no-of-diff blocks to be eliminated in the for-loop below
        
        print('SEEEEEEEEF '+' >>> new_num_convo_blk='+str(new_num_convo_blk)+'    metadata='+str(row_minus_1)+'  >>>  solution='+str(row0_n)+'  >>>  aug_solution='+str(tmp))
        valid_blks=0
        cblks=[]
        for cv in range(1, new_num_convo_blk): #we start from 1 becose row0 is already taken above
            #row1-row_n will hold: [is_alive (e.g use to drop block if num_convo_blk reduces), cc, caf, ck, cf, cps, cpt, cr] per-block
            n_row=row0_n[cv] #avoid index 0 since it rep row0 above
            #is_alive=n_row[0] # we can use this to eliminate convo_blks if num_convo_blk reduces due to EOSA 
            cc=math.ceil(int(float(n_row[1])))
            if (cc <=0 and new_num_convo_blk==1) or (cc <=0 and new_num_convo_blk >1 and cv==(new_num_convo_blk-1)):
                cc=1 #if we have only 1 convo blk and it is optimized to 0, then retain it (1) and not remove (0)
            print(' old c='+str(math.ceil(int(float(n_row[1]))))+'  new c='+str(cc))
            caflist=self.search_paras['activations']
            caf=caflist[ssp._check_within_bounds__(int(float(n_row[2])), self.caf)]
            cktmp=math.ceil(int(float(n_row[3])))
            cktmp= cktmp if cktmp%2==0 else cktmp+1
            ck=2**int(self._get2power__(cktmp)) #takes nearest square of 2
            #print(str(cktmp)+'  takes near    '+str(math.sqrt(cktmp))+'   est square of '+str(ck))
            cf=int(float(n_row[4])) if int(float(n_row[4]))>=1 and int(float(n_row[4]))%2!=0 else 3
            cps=int(float(n_row[5])) if int(float(n_row[5])) >=2 and int(float(n_row[5]))%2 != 1 else 2
            cptlist=self.search_paras['pooling_operations']
            cpt=cptlist[ssp._check_within_bounds__(int(float(n_row[6])), self.cpt)]
            crlist=self.search_paras['regularizers']
            cr=crlist[ssp._check_within_bounds__(int(float(n_row[7])), self.cr)]
            #row1-to-row_n
            if cc>0: #add only optimized convo blks that do not have convo operation to be 0
                cblks.append([cc, caf, ck, cf, cps, cpt, cr])
                valid_blks=valid_blks+1
        cb=(valid_blks, cblks)
                
        raw_solutions_4_cnn_design=gbi, iz, cb, fcb, lfb
        print('Unformalized for CNN build: '+str(raw_solutions_4_cnn_design))
        
        return raw_solutions_4_cnn_design
    
    def _get2power__(self, val): #to determine value n so that 2**n=val
        n=1
        while True:
            tmp=2**n
            if tmp == val:
                break
            if tmp > val:
                n=n-1
                break
            n=n+1
        return n
    def _create_initial_timelines__(self):
        timelines =[np.random.rand() for _ in range(self.pop_size)]
        return timelines
        
    def _evaluate_solution__(self, solution=None, minmax=1): 
        i=0
        for cnn_sol in self.evaluated_solutions:
            raw_solutions, loss_train, accuracy_train, val_loss_train, val_accuracy_train, time_total_train, time_predict, avg_pred, y_test, y_pred=cnn_sol
            fitness = self._fitness_model__(loss_train=loss_train, accuracy_train=accuracy_train, time_total_train=time_total_train, minmax=minmax)
            meta_data, formalized_sol=self._formalize__(raw_solutions)
            self.solutions_meta_data.append((i, meta_data))         
            sol=i, [formalized_sol, fitness]
            #print('Potential sol '+str(sol))
            self.fit.append(sol) #cnn_sol            
            self.sols_config_and_perfromances.append((sol, cnn_sol))
            i=i+1
        return self.fit
    
    def _create_solution__(self, meta_data_index=None, ssp=None, minmax=0):
        solution = ssp._create_solution__() 
        cb =CNNAutoBuilder(cnn_configs_paras=self.cnn_configs, solutions=[solution], num_classes=self.num_classes, pos_ids=self.pos_ids)
        model_configuration=cb._build_architecture__()
        ss_paras={'model_config':model_configuration, 'log_mode':log_mode, 'train_result':train_result, 'test_result':test_result, 'input_dataset':input_dataset}
        sst=EOSANASSearchStrategy(search_strategy_paras=ss_paras, search_paras=self.search_paras, cnn_configs_paras=self.cnn_configs)
        cnn_sol=sst._training_all__()
        raw_solutions, loss_train, accuracy_train, val_loss_train, val_accuracy_train, time_total_train, time_predict, avg_pred, y_test, y_pred=cnn_sol[0]
        fitness = self._fitness_model__(loss_train=loss_train, accuracy_train=accuracy_train, time_total_train=time_total_train, minmax=minmax)
        #update the sol_config, fit and perfromance of the solution we are re-evaluating
        for sols_config in self.sols_config_and_perfromances:
            sol, _=sols_config
            i, _=sol
            if i == meta_data_index:
                sol=i, [solution, fitness]
                self.sols_config_and_perfromances[i]=sol, cnn_sol[0]
        return [solution, fitness]
         
    def _reevaluate_solution__(self, meta_data_index=None, ssp=None, solution=None, minmax=1): 
        for mt in self.solutions_meta_data:
            indx, mt_data=mt
            if indx == meta_data_index:
                break
        sol_formalized=mt_data, solution
        sol_formalized=self._unformalize__(ssp, sol_formalized)
        cb =CNNAutoBuilder(cnn_configs_paras=cnn_configs, solutions=[sol_formalized], num_classes=self.num_classes, pos_ids=self.pos_ids)
        model_configuration=cb._build_architecture__()
        ss_paras={'model_config':model_configuration, 'log_mode':log_mode, 'train_result':train_result, 'test_result':test_result, 'input_dataset':input_dataset}
        sst=EOSANASSearchStrategy(search_strategy_paras=ss_paras, search_paras=self.search_paras, cnn_configs_paras=self.cnn_configs)
        cnn_sol=sst._training_all__()
        raw_solutions, loss_train, accuracy_train, val_loss_train, val_accuracy_train, time_total_train, time_predict, avg_pred, y_test, y_pred=cnn_sol[0]
        fitness = self._fitness_model__(loss_train=loss_train, accuracy_train=accuracy_train, time_total_train=time_total_train, minmax=minmax)
        #self.fit.append([raw_solutions, fitness]) #cnn_sol
        #update the sol_config, fit and perfromance of the solution we are re-evaluating
        for sols_config in self.sols_config_and_perfromances:
            sol, _=sols_config
            i, _=sol
            if i == meta_data_index:
                sol=i, [solution, fitness]
                self.sols_config_and_perfromances[i]=sol, cnn_sol[0]
        return fitness
            
    def _fitness_model__(self, loss_train=None, accuracy_train=None, time_total_train=None, minmax=0):
        """ Assumption that objective function always return the original value """
        return self.objective_func(loss_train=loss_train, accuracy_train=accuracy_train, time_total_train=time_total_train) if minmax == 0 \
            else 1.0 / self.objective_func(loss_train=loss_train, accuracy_train=accuracy_train, time_total_train=time_total_train)

    def _fitness_encoded__(self, encoded=None, id_pos=None, minmax=0):
        return self._fitness_model__(solution=encoded[id_pos], minmax=minmax)

    def _get_global_best__(self, pop=None, id_fitness=None, id_best=None):
        #sorted_pop = sorted(pop, key=cmp_to_key(ucompare, id_fitness))
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fitness])
        return deepcopy(sorted_pop[id_best])

    def _get_global_worst__(self, pop=None, id_fitness=None, id_worst=None):
        #sorted_pop = sorted(pop, key=lambda temp: temp[id_fitness])
        sorted_pop = sorted(pop, key=cmp_to_key(ucompare, id_fitness))
        return deepcopy(sorted_pop[id_worst])

    def _amend_solution_and_return__(self, solution=None):
        for i in range(self.problem_size):
            if solution[i] < self.domain_range[0]:
                solution[i] = self.domain_range[0]
            if solution[i] > self.domain_range[1]:
                solution[i] = self.domain_range[1]
        return solution
    
    def _create_opposition_solution__(self, solution=None, g_best=None):
        temp = [self.domain_range[0] + self.domain_range[1] - g_best[i] + np.random.random() * (g_best[i] - solution[i])
                      for i in range(self.problem_size)]
        return np.array(temp)


    def _train__(self):
        pass