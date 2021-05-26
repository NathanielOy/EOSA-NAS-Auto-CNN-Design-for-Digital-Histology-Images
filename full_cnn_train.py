# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:51:44 2021

@author: 77132
"""

from models.multiple_solution.biology_based.EBOA import BaseEBOA
from utils.FunctionUtil import *
from utils.Ucomp import cmp_to_key, ucompare, ureverse
from utils.Settings import *
from time import time
import matplotlib.pyplot as plt
from csv import reader
from utils.SavedResult import ProcessResult
from utils.animate_scatter import AnimateScatter
from eosa_nas.search_space import EOSANASSearchSpace
from eosa_nas.evaluation_strategy import EOSANASEvaluationStrategy
from eosa_nas.search_strategy import EOSANASSearchStrategy
from eosa_nas.cnn_builder import CNNAutoBuilder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score

def prediction(Y_pred_2, val_data_2):
    yhat_classes_2 = Y_pred_2.argmax(axis=1) #https://www.geeksforgeeks.org/numpy-argmax-python/
    classes_2=val_data_2.argmax(axis=1)
    print(Y_pred_2)
    print(Y_pred_2.shape)
    print(classes_2)
    print(str(len(classes_2))+'Confusion Matrix1'+str(yhat_classes_2.shape))
    print(confusion_matrix(classes_2, yhat_classes_2))
    cm = confusion_matrix(classes_2, yhat_classes_2)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('Confusion matrix ')
    plt.colorbar()
    plt.show()
    
    print('Classification Report')
    print(classification_report(classes_2, yhat_classes_2))
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(classes_2, yhat_classes_2)
    print('Accuracy: %f' % accuracy)
    
    # precision tp / (tp + fp)
    precision = precision_score(classes_2, yhat_classes_2, average='micro')
    print('Precision micro: %f' % precision)
    precision = precision_score(classes_2, yhat_classes_2, average='macro')
    print('Precision macro: %f' % precision)
    precision = precision_score(classes_2, yhat_classes_2, average='weighted')
    print('Precision weighted: %f' % precision)
    precision = precision_score(classes_2, yhat_classes_2, average=None)
    print('Precision None: ' + str(precision))
    
    
    # recall: tp / (tp + fn)
    recall = recall_score(classes_2, yhat_classes_2, average='micro')
    print('Recall micro: %f' % recall)
    recall = recall_score(classes_2, yhat_classes_2, average='macro')
    print('Recall macro: %f' % recall)
    recall = recall_score(classes_2, yhat_classes_2, average='weighted')
    print('Recall weighted: %f' % recall)
    recall = recall_score(classes_2, yhat_classes_2, average=None)
    print('Recall None: ' + str(recall))
    
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(classes_2, yhat_classes_2, average='micro')
    print('F1 score micro: %f' % f1)
    f1 = f1_score(classes_2, yhat_classes_2, average='macro')
    print('F1 score macro: %f' % f1)
    f1 = f1_score(classes_2, yhat_classes_2, average='weighted')
    print('F1 score weighted: %f' % f1)
    f1 = f1_score(classes_2, yhat_classes_2, average=None)
    print('F1 score None: ' + str(f1))
    
    # kappa
    kappa = cohen_kappa_score(classes_2, yhat_classes_2)
    print('Cohens kappa: %f' % kappa)
    
    
    CM = confusion_matrix(classes_2, yhat_classes_2)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    specificity = TN / (TN+FP)
    sensitivity  = TP / (TP+FN)
    print('specificity3: %f' % specificity)
    print('sensitivity3: %f' % sensitivity)
    
    # ROC AUC
    try:
        auc = roc_auc_score(classes_2, Y_pred_2,  multi_class="ovr",average='macro')
        print('ROC AUC: %f' % auc)
        auc = roc_auc_score(classes_2, Y_pred_2,  multi_class="ovr",average='weighted')
        print('ROC AUC: %f' % auc)
    except ValueError:
        pass
def plot(hist, N):
    plt.style.use('seaborn-white')
    plt.title("Training/Validation Loss on Dataset ")
    plt.plot(np.arange(0, N),hist['loss'], label='training')
    plt.plot(np.arange(0, N),hist['val_loss'], label='validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch #")
    plt.legend(['training', 'validation'], loc="lower left")
    plt.show()
    
def get_data(myfile=None, N=0, batch_mode=1):
    history=[]
    with open(myfile, 'r') as read_obj:
        csv_reader = reader(read_obj)
        index=0
        for row in csv_reader:
            sols=[]                
            if index > 1:
                raw_network=row[21]
                #print(raw_network)
                openBraze = raw_network.find('[')
                closeBraze =raw_network.rfind(']')
                raw_network=raw_network[(openBraze+2):(closeBraze)]
                raw_network=raw_network.split('[')
                n=0
                convo_blocks=[]
                cl=0
                for rs in raw_network:
                    #if rs
                    #print(row)
                    row=rs.split(' ')
                    rw=[]
                    for i in row:
                        if i !='':                            
                            item=i.replace('\n', ' ')
                            rw.append(i)
                    row=rw        
                    if n==0:  # num_convo_blk                
                        item=row[0]
                        gl=int(float(item.strip()))
                        item=row[1]
                        go=int(float(item.strip()))
                        item=row[2]
                        z=int(float(item.strip()))
                        z=True if z==1 else False
                        gb=[batch_mode, gl, go, N]
                        item=row[3]
                        fl=int(float(item.strip()))
                        item=row[4]
                        faf=int(float(item.strip()))
                        faf=factivations[faf]
                        item=row[5]
                        fd=float(item.strip())
                        item=row[6]
                        fr=int(float(item.strip()))
                        fr=regularizers[fr]
                        fcb=(fl,[faf, fd, fr])
                        item=row[7]                        
                        if item.find(']') > 0:
                            cl=int(float(item.replace(']', '').strip()))
                        else:
                            cl=int(float(item.strip()))
                        n=n+1
                    else:
                        item=row[1]  #item=row[0] is the isAlive value
                        cc=int(float(item.strip()))
                        item=row[2]
                        caf=int(float(item.strip()))
                        caf=activations[caf]
                        item=row[3]
                        ck=int(float(item.strip()))
                        item=row[4]
                        cf=int(float(item.strip()))
                        item=row[5]
                        cps=int(float(item.strip()))
                        item=row[6]
                        cpt=int(float(item.strip()))
                        cpt=pooling_operations[cpt]
                        item=row[7]
                        if item.find(']') > 0:
                            cr=int(float(item.replace(']', '').strip()))
                        else:
                            cr=int(float(item.strip()))
                        cr=regularizers[cr]
                        blk=[cc, caf, ck, cf, cps, cpt, cr]
                        convo_blocks.append(blk)    
                        
                cb=(cl, convo_blocks)
                lfb=loss_functions[0]
                s=gb, z, cb, fcb, lfb
                print(s)
                sols.append(s)
                break
            index+=1
            
    return sols

search_paras={"blocks":blocks, "subblocks":subblocks, "problem_size":problem_size, "optimizers":optimizers, 
                  "activations":activations, "pooling_operations":pooling_operations, "regularizers":regularizers,
                  "factivations":factivations, "loss_functions":loss_functions, "learning_rates":learning_rates,
                  "classes":classes, "num_classes":num_classes, "cnn_epoch":cnn_epoch}

solutions=[]
top_best_file='./dump/23epochs/epoch_11_result_file.csv'
N=3
batch_mode=1
solutions= get_data(top_best_file, N)
cb =CNNAutoBuilder(cnn_configs_paras=cnn_configs, solutions=solutions, num_classes=num_classes, pos_ids=pos_ids)
model_configuration=cb._build_architecture__()
ss_paras={
              'model_config':model_configuration, 
              'log_mode':log_mode, 'train_result':train_result, 
              'test_result':test_result, 
              'input_dataset':input_dataset
             }
sst=EOSANASSearchStrategy(search_strategy_paras=ss_paras, search_paras=search_paras, cnn_configs_paras=cnn_configs)
sst._show_summary__()
evaluated_solutions=sst._training_all__()

eval_strategy_paras={'problem_size':problem_size, 'objective_func':objective_func}
#evs=EOSANASEvaluationStrategy(eval_strategy_paras=eval_strategy_paras, evaluated_solutions=evaluated_solutions, pos_ids=pos_ids, search_paras=search_paras)
#solutions=evs._evaluate_solution__()    
eboa_paras = {
            "eosa_epoch": eosa_epoch,
            "problem_size": problem_size
        }
md = BaseEBOA(root_algo_paras=eval_strategy_paras, evaluated_solutions=evaluated_solutions, 
                  eboa_paras=eboa_paras, model_rates=modelrates, pos_ids=pos_ids, 
                  ssp=ssp, search_paras=search_paras, cnn_configs=cnn_configs, bounds=bound_labels)
sols=md._train__()
params={"save_results_dir":save_results_dir, "eosa_epoch": eosa_epoch, 'problem_size':problem_size}
pr=ProcessResult(params=params)
pr._save_results__(solutions=md.sols_config_and_perfromances, epoch_record=False)

'''
hist=sst.history
n=0
for cnn in evaluated_solutions:
    raw_sol, loss_train, accuracy_train, time_total_train, time_predict, avg_pred, y_test, y_pred=cnn
    plot(hist[n], N)
    prediction(y_pred, y_test)
    n=n+1
'''

'''
optimizer='Adam'
learning_rate=0.0001
config=0, 0, 0, 1, 0
x_train, y_train, x_test, y_test, batch_size=sst._processing_train_input__(config=config)
hist, y_pred, y_test,avg_pred=cb._manual_architecture__(x_train, y_train,x_test,y_test, optimizer,learning_rate)
plot(hist, N)
prediction(y_pred, y_test)
'''