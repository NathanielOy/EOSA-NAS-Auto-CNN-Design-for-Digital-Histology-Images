# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:05:41 2021

@author: 77132
"""

from models.multiple_solution.biology_based.EBOA import BaseEBOA
from utils.FunctionUtil import *
from utils.Ucomp import cmp_to_key, ucompare, ureverse
from utils.Settings import *
from time import time
from csv import reader
from utils.SavedResult import ProcessResult
from utils.MeasureUtil import MeasureClassification
from utils.animate_scatter import AnimateScatter
from eosa_nas.search_space import EOSANASSearchSpace
from eosa_nas.evaluation_strategy import EOSANASEvaluationStrategy
from eosa_nas.search_strategy import EOSANASSearchStrategy
from eosa_nas.cnn_builder import CNNAutoBuilder
import matplotlib.pyplot as plt
from keras.utils import plot_model
from utils.IOUtil import _save_results_to_csv__, _save_solutions_to_csv__, _save_epoch_results_to_csv__
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
import pickle
import ast

def compute_metrics():
    top1loss=[]
    top1acc=[]
    top1latency=[]
    
    top2loss=[]
    top2acc=[]
    top2latency=[]
    
    top3loss=[]
    top3acc=[]
    top3latency=[]
    
    top4loss=[]
    top4acc=[]
    top4latency=[]
    
    top5loss=[]
    top5acc=[]
    top5latency=[]
    
    for i in range(261):
        myfile='./dump/applied_experiement/epoch_'+str(i)+'_result_file.csv'
        data=read_metrics_data(myfile)
        loss, acc, latency=data[0]
        top1loss.append(loss)  
        top1acc.append(acc)
        top1latency.append(latency)
        
        loss, acc, latency=data[1]
        top2loss.append(loss)  
        top2acc.append(acc)
        top2latency.append(latency)
        
        loss, acc, latency=data[2]
        top3loss.append(loss)  
        top3acc.append(acc)
        top3latency.append(latency)
        
        loss, acc, latency=data[3]
        top4loss.append(loss)  
        top4acc.append(acc)
        top4latency.append(latency)
        
        loss, acc, latency=data[4]
        top5loss.append(loss)  
        top5acc.append(acc)
        top5latency.append(latency)
    
    tops=[(top1loss,top1acc,top1latency), (top2loss,top2acc,top2latency),
          (top3loss,top3acc,top3latency), (top4loss,top4acc,top4latency), (top5loss,top5acc,top5latency)]
    for i in range(5):
        loss, accuracy, latency=tops[i]
        measure=MeasureClassification()
        acc_metrics=measure._eosa_fit_accuarcy__(accuracy)
        measure=MeasureClassification()
        loss_metrics=measure._eosa_fit_loss__(loss)
        print('best   mean   median  worst   std  min-loss    max-loss   latency')
        print('Top '+str(i+1)+': '+str(acc_metrics)+'   '+str(loss_metrics)+'  '+str(min(latency)))
        metrics = {
                    'best-acc': acc_metrics[0],
                    'mean-acc':acc_metrics[1],
                    'median-acc':acc_metrics[2],
                    'worst-acc':acc_metrics[3],
                    'std-acc':acc_metrics[4],
                    'min-loss': loss_metrics[0],
                    'median-loss':loss_metrics[1],
                    'worst-loss':loss_metrics[2],
                    'latency': min(latency)
                    }
        _save_epoch_results_to_csv__(metrics, '_metrics', save_results_dir)
    
    return None

def read_metrics_data(myfile=None):#def get_data(myfile=None, algorithms=None, iterations=0, benchmarkfuncs=[]):
    data=[]
    interested_rows=[6, 9, 35, 24, 38]#[4,17,21,24,38]
    with open(myfile, 'r') as read_obj:
        csv_reader = reader(read_obj)
        index=0
        for row in csv_reader:            
            if index > 0 and index in interested_rows: #s
                #print(row)
                loss=row[1]               
                accuracy=row[2]
                latency=row[3]                
                loss=ast.literal_eval(loss)
                accuracy=ast.literal_eval(accuracy)
                latency=float(latency)
                #print(max(loss), max(accuracy), latency)
                data.append((min(loss), max(accuracy), latency))
            index+=1
    return data

def plot_results(hist, y_pred, y_test,avg_pred, epoch):
    N=epoch
    plt.style.use('seaborn-whitegrid')
    plt.title("Training/Validation Loss")
    plt.plot(np.arange(0, N),hist.history['loss'], label='training')
    plt.plot(np.arange(0, N),hist.history['val_loss'], label='validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch #")
    plt.legend(['training', 'validation'], loc="lower left")
    plt.show()
    
    plt.title("Training/Validation Accuracy")
    plt.plot(np.arange(0, N),hist.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, N),hist.history["val_accuracy"], label="validation_accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch #")
    plt.legend(['training', 'validation'], loc="lower left")
    plt.show()
    
    yhat_classes_2 = y_pred.argmax(axis=1) #https://www.geeksforgeeks.org/numpy-argmax-python/
    classes_2 = y_test.argmax(axis=1)
    
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
    precision2 = precision_score(classes_2, yhat_classes_2, average='macro')
    print('Precision macro: %f' % precision2)
    precision3 = precision_score(classes_2, yhat_classes_2, average='weighted')
    print('Precision weighted: %f' % precision3)
    precision4 = precision_score(classes_2, yhat_classes_2, average=None)
    print('Precision None: ' + str(precision4))
    # recall: tp / (tp + fn)
    recall = recall_score(classes_2, yhat_classes_2, average='micro')
    print('Recall micro: %f' % recall)
    recall2 = recall_score(classes_2, yhat_classes_2, average='macro')
    print('Recall macro: %f' % recall2)
    recall3 = recall_score(classes_2, yhat_classes_2, average='weighted')
    print('Recall weighted: %f' % recall3)
    recall4 = recall_score(classes_2, yhat_classes_2, average=None)
    print('Recall None: ' + str(recall4))
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(classes_2, yhat_classes_2, average='micro')
    print('F1 score micro: %f' % f1)
    f2 = f1_score(classes_2, yhat_classes_2, average='macro')
    print('F1 score macro: %f' % f2)
    f3 = f1_score(classes_2, yhat_classes_2, average='weighted')
    print('F1 score weighted: %f' % f3)
    f4 = f1_score(classes_2, yhat_classes_2, average=None)
    print('F1 score None: ' + str(f4))
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
    auc =0
    try:
        auc = roc_auc_score(classes_2, y_pred,  multi_class="ovr",average='macro')
        print('ROC AUC: %f' % auc)
        auc = roc_auc_score(classes_2, y_pred,  multi_class="ovr",average='weighted')
        print('ROC AUC: %f' % auc)
    except ValueError:
        pass
    
    solution = {
                    'epoch': epoch,
                    'loss_hist':hist.history['loss'],
                    'val_loss':hist.history['val_loss'],
                    'accuracy_hist':hist.history['accuracy'],
                    'val_accuracy':hist.history['val_accuracy'],
                    'cm': cm,
                    'accuracy':accuracy,
                    'precision': [precision, precision2, precision3, precision4],
                    'recall':[recall, recall2, recall3, recall4],
                    'f1': [f1, f2, f3, f4],
                    'kappa':kappa,
                    'CM': CM, 
                    'specificity': specificity, 
                    'sensitivity': sensitivity, 
                    'auc': auc,
                    'avg_pred':avg_pred,
                    }
    _save_epoch_results_to_csv__(solution, '_topbest_full_train', save_results_dir)

def get_data(myfile=None, N=0, batch_mode=1):
    history=[]
    with open(myfile, 'r') as read_obj:
        csv_reader = reader(read_obj)
        index=0
        for row in csv_reader:
            sols=[]                
            if index > 0:
                raw_network=row[21]
                openBraze = raw_network.find('[')
                closeBraze =raw_network.rfind(']')
                raw_network=raw_network[(openBraze+2):(closeBraze)]
                raw_network=raw_network.split('[')
                n=0
                convo_blocks=[]
                cl=0
                for rs in raw_network:
                    #if rs
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

           
if __name__ == "__main__":
    '''
    compute_metrics()
    '''
    search_paras={"blocks":blocks, "subblocks":subblocks, "problem_size":problem_size, "optimizers":optimizers, 
                  "activations":activations, "pooling_operations":pooling_operations, "regularizers":regularizers,
                  "factivations":factivations, "loss_functions":loss_functions, "learning_rates":learning_rates,
                  "classes":classes, "num_classes":num_classes, "cnn_epoch":cnn_epoch}
    ssp = EOSANASSearchSpace(serch_space_paras=search_paras, upper_lower_bounds=upper_lower_bounds, mode='init', bounds=bound_labels)
    solutions=[]#ssp._generate_solutions__()
    cb =CNNAutoBuilder(cnn_configs_paras=cnn_configs, solutions=solutions, num_classes=num_classes, pos_ids=pos_ids)
    model_configuration=[] #cb._build_architecture__()
    ss_paras={
              'model_config':model_configuration, 
              'log_mode':log_mode, 'train_result':train_result, 
              'test_result':test_result, 
              'input_dataset':input_dataset
             }
    sst=EOSANASSearchStrategy(search_strategy_paras=ss_paras, search_paras=search_paras, cnn_configs_paras=cnn_configs)
    #sst._show_summary__()
    
    top_best_file='./topbest.csv'
    epoch=60
    #solutions= get_data(top_best_file, epoch)
    config=0,0,0,1,0
    
    optimizer='RMSprop'#Top1: 0.05, 'RMSprop';  Top2: 0.005, 'Adagrad';  Top3: 0.05, 'RMSprop'; Top4:  0.005, 'Adagrad'; Top5:1e-05, 'Adam'
    learning_rate=0.05
    x_train, y_train, x_test, y_test, batch_size=sst._processing_train_input__(config=config)
    hist, y_pred, y_test,avg_pred=cb._top3_architecture__(x_train, y_train,x_test,y_test, optimizer,learning_rate,epoch)
    plot_results(hist, y_pred, y_test,avg_pred, epoch)
    