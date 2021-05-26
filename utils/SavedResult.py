# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:42:58 2021

@author: Oyelade
"""
from utils.MeasureUtil import MeasureClassification
from utils.IOUtil import _save_results_to_csv__, _save_solutions_to_csv__, _save_epoch_results_to_csv__ 
from utils.GraphUtil import _draw_rates__
from utils.paper12plotpropagation import _draw_propagation_rates__
import numpy as np


class ProcessResult:
	
    def __init__(self, params=None):
        self.pop_size = params["problem_size"]
        self.epoch = params["eosa_epoch"]
        self.log_filename='result_file'
        self.path_save_result=params["save_results_dir"]
        
    def _save_results__(self, solutions=None, epoch_record=False):
        for sols_config in solutions:
            sol, cnn_sol=sols_config
            raw_solutions, loss_train, accuracy_train, val_loss_train, val_accuracy_train, time_total_train, time_predict, avg_pred, y_true, y_pred=cnn_sol
            y_true=np.argmax(y_true, axis=1)
            y_pred=np.argmax(y_pred, axis=1)
            _, formated_fits=sol
            formalized_sol=formated_fits[0]
            fitness=formated_fits[1]
            measure = MeasureClassification(y_true=y_true, y_pred=y_pred, number_rounding=4)
            measure._fit__()
            solution = {
                    'model_name': str(sols_config), 
                    'loss_train':loss_train,
                    'accuracy_train': accuracy_train, 
                    'time_total_train': time_total_train, 
                    'time_predict': time_predict, 
                    'score_confusion_matrix': measure.score_confusion_matrix,
                    'score_classification_report': measure.score_classification_report,
                    'score_accuracy':measure.score_accuracy,
                    'score_precision':measure.score_precision,
                    'score_recall': measure.score_recall,
                    'score_f1':measure.score_f1,
                    'cohen_kappa': measure.cohen_kappa,
                    'sensitivity':measure.sensitivity,
                    'specificity': measure.specificity,
                    'score_matthews_corrcoef':measure.score_matthews_corrcoef,
                    'score_roc_auc': measure.score_roc_auc,
                    'score_top_k_accuracy':measure.score_top_k_accuracy,
                    'score_multilabel_confusion_matrix': measure.score_multilabel_confusion_matrix,
                    'score_jaccard_score':measure.score_jaccard_score,
                    'avg_pred': avg_pred, 
                    'fitness': fitness, 
                    'formalized_sol': formalized_sol, 
                    'raw_solutions': raw_solutions,
                    }
            if epoch_record:
                _save_epoch_results_to_csv__(solution, self.log_filename, self.path_save_result)
            else:
                _save_results_to_csv__(solution, self.log_filename, self.path_save_result)
        print('Completed EOSA-NAS run ')