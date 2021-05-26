from models.multiple_solution.biology_based.EBOA import BaseEBOA
from utils.FunctionUtil import *
from utils.Ucomp import cmp_to_key, ucompare, ureverse
from utils.Settings import *
from time import time
from utils.SavedResult import ProcessResult
from utils.animate_scatter import AnimateScatter
from eosa_nas.search_space import EOSANASSearchSpace
from eosa_nas.evaluation_strategy import EOSANASEvaluationStrategy
from eosa_nas.search_strategy import EOSANASSearchStrategy
from eosa_nas.cnn_builder import CNNAutoBuilder


if __name__ == "__main__":
    search_paras={"blocks":blocks, "subblocks":subblocks, "problem_size":problem_size, "optimizers":optimizers, 
                  "activations":activations, "pooling_operations":pooling_operations, "regularizers":regularizers,
                  "factivations":factivations, "loss_functions":loss_functions, "learning_rates":learning_rates,
                  "classes":classes, "num_classes":num_classes, "cnn_epoch":cnn_epoch}
    ssp = EOSANASSearchSpace(serch_space_paras=search_paras, upper_lower_bounds=upper_lower_bounds, mode='init', bounds=bound_labels)
    solutions=ssp._generate_solutions__()
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
