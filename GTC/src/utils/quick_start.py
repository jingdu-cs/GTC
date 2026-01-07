from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os
import torch
from torch.nn import DataParallel


def quick_start(model, dataset, config_dict, save_model=True, mg=False, save_dir='./'):
    # merge config dict
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))
    # File for storing test/validation results
    config['save_dir'] = "/scratch/" + str(config['dataset']) + '/' + str(model) + '/' + 'diff_' + str(config['diff_weight']) + '_symile_' + str(config['symile_weight']) + '_alpha_' + str(config['alpha']) + '_step_' + str(config['noise_steps'])
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
    results_file = os.path.join(config['save_dir'], 'results.log')
    with open(results_file, 'w') as f_results:
        f_results.write("Logging test and validation results\n")

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))


    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer()(config, model, mg)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

                # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        # Append results to file
        with open(results_file, 'a') as f_results:
            f_results.write(f"Hyper-parameters: {hyper_tuple}\n")
            f_results.write(f"Best valid result: {dict2str(best_valid_result)}\n")
            f_results.write(f"Test result: {dict2str(best_test_upon_valid)}\n\n")
        
        # Save best model if it has the highest validation score
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
            # Save model parameters
            best_model_path = os.path.join(config['save_dir'], 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved at {best_model_path}")
        

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # log info
    logger.info('Training finished.')
    
    # Log and write final best results
    logger.info(f"Best model at index {best_test_idx} with hyper-parameters {hyper_ret[best_test_idx][0]}")
    with open(results_file, 'a') as f_results:
        f_results.write("\nBest Results\n")
        f_results.write(f"Best model index,{best_test_idx}\n")
        f_results.write(f"Best hyper-parameters,{str(hyper_ret[best_test_idx][0])}\n")
        f_results.write("Metric,Valid,Test\n")
        
        # Generate valid_metric_list from metrics and topk if not already defined
        if config.get('valid_metric_list') is None:
            metrics = config.get('metrics', [])
            topk = config.get('topk', [])
            valid_metric_list = []
            for metric in metrics:
                for k in topk:
                    valid_metric_list.append(f"{metric.lower()}@{k}")
            config['valid_metric_list'] = valid_metric_list
        
        for metric in config['valid_metric_list']:
            valid_value = hyper_ret[best_test_idx][1].get(metric, '')
            test_value = hyper_ret[best_test_idx][2].get(metric, '')
            f_results.write(f"{metric},{valid_value},{test_value}\n")
        
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))

