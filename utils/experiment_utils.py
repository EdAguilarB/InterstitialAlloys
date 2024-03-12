import argparse
import os
import pandas as pd
import sys
import joblib
import torch
from sklearn.linear_model import LinearRegression
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error
from math import sqrt
from options.base_options import BaseOptions
from data.alloy import carbide
from utils.model_utils import network_outer_report, split_data, tml_report, network_report

from icecream import ic

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def train_tml_model_nested_cv(opt: argparse.Namespace, parent_dir:str) -> None:

    print('Initialising chiral ligands selectivity prediction using a traditional ML approach.')
    
    # Get the current working directory 
    current_dir = parent_dir

    # Load the data
    filename = opt.filename[:-4] + '_folds' + opt.filename[-4:]
    data = pd.read_csv(f'{opt.root}{opt.exp_name}/{filename}')

    data['dft'] = (data['dft']-min(data['dft']))*13.605693122994
    data = data[['dft', 'c', 'bv', 'ang', 'file', 'fold']]
    descriptors = ['c', 'bv', 'ang']
    
    # Nested cross validation
    ncv_iterator = split_data(data)

    # Initiate the counter of the total runs and the total number of runs
    counter = 0
    TOT_RUNS = opt.folds*(opt.folds-1)    
    print("Number of splits: {}".format(opt.folds))
    print("Total number of runs: {}".format(TOT_RUNS))

    
    # Loop through the nested cross validation iterators
    # The outer loop is for the outer fold or test fold
    for outer in range(1, opt.folds+1):
        # The inner loop is for the inner fold or validation fold
        for inner in range(1, opt.folds):

            # Inner fold is incremented by 1 to avoid having same inner and outer fold number for logging purposes
            real_inner = inner +1 if outer <= inner else inner
            # Increment the counter
            counter += 1

            # Get the train, validation and test sets
            train_set, val_set, test_set = next(ncv_iterator)
            # Choose the model
            model = LinearRegression()
            # Fit the model
            model.fit(train_set[descriptors], train_set['dft'])
            # Predict the train set
            preds = model.predict(train_set[descriptors])
            train_rmse = sqrt(mean_squared_error(train_set['dft'], preds))
            # Predict the validation set
            preds = model.predict(val_set[descriptors])
            val_rmse = sqrt(mean_squared_error(val_set['dft'], preds))
            # Predict the test set
            preds = model.predict(test_set[descriptors])
            test_rmse = sqrt(mean_squared_error(test_set['dft'], preds))

            print('Outer: {} | Inner: {} | Run {}/{} | Train RMSE {:.3f} eV | Val RMSE {:.3f} eV | Test RMSE {:.3f} eV'.\
                  format(outer, real_inner, counter, TOT_RUNS, train_rmse, val_rmse, test_rmse) )
            
            # Generate a report of the model performance
            tml_report(log_dir=f"{current_dir}/{opt.log_dir_results}/{opt.exp_name}/results_atomistic_potential/",
                       data = (train_set, val_set, test_set),
                       outer = outer,
                       inner = real_inner,
                       model = model,
                       )
            
            # Reset the variables of the training
            del model, train_set, val_set, test_set
        
        print('All runs for outer test fold {} completed'.format(outer))
        print('Generating outer report')

        # Generate a report of the model performance for the outer/test fold
        network_outer_report(
            log_dir=f"{current_dir}/{opt.log_dir_results}/{opt.exp_name}/results_atomistic_potential/Fold_{outer}_test_set/",
            outer=outer,
        )

        print('---------------------------------')
        
    print('All runs completed')




def predict_final_test(parent_dir:str, opt: argparse.Namespace) -> None:

    opt = BaseOptions().parse()

    current_dir = parent_dir
    
    # Load the final test set
    final_test = carbide(opt, opt.root, opt.filename, opt.max_d, opt.step, 'Mo2C_224', include_fold=False, norm=False)
    test_loader = DataLoader(final_test, shuffle=False)

    # Load the data for tml
    test_set = pd.read_csv(f'{opt.root}/Mo2C_224/{opt.filename}')

    experiments_gnn = os.path.join(current_dir, opt.log_dir_results, 'Mo2C_224', 'results_GNN')
    experiments_tml = os.path.join(current_dir, opt.log_dir_results, 'Mo2C_224', f'results_atomistic_potential')

    for outer in range(1, opt.folds+1):
        print('Analysing models trained using as test set {}'.format(outer))
        for inner in range(1, opt.folds):
    
            real_inner = inner +1 if outer <= inner else inner
            
            print('Analysing models trained using as validation set {}'.format(real_inner))

            model_dir = os.path.join(current_dir, opt.log_dir_results, 'Mo2C_222', 'results_GNN', f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            model = torch.load(model_dir+'/model.pth')
            model_params = torch.load(model_dir+'/model_params.pth')
            train_loader = torch.load(model_dir+'/train_loader.pth')
            val_loader = torch.load(model_dir+'/val_loader.pth')

            network_report(log_dir=experiments_gnn,
                           loaders=(train_loader, val_loader, test_loader),
                           outer=outer,
                           inner=real_inner,
                           loss_lists=[None, None, None],
                            lr_list=None,
                           model=model,
                           model_params=model_params,
                           best_epoch=None,
                           save_all=False,
                           normalize=True)
            
            tml_dir = os.path.join(current_dir, opt.log_dir_results, 'Mo2C_222', f'results_atomistic_potential', f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            model = joblib.load(tml_dir+'/model.sav')
            train_data = pd.read_csv(tml_dir+'/train.csv')
            val_data = pd.read_csv(tml_dir+'/val.csv')

            tml_report(log_dir=experiments_tml,
                       outer=outer,
                       inner=real_inner,
                       model=model,
                       data=(train_data,val_data,test_set),
                       save_all=False,
                       normalize=True,)
            
                        
        network_outer_report(log_dir=f"{experiments_gnn}/Fold_{outer}_test_set/", 
                             outer=outer)
        
        network_outer_report(log_dir=f"{experiments_tml}/Fold_{outer}_test_set/", 
                             outer=outer)