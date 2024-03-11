import argparse
import os
import pandas as pd
import sys

from utils.model_utils import network_outer_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))






def train_tml_model_nested_cv(opt: argparse.Namespace, parent_dir:str) -> None:

    print('Initialising chiral ligands selectivity prediction using a traditional ML approach.')
    
    # Get the current working directory 
    current_dir = parent_dir

    # Load the data
    filename = opt.filename[:-4] + '_folds' + opt.filename[-4:]
    data = pd.read_csv(f'{opt.root}/{opt.exp_name}/raw/{filename}')
    data = data[['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
               'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4', '%top', 'fold', 'index']]
    descriptors = ['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
               'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4']
    
    # Nested cross validation
    ncv_iterator = split_data(data)

    # Initiate the counter of the total runs and the total number of runs
    counter = 0
    TOT_RUNS = opt.folds*(opt.folds-1)    
    print("Number of splits: {}".format(opt.folds))
    print("Total number of runs: {}".format(TOT_RUNS))

    # Hyperparameter optimisation
    print("Hyperparameter optimisation starting...")
    X, y, _ = load_variables(f'{opt.root}/raw/learning_folds.csv')
    best_params = hyperparam_tune(X, y, choose_model(best_params=None, algorithm = opt.tml_algorithm), 123456789)
    print('Hyperparameter optimisation has finalised')
    print("Training starting...")
    print("********************************")
    
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
            model = choose_model(best_params, opt.tml_algorithm)
            # Fit the model
            model.fit(train_set[descriptors], train_set['%top'])
            # Predict the train set
            preds = model.predict(train_set[descriptors])
            train_rmse = sqrt(mean_squared_error(train_set['%top'], preds))
            # Predict the validation set
            preds = model.predict(val_set[descriptors])
            val_rmse = sqrt(mean_squared_error(val_set['%top'], preds))
            # Predict the test set
            preds = model.predict(test_set[descriptors])
            test_rmse = sqrt(mean_squared_error(test_set['%top'], preds))

            print('Outer: {} | Inner: {} | Run {}/{} | Train RMSE {:.3f} % | Val RMSE {:.3f} % | Test RMSE {:.3f} %'.\
                  format(outer, real_inner, counter, TOT_RUNS, train_rmse, val_rmse, test_rmse) )
            
            # Generate a report of the model performance
            tml_report(log_dir=f"{current_dir}/{opt.log_dir_results}/learning_set/results_{opt.tml_algorithm}/",
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
            log_dir=f"{current_dir}/{opt.log_dir_results}/learning_set/results_{opt.tml_algorithm}/Fold_{outer}_test_set/",
            outer=outer,
        )

        print('---------------------------------')
        
    print('All runs completed')