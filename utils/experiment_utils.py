import argparse
import os
import time
import numpy as np
import pandas as pd
import sys
import joblib
import torch
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error
from math import sqrt
from options.base_options import BaseOptions
from data.alloy import carbide
from utils.model_utils import network_outer_report, split_data, tml_report, network_report, extract_metrics,\
    train_network, eval_network
from utils.plot_utils import create_bar_plot, create_violin_plot, create_strip_plot, plot_parity_224, plot_num_points_effect
from call_methods import make_network, create_loaders
from icecream import ic

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def train_tml_model_nested_cv(opt: argparse.Namespace, parent_dir:str, num_points:int=None) -> None:

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
    ncv_iterator = split_data(data, num_points)

    # Initiate the counter of the total runs and the total number of runs
    counter = 0
    TOT_RUNS = opt.folds*(opt.folds-1)    
    print("Number of splits: {}".format(opt.folds))
    print("Total number of runs: {}".format(TOT_RUNS))

    if num_points:
        log_dir = f"{current_dir}/{opt.log_dir_results}/{opt.exp_name}/less_points_exps/num_points_{num_points}/results_atomistic_potential/"
    else:
        log_dir = f"{current_dir}/{opt.log_dir_results}/{opt.exp_name}/results_atomistic_potential/"

    
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
            tml_report(log_dir=log_dir,
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
            log_dir=f"{log_dir}/Fold_{outer}_test_set/",
            outer=outer,
        )

        print('---------------------------------')
        
    print('All runs completed')


def train_networt_less_points(opt: argparse.Namespace, current_dir:str, n_points:int=None) -> None:

    print('Initialising interstitial alloy experiment using early stopping and {} points'.format(n_points))

    # Set the device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the dataset
    carbides = carbide(opt = opt, root=opt.root, filename=opt.filename, max_d=opt.max_d, step=opt.step, name='Mo2C_222')

    # Create the loaders and nested cross validation iterators
    ncv_iterators = create_loaders(carbides, opt, num_points=n_points)

    # Initiate the counter of the total runs
    counter = 0
    TOT_RUNS = opt.folds*(opt.folds-1)

    # Loop through the nested cross validation iterators
    # The outer loop is for the outer fold or test fold
    for outer in range(1, opt.folds+1):
        # The inner loop is for the inner fold or validation fold
        for inner in range(1, opt.folds):

            # Inner fold is incremented by 1 to avoid having same inner and outer fold number for logging purposes
            real_inner = inner +1 if outer <= inner else inner

            # Initiate the early stopping parameters
            val_best_loss = 1000
            early_stopping_counter = 0
            best_epoch = 0
            # Increment the counter
            counter += 1
            # Get the data loaders
            train_loader, val_loader, test_loader = next(ncv_iterators)
            # Initiate the lists to store the losses
            train_list, val_list, test_list = [], [], []
            lr_list = []
            # Create the GNN model
            model = make_network(network_name = "CGC",
                                 opt = opt, 
                                 n_node_features= carbides.num_node_features,
                                 n_edge_features= carbides.num_edge_features).to(device)
            
            # Start the timer for the training
            start_time = time.time()

            for epoch in range(1, opt.epochs+1):

                # Checks if the early stopping counter is less than the early stopping parameter
                if early_stopping_counter <= opt.early_stopping:

                    # Get the learning rate
                    lr = model.scheduler.optimizer.param_groups[0]['lr']

                    # Train the model
                    train_loss = train_network(model, train_loader, device)
                    # Evaluate the model
                    val_loss = eval_network(model, val_loader, device)
                    model.scheduler.step(val_loss)
                    # Evaluate the model on the test set
                    test_loss = eval_network(model, test_loader, device)  

                    print('{}/{}-Epoch {:03d} | LR: {:.5f} | Train loss: {:.3f} eV | Validation loss: {:.3f} eV | '             
                        'Test loss: {:.3f} eV'.format(counter, TOT_RUNS, epoch, lr, train_loss, val_loss, test_loss))
                    
                    if epoch %5 == 0:
                        # Append the losses to the lists
                        train_list.append(train_loss)
                        val_list.append(val_loss)
                        test_list.append(test_loss)
                        lr_list.append(lr)
                        
                    # Save the model if the validation loss is the best
                    if val_loss < val_best_loss:
                            # Best validation loss and early stopping counter updated
                            val_best_loss, best_epoch = val_loss, epoch
                            early_stopping_counter = 0
                            print('New best validation loss: {:.4f} found at epoch {}'.format(val_best_loss, best_epoch))
                            # Save the  best model parameters
                            best_model_params = deepcopy(model.state_dict())
                    else:
                            # Early stopping counter is incremented
                            early_stopping_counter += 1

                    if epoch == opt.epochs-1:
                        print('Maximum number of epochs reached')

                else:
                    print('Early stopping limit reached')
                    break
            
            print('---------------------------------')
            # End the timer for the training
            training_time = (time.time() - start_time)/60
            print('Training time: {:.2f} minutes'.format(training_time))

            print(f"Training for test outer fold: {outer}, and validation inner fold: {real_inner} completed.")
            print(f"Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)}")

            print('---------------------------------')

            # Report the model performance
            network_report(
                log_dir=f"{current_dir}/{opt.log_dir_results}/Mo2C_222/less_points_exps/num_points_{n_points}/results_GNN/",
                loaders=(train_loader, val_loader, test_loader),
                outer=outer,
                inner=real_inner,
                loss_lists=(train_list, val_list, test_list),
                lr_list=lr_list,
                save_all=True,
                model=model,
                model_params=best_model_params,
                best_epoch=best_epoch,
            )

            # Reset the variables of the training
            del model, train_loader, val_loader, test_loader, train_list, val_list, test_list, best_model_params, best_epoch
        
        print(f'All runs for outer test fold {outer} completed')
        print('Generating outer report')

        network_outer_report(
            log_dir=f"{current_dir}/{opt.log_dir_results}/Mo2C_222/less_points_exps/num_points_{n_points}/results_GNN/Fold_{outer}_test_set/",
            outer=outer,
            folds=opt.folds,
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
        
    print('All runs completed')


def plot_results(exp_dir, opt: argparse.Namespace) -> None:

    experiments_gnn = os.path.join(exp_dir, opt.exp_name, 'results_GNN')
    experiments_tml = os.path.join(exp_dir, opt.exp_name, f'results_atomistic_potential')

    r2_gnn, mae_gnn, rmse_gnn = [], [], []
    r2_mlr, mae_mlr, rmse_mlr = [], [], []

    results_all = pd.DataFrame(columns = ['index', 'Test_Fold', 'Val_Fold', 'Method', 'DFT_energy(eV)', 'ML_Predicted_Energy(eV)'])

    if opt.exp_name == 'Mo2C_222':
        results_224 = pd.DataFrame(columns = ['index', 'Test_Fold', 'Val_Fold', 'Method', 'DFT_energy(eV)', 'ML_Predicted_Energy(eV)'])
        r2_gnn_224, mae_gnn_224, rmse_gnn_224 = [], [], []
        r2_mlr_224, mae_mlr_224, rmse_mlr_224 = [], [], []
        experiments_gnn_224 = os.path.join(exp_dir, 'Mo2C_224', 'results_GNN')
        experiments_tml_224 = os.path.join(exp_dir, 'Mo2C_224', f'results_atomistic_potential')

    for outer in range(1, opt.folds+1):

        outer_gnn = os.path.join(experiments_gnn, f'Fold_{outer}_test_set')
        outer_tml = os.path.join(experiments_tml, f'Fold_{outer}_test_set')

        metrics_gnn = extract_metrics(file=outer_gnn+f'/performance_outer_test_fold{outer}.txt')
        metrics_mlr = extract_metrics(file=outer_tml+f'/performance_outer_test_fold{outer}.txt')

        r2_gnn.append(metrics_gnn['R2'])
        mae_gnn.append(metrics_gnn['MAE'])
        rmse_gnn.append(metrics_gnn['RMSE'])

        r2_mlr.append(metrics_mlr['R2'])
        mae_mlr.append(metrics_mlr['MAE'])
        rmse_mlr.append(metrics_mlr['RMSE'])

        for inner in range(1, opt.folds):
    
            real_inner = inner +1 if outer <= inner else inner
            
            gnn_dir = os.path.join(experiments_gnn, f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            df_gnn = pd.read_csv(gnn_dir+'/predictions_test_set.csv')
            df_gnn['Test_Fold'] = outer
            df_gnn['Val_Fold'] = real_inner
            df_gnn['Method'] = 'GNN'

            results_all = pd.concat([results_all, df_gnn], axis=0, ignore_index=True)

            tml_dir = os.path.join(experiments_tml, f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            df_tml = pd.read_csv(tml_dir+'/predictions_test_set.csv')
            df_tml['Test_Fold'] = outer
            df_tml['Val_Fold'] = real_inner
            df_tml['Method'] = 'AtomisticPotential'

            results_all = pd.concat([results_all, df_tml], axis=0, ignore_index=True)

            if opt.exp_name == 'Mo2C_222':
                gnn_dir = os.path.join(experiments_gnn_224, f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')
                df_gnn_224 = pd.read_csv(gnn_dir+'/predictions_test_set.csv')
                df_gnn_224['Test_Fold'] = outer
                df_gnn_224['Val_Fold'] = real_inner
                df_gnn_224['Method'] = 'GNN'

                results_224 = pd.concat([results_224, df_gnn_224], axis=0, ignore_index=True)

                tml_dir = os.path.join(experiments_tml_224, f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')
                df_tml_224 = pd.read_csv(tml_dir+'/predictions_test_set.csv')
                df_tml_224['Test_Fold'] = outer
                df_tml_224['Val_Fold'] = real_inner
                df_tml_224['Method'] = 'AtomisticPotential'

                results_224 = pd.concat([results_224, df_tml_224], axis=0, ignore_index=True)

    save_dir = f'{exp_dir}/{opt.exp_name}/GNN_vs_MLR'
    os.makedirs(save_dir, exist_ok=True)
    
    mae_mean_gnn = np.array([entry['mean'] for entry in mae_gnn])
    mae_gnn_std = np.array([entry['std'] for entry in mae_gnn])

    mae_mean_mlr = np.array([entry['mean'] for entry in mae_mlr])
    mae_mlr_std = np.array([entry['std'] for entry in mae_mlr])

    rmse_mean_gnn = np.array([entry['mean'] for entry in rmse_gnn])
    rmse_gnn_std = np.array([entry['std'] for entry in rmse_gnn])

    rmse_mean_mlr = np.array([entry['mean'] for entry in rmse_mlr])
    rmse_mlr_std = np.array([entry['std'] for entry in rmse_mlr])

    minimun = np.min(np.array([
        (mae_mean_gnn - mae_gnn_std).min(), 
        (mae_mean_mlr - mae_mlr_std).min(),
        (rmse_mean_gnn - rmse_gnn_std).min(), 
        (rmse_mean_mlr - rmse_mlr_std).min()]))
    
    maximun = np.max(np.array([
        (mae_mean_gnn + mae_gnn_std).max(),
        (mae_mean_mlr + mae_mlr_std).max(),
        (rmse_mean_gnn + rmse_gnn_std).max(), 
        (rmse_mean_mlr + rmse_mlr_std).max()]))


    create_bar_plot(means=(mae_mean_gnn, mae_mean_mlr), stds=(mae_gnn_std, mae_mlr_std), min = minimun, max = maximun, metric = 'MAE', save_path= save_dir, tml_algorithm='Atomistic Potential')
    create_bar_plot(means=(rmse_mean_gnn, rmse_mean_mlr), stds=(rmse_gnn_std, rmse_mlr_std), min = minimun, max = maximun, metric = 'RMSE', save_path= save_dir, tml_algorithm='Atomistic Potential')

    r2_mean_gnn = np.array([entry['mean'] for entry in r2_gnn])
    r2_gnn_std = np.array([entry['std'] for entry in r2_gnn])

    r2_mean_mlr = np.array([entry['mean'] for entry in r2_mlr])
    r2_mlr_std = np.array([entry['std'] for entry in r2_mlr])

    minimun = np.min(np.array([
    (r2_mean_gnn - r2_gnn_std).min(),
    (r2_mean_mlr - r2_mlr_std).min(),
    ]))

    maximun = np.max(np.array([
    (r2_mean_gnn + r2_gnn_std).max(),
    (r2_mean_mlr + r2_mlr_std).max(),
    ]))

    create_bar_plot(means=(r2_mean_gnn, r2_mean_mlr), stds=(r2_gnn_std, r2_mlr_std), min = minimun, max = maximun, metric = 'R2', save_path= save_dir, tml_algorithm='Atomistic Potential')
    
    results_all['ML_Predicted_Energy(eV)'] = results_all['GNN_energy(eV)'].fillna(results_all['AtomisticPotential_energy(eV)'])
    results_all = results_all.drop(['GNN_energy(eV)', 'AtomisticPotential_energy(eV)'], axis=1)
    results_all['Error'] = results_all['DFT_energy(eV)'] - results_all['ML_Predicted_Energy(eV)']

    results_all.to_csv(f'{save_dir}/all_predictions.csv', index=False)

    create_violin_plot(data=results_all, save_path= save_dir)
    create_strip_plot(data=results_all, save_path= save_dir)

    if opt.exp_name == 'Mo2C_222':
        save_dir = f'{exp_dir}/Mo2C_224/GNN_vs_MLR'
        os.makedirs(save_dir, exist_ok=True)

        results_224['ML_Predicted_Energy(eV)'] = results_224['GNN_energy(eV)'].fillna(results_224['AtomisticPotential_energy(eV)'])
        results_224 = results_224.drop(['GNN_energy(eV)', 'AtomisticPotential_energy(eV)'], axis=1)
        results_224['Error'] = results_224['DFT_energy(eV)'] - results_224['ML_Predicted_Energy(eV)']

        results_224.to_csv(f'{save_dir}/all_predictions.csv', index=False)

        df_gnn = results_224.loc[results_224['Method'] == 'GNN']

        for structure in df_gnn['index'].unique():
            df_gnn.loc[df_gnn['index'] == structure, 'Mean_Delta_E'] = df_gnn.loc[df_gnn['index'] == structure, 'ML_Predicted_Energy(eV)'].mean()
            df_gnn.loc[df_gnn['index'] == structure, 'Std_Delta_E'] = df_gnn.loc[df_gnn['index'] == structure, 'ML_Predicted_Energy(eV)'].std()

        df_gnn = df_gnn.drop_duplicates(subset='index', keep='first')

        df_mlr = results_224.loc[results_224['Method'] == 'AtomisticPotential']

        for structure in df_mlr['index'].unique():
            df_mlr.loc[df_mlr['index'] == structure, 'Mean_Delta_E'] = df_mlr.loc[df_mlr['index'] == structure, 'ML_Predicted_Energy(eV)'].mean()
            df_mlr.loc[df_mlr['index'] == structure, 'Std_Delta_E'] = df_mlr.loc[df_mlr['index'] == structure, 'ML_Predicted_Energy(eV)'].std()

        df_mlr = df_mlr.drop_duplicates(subset='index', keep='first')

        plot_parity_224(df_gnn, df_mlr, save_path=save_dir)


def plot_num_points_exp(exp_dir, opt: argparse.Namespace) -> None:

    r2_gnn_all, mae_gnn_all, rmse_gnn_all = [], [], []
    r2_std_gnn_all, mae_std_gnn_all, rmse_std_gnn_all = [], [], []

    r2_ap_all, mae_ap_all, rmse_ap_all = [], [], []
    r2_std_ap_all, mae_std_ap_all, rmse_std_ap_all = [], [], []

    for i in range(100,1001, opt.sampling_size):

        r2_gnn, mae_gnn, rmse_gnn = [], [], []
        r2_ap, mae_ap, rmse_ap = [], [], []

        points_dir = f'{exp_dir}/num_points_{i}'

        gnn_dir_p = os.path.join(points_dir, 'results_GNN')
        ap_dir_p = os.path.join(points_dir, 'results_atomistic_potential')

        for outer in range(1, opt.folds+1):
            for inner in range(1, opt.folds):

                real_inner = inner +1 if outer <= inner else inner

                gnn_dir = os.path.join(gnn_dir_p, f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')
                ap_dir = os.path.join(ap_dir_p, f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

                metrics_gnn = extract_metrics(file=gnn_dir+f'/performance.txt', outer=False)
                metrics_ap = extract_metrics(file=ap_dir+f'/performance.txt', outer=False)
                
                r2_gnn.append(metrics_gnn['R2'] if metrics_gnn['R2'] != None else 0)
                mae_gnn.append(metrics_gnn['MAE'])
                rmse_gnn.append(metrics_gnn['RMSE'])

                r2_ap.append(metrics_ap['R2'])
                mae_ap.append(metrics_ap['MAE'])
                rmse_ap.append(metrics_ap['RMSE'])
        
        r2_gnn_all.append(np.mean(r2_gnn))
        r2_std_gnn_all.append(np.std(r2_gnn))
        mae_gnn_all.append(np.mean(mae_gnn))
        mae_std_gnn_all.append(np.std(mae_gnn))
        rmse_gnn_all.append(np.mean(rmse_gnn))
        rmse_std_gnn_all.append(np.std(rmse_gnn))

        r2_ap_all.append(np.mean(r2_ap))
        r2_std_ap_all.append(np.std(r2_ap))
        mae_ap_all.append(np.mean(mae_ap))
        mae_std_ap_all.append(np.std(mae_ap))
        rmse_ap_all.append(np.mean(rmse_ap))
        rmse_std_ap_all.append(np.std(rmse_ap))

    exp_dir = f'{exp_dir}/results_GNN_vs_AP'

    os.makedirs(exp_dir, exist_ok=True)

    plot_num_points_effect(means=(r2_gnn_all, r2_ap_all), stds=(r2_std_gnn_all, r2_std_ap_all), num_points=list(range(100,1001, opt.sampling_size)), metric='R2', save_path=exp_dir)
    plot_num_points_effect(means=(mae_gnn_all, mae_ap_all), stds=(mae_std_gnn_all, mae_std_ap_all), num_points=list(range(100,1001, opt.sampling_size)), metric='MAE', save_path=exp_dir)
    plot_num_points_effect(means=(rmse_gnn_all, rmse_ap_all), stds=(rmse_std_gnn_all, rmse_std_ap_all), num_points=list(range(100,1001, opt.sampling_size)), metric='RMSE', save_path=exp_dir)




