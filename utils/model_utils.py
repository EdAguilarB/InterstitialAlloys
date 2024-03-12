import torch
import os
import csv
import re
import pickle
import numpy as np
from datetime import date, datetime
from copy import copy, deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, \
    accuracy_score, precision_score, recall_score, f1_score
from math import sqrt
from utils.plot_utils import *
from icecream import ic
from sklearn.preprocessing import RobustScaler


def calculate_metrics(y_true:list, y_predicted: list,  task = 'r'):

    if task == 'r':
        r2 = r2_score(y_true=y_true, y_pred=y_predicted)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_predicted)
        rmse = sqrt(mean_squared_error(y_true=y_true, y_pred=y_predicted))  
        error = [(y_predicted[i]-y_true[i]) for i in range(len(y_true))]
        prctg_error = [ abs(error[i] / y_true[i]) for i in range(len(error)) if y_true[i] != 0]
        mbe = np.mean(error)
        mape = np.mean(prctg_error)
        error_std = np.std(error)
        metrics = [r2, mae, rmse, mbe, mape, error_std]
        metrics_names = ['R2', 'MAE', 'RMSE', 'Mean Bias Error', 'Mean Absolute Percentage Error', 'Error Standard Deviation']

    elif task == 'c':
        accuracy = accuracy_score(y_true=y_true, y_pred=y_predicted)
        precision = precision_score(y_true=y_true, y_pred=y_predicted)
        recall = recall_score(y_true=y_true, y_pred=y_predicted)
        f1 = f1_score(y_true=y_true, y_pred=y_predicted)
        metrics = [accuracy, precision, recall, f1]
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']

    return np.array(metrics), metrics_names

######################################
######################################
######################################
###########  GNN functions ###########
######################################
######################################
######################################

def train_network(model, train_loader, device):

    train_loss = 0
    model.train()

    for batch in train_loader:
        batch = batch.to(device)
        model.optimizer.zero_grad()
        out = model(batch)
        loss = torch.sqrt(model.loss(out, torch.unsqueeze(batch.y.float(), dim=1)))
        loss.backward()
        model.optimizer.step()

        train_loss += loss.item() * batch.num_graphs

    return train_loss / len(train_loader.dataset)

def eval_network(model, loader, device):
    model.eval()
    loss = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss += torch.sqrt(model.loss(out, torch.unsqueeze(batch.y, dim = 1) )).item() * batch.num_graphs
    return loss / len(loader.dataset)


def predict_network(model, loader):
    model.to('cpu')
    model.eval()

    y_pred, y_true, idx = [], [], []

    for batch in loader:
        batch = batch.to('cpu')
        out = model(batch)

        y_pred.append(out.cpu().detach().numpy())
        y_true.append(batch.y.cpu().detach().numpy())
        idx.append(batch.file_name)

    y_pred = np.concatenate(y_pred, axis=0).ravel()
    y_true = np.concatenate(y_true, axis=0).ravel()
    idx = np.concatenate(idx, axis=0).ravel()

    return y_pred, y_true, idx


def network_report(log_dir,
                   loaders,
                   outer,
                   inner, 
                   loss_lists,
                   lr_list,
                   save_all,
                   model, 
                   model_params,
                   best_epoch,
                   normalize = False,):


    #1) Create a directory to store the results
    log_dir = "{}/Fold_{}_test_set/Fold_{}_val_set".format(log_dir, outer, inner)
    os.makedirs(log_dir, exist_ok=True)

    #2) Time of the run
    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]
    run_period = "{}, {}\n".format(today_str, time)

    #3) Unfold loaders and save loaders and model
    train_loader, val_loader, test_loader = loaders[0], loaders[1], loaders[2]
    N_train, N_val, N_test = len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)
    N_tot = N_train + N_val + N_test     
    if save_all == True:
        torch.save(train_loader, "{}/train_loader.pth".format(log_dir))
        torch.save(val_loader, "{}/val_loader.pth".format(log_dir))
        torch.save(model, "{}/model.pth".format(log_dir))
        torch.save(model_params, "{}/model_params.pth".format(log_dir))
    torch.save(test_loader, "{}/test_loader.pth".format(log_dir)) 
    loss_function = 'RMSE_eV'

    #4) loss trend during training
    train_list = loss_lists[0]
    val_list = loss_lists[1] 
    test_list = loss_lists[2]
    if train_list is not None and val_list is not None and test_list is not None:
        with open('{}/{}.csv'.format(log_dir, 'learning_process'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch", "Train_{}".format(loss_function), "Val_{}".format(loss_function), "Test_{}".format(loss_function), "Learning Rate"])
            for i in range(len(train_list)):
                writer.writerow([(i+1)*5, train_list[i], val_list[i], test_list[i], lr_list[i]])
        create_training_plot(df='{}/{}.csv'.format(log_dir, 'learning_process'), save_path='{}'.format(log_dir))


    #5) Start writting report
    file1 = open("{}/performance.txt".format(log_dir), "w")
    file1.write(run_period)
    file1.write("---------------------------------------------------------\n")
    file1.write("GNN TRAINING AND PERFORMANCE\n")
    file1.write("Best epoch: {}\n".format(best_epoch))
    file1.write("Dataset Size = {}\n".format(N_tot))
    file1.write("***************\n")

    model.load_state_dict(model_params)

    y_pred, y_true, idx = predict_network(model, train_loader)
    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    file1.write("Training set\n")
    file1.write("Set size = {}\n".format(N_train))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    
    file1.write("***************\n")
    y_pred, y_true, idx = predict_network(model, val_loader)
    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    file1.write("Validation set\n")
    file1.write("Set size = {}\n".format(N_val))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("***************\n")

    y_pred, y_true, idx = predict_network(model, test_loader)

    if normalize:
        min_index = np.argmin(y_true)
        y_pred = np.array([pred - y_pred[min_index] for pred in y_pred])

    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    pd.DataFrame({'DFT_energy(eV)': y_true, 'GNN_energy(eV)': y_pred, 'index': idx}).to_csv("{}/predictions_test_set.csv".format(log_dir))

    file1.write("Test set\n")
    file1.write("Set size = {}\n".format(N_test))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("---------------------------------------------------------\n")

    create_st_parity_plot(real = y_true, predicted = y_pred, figure_name = 'outer_{}_inner_{}'.format(outer, inner), save_path = "{}".format(log_dir))
    #create_it_parity_plot(real = y_true, predicted = y_pred, index = idx, figure_name='outer_{}_inner_{}.html'.format(outer, inner), save_path="{}".format(log_dir))

    file1.write("OUTLIERS (TEST SET)\n")
    error_test = [(y_pred[i] - y_true[i]) for i in range(len(y_pred))]
    abs_error_test = [abs(error_test[i]) for i in range(len(y_pred))]
    std_error_test = np.std(error_test)

    outliers_list, outliers_error_list, index_list = [], [], []

    counter = 0

    for sample in range(len(y_pred)):
        if abs_error_test[sample] >= 3 * std_error_test:  
            counter += 1
            outliers_list.append(idx[sample])
            outliers_error_list.append(error_test[sample])
            index_list.append(sample)
            if counter < 10:
                file1.write("0{}) {}    Error: {:.2f} eV    (index={})\n".format(counter, idx[sample], error_test[sample], sample))
            else:
                file1.write("{}) {}    Error: {:.2f} eV    (index={})\n".format(counter, idx[sample], error_test[sample], sample))

    file1.close()

    return 'Report saved in {}'.format(log_dir)


def network_outer_report(log_dir: str,
                         outer: int,
                         folds: int = 5,):
    
    
    r2, mae, rmse = [], [], [] 

    files = [log_dir+f'Fold_{i}_val_set/performance.txt' for i in range(1, folds+1) if i != outer]

    # Define regular expressions to match metric lines
    r2_pattern = re.compile(r"R2 = (\d+\.\d+)")
    mae_pattern = re.compile(r"MAE = (\d+\.\d+)")
    rmse_pattern = re.compile(r"RMSE = (\d+\.\d+)")
    
    for file in files:
        with open(os.path.join(file), 'r') as f:
            content = f.read()
        
        # Split the content by '*' to separate different sets
        sets = content.split('*')

        for set_content in sets:
            # Check if "Test set" is in the set content
            if "Test set" in set_content:
                # Extract metric values using regular expressions
                r2_match = r2_pattern.search(set_content)
                try:
                    r2.append(float(r2_match.group(1)))
                except:
                    r2.append(0)
                mae_match = mae_pattern.search(set_content)
                mae.append(float(mae_match.group(1)))
                rmse_match = rmse_pattern.search(set_content)
                rmse.append(float(rmse_match.group(1)))

    # Calculate mean and standard deviation for each metric
    r2_mean = np.mean(r2)
    r2_std = np.std(r2)
    mae_mean = np.mean(mae)
    mae_std = np.std(mae)
    rmse_mean = np.mean(rmse)
    rmse_std = np.std(rmse)

    # Write the results to the file
    file1 = open("{}/performance_outer_test_fold{}.txt".format(log_dir, outer), "w")
    file1.write("---------------------------------------------------------\n")
    file1.write("Test Set Metrics (mean ± std)\n")
    file1.write("R2: {:.3f} ± {:.3f}\n".format(r2_mean, r2_std))
    file1.write("MAE: {:.3f} ± {:.3f}\n".format(mae_mean, mae_std))
    file1.write("RMSE: {:.3f} ± {:.3f}\n".format(rmse_mean, rmse_std))
    file1.write("---------------------------------------------------------\n")

    return 'Report saved in {}'.format(log_dir)



######################################
######################################
######################################
###########  TML functions ###########
######################################
######################################
######################################

def split_data(df:pd.DataFrame, num_points:int=None):

    if num_points:
        short_df = pd.DataFrame(columns=df.columns)
        k,m = divmod(num_points, len(np.unique(df['fold'])))
        for fold in np.unique(df['fold']):
            short_df = pd.concat([short_df, df.loc[df['fold'] == fold][:k+1 if fold < m+1 else k]])
        del df
        df = short_df
        
    for outer in np.unique(df['fold']):
        proxy = copy(df)
        test = proxy[proxy['fold'] == outer]

        for inner in np.unique(df.loc[df['fold'] != outer, 'fold']):

            val = proxy.loc[proxy['fold'] == inner]
            train = proxy.loc[(proxy['fold'] != outer) & (proxy['fold'] != inner)]
            yield deepcopy((train, val, test))



def predict_tml(model, data:pd.DataFrame):
    descriptors = ['c', 'bv', 'ang']
    y_pred = model.predict(data[descriptors])
    y_true = list(data['dft'])
    idx = list(data['file'])
    return np.array(y_pred), np.array(y_true), np.array(idx)


def tml_report(log_dir,
               outer, 
                inner,
               model, 
               data, 
               save_all=True,
               normalize = False,):
    
    #1) create a directory to store the results
    log_dir = "{}/Fold_{}_test_set/Fold_{}_val_set".format(log_dir, outer, inner)
    os.makedirs(log_dir, exist_ok=True)

    #2) Get time of the run
    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]
    run_period = "{}, {}\n".format(today_str, time)

    #3) Unfold train/val/test dataloaders
    train_data, val_data, test_data = data[0], data[1], data[2]
    N_train, N_val, N_test = len(train_data), len(val_data), len(test_data)
    N_tot = N_train + N_val + N_test 

    #4) Save dataframes for future use
    if save_all:
        train_data.to_csv("{}/train.csv".format(log_dir))
        val_data.to_csv("{}/val.csv".format(log_dir))
        pickle.dump(model, open("{}/model.sav".format(log_dir), 'wb'))

    test_data.to_csv("{}/test.csv".format(log_dir))

    #5) Performance Report
    file1 = open("{}/performance.txt".format(log_dir), "w")
    file1.write(run_period)
    file1.write("---------------------------------------------------------\n")
    file1.write("Traditional ML algorithm Performance\n")
    file1.write("Dataset Size = {}\n".format(N_tot))
    file1.write("c term coefficient = {}\n".format(model.coef_[0]))
    file1.write("bv term coefficient = {}\n".format(model.coef_[1]))
    file1.write("ang term coefficient = {}\n".format(model.coef_[2]))
    file1.write("***************\n")

    y_pred, y_true, idx = predict_tml(model, train_data)
    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    file1.write("Training set\n")
    file1.write("Set size = {}\n".format(N_train))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))
    
    file1.write("***************\n")

    y_pred, y_true, idx = predict_tml(model, val_data)
    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    file1.write("Validation set\n")
    file1.write("Set size = {}\n".format(N_val))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))
    
    file1.write("***************\n")

    y_pred, y_true, idx = predict_tml(model=model, data=test_data)

    if normalize:
        min_index = np.argmin(y_true)
        y_pred = np.array([pred - y_pred[min_index] for pred in y_pred])

    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    pd.DataFrame({'DFT_energy(eV)': y_true, 'AtomisticPotential_energy(eV)': y_pred, 'index': idx}).to_csv("{}/predictions_test_set.csv".format(log_dir))

    file1.write("Test set\n")
    file1.write("Set size = {}\n".format(N_val))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("---------------------------------------------------------\n")

    create_st_parity_plot(real = y_true, predicted = y_pred, figure_name = 'outer_{}_inner_{}'.format(outer, inner), save_path = "{}".format(log_dir))
    #create_it_parity_plot(real = y_true, predicted = y_pred, index = idx, figure_name='outer_{}_inner_{}.html'.format(outer, inner), save_path="{}".format(log_dir))

    file1.write("OUTLIERS (TEST SET)\n")
    error_test = [(y_pred[i] - y_true[i]) for i in range(len(y_pred))]
    abs_error_test = [abs(error_test[i]) for i in range(len(y_pred))]
    std_error_test = np.std(error_test)

    outliers_list, outliers_error_list, index_list = [], [], []

    counter = 0

    for sample in range(len(y_pred)):
        if abs_error_test[sample] >= 3 * std_error_test:  
            counter += 1
            outliers_list.append(idx[sample])
            outliers_error_list.append(error_test[sample])
            index_list.append(sample)
            if counter < 10:
                file1.write("0{}) {}    Error: {:.3f} eV    (index={})\n".format(counter, idx[sample], error_test[sample], sample))
            else:
                file1.write("{}) {}    Error: {:.3f} eV    (index={})\n".format(counter, idx[sample], error_test[sample], sample))

    file1.close()

    return 'Report saved in {}'.format(log_dir)




######################################
######################################
######################################
#########  General functions #########
######################################
######################################
######################################




def extract_metrics(file, outer = True):

    metrics = {'R2': None, 'MAE': None, 'RMSE': None}

    with open(file, 'r') as file:
            content = file.read()

    
    if outer:
        # Define regular expressions to match metric lines
        r2_pattern = re.compile(r"R2: (\d+\.\d+) ± (\d+\.\d+)")
        mae_pattern = re.compile(r"MAE: (\d+\.\d+) ± (\d+\.\d+)")
        rmse_pattern = re.compile(r"RMSE: (\d+\.\d+) ± (\d+\.\d+)")

    else:
        # Define regular expressions to match metric lines
        r2_pattern = re.compile(r"R2 = (\d+\.\d+)")
        mae_pattern = re.compile(r"MAE = (\d+\.\d+)")
        rmse_pattern = re.compile(r"RMSE = (\d+\.\d+)")

    r2_match = r2_pattern.search(content)
    mae_match = mae_pattern.search(content)
    rmse_match = rmse_pattern.search(content)

    if outer:
        # Update the metrics dictionary with extracted values
        if r2_match:
            metrics['R2'] = {'mean': float(r2_match.group(1)), 'std': float(r2_match.group(2))}
        if mae_match:
            metrics['MAE'] = {'mean': float(mae_match.group(1)), 'std': float(mae_match.group(2))}
        if rmse_match:
            metrics['RMSE'] = {'mean': float(rmse_match.group(1)), 'std': float(rmse_match.group(2))}
    
    else:
        # Update the metrics dictionary with extracted values
        if r2_match:
            metrics['R2'] = float(r2_match.group(1))
        if mae_match:
            metrics['MAE'] = float(mae_match.group(1))
        if rmse_match:
            metrics['RMSE'] = float(rmse_match.group(1))

    return metrics
