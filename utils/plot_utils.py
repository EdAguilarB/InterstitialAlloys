import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import os
from seaborn import violinplot, stripplot, kdeplot
from math import sqrt



#########################################################
#########################################################
#########################################################
#############Functions to create plots###################
#########################################################
#########################################################
#########################################################

def create_st_parity_plot(real, predicted, figure_name, save_path=None):
    """
    Create a parity plot and display R2, MAE, and RMSE metrics.

    Args:
        real (numpy.ndarray): An array of real (actual) values.
        predicted (numpy.ndarray): An array of predicted values.
        save_path (str, optional): The path where the plot should be saved. If None, the plot is not saved.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object.
        matplotlib.axes._axes.Axes: The Matplotlib axes object.
    """
    # Calculate R2, MAE, and RMSE
    r2 = r2_score(real, predicted)
    mae = mean_absolute_error(real, predicted)
    rmse = np.sqrt(mean_squared_error(real, predicted))
    
    # Create the parity plot
    plt.figure(figsize=(8, 8))
    plt.scatter(real, predicted, alpha=0.7)
    plt.plot([min(real), max(real)], [min(real), max(real)], color='red', linestyle='--')
    plt.xlabel('DFT Calculated Internal Energy (eV)', fontsize=16)
    plt.ylabel('ML Predicted Internal Energy (eV)', fontsize=16)
    
    # Display R2, MAE, and RMSE as text on the plot
    textstr = f'$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
    plt.gcf().text(0.15, 0.75, textstr, fontsize=12)
    
    # Save the plot if save_path is provided
    if save_path:
        # Ensure the directory exists
        save_path = os.path.join(save_path, figure_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')

    plt.close()

    

def create_it_parity_plot(real, predicted, index, figure_name, save_path=None):
    r2 = round(r2_score(real, predicted), 3)
    mae = round(mean_absolute_error(real, predicted), 3)
    rmse = round(np.sqrt(mean_squared_error(real, predicted)), 3)


    df = pd.DataFrame({'Real':real,
                       'Predicted': predicted,
                       'Idx': index})

    # Create a scatter plot
    fig = px.scatter(df, x='Real', y='Predicted', text = 'Idx', labels={'x': 'DFT Calculated Internal Energy (eV)', 'y': 'GNN Predicted Internal Energy (eV)'}, hover_data=['Idx', 'Real', 'Predicted'])
    fig.add_trace(go.Scatter(x=real, y=real, mode='lines', name='Perfect Fit', line=dict(color='red', dash='dash')))

    # Customize the layout
    fig.update_layout(
        title=f'Parity Plot',
        showlegend=True,
        legend=dict(x=0, y=1),
        xaxis=dict(showgrid=True, showline=True, zeroline=True, linewidth=1, linecolor='black'),
        yaxis=dict(showgrid=True, showline=True, zeroline=True, linewidth=1, linecolor='black'),
        plot_bgcolor='white',  # Set background color to white
    )

    # Display R2, MAE, and RMSE as annotations on the plot
    text_annotation = f'R2 = {r2:.3f}<br>MAE = {mae:.3f}<br>RMSE = {rmse:.3f}'
    fig.add_annotation(
        text=text_annotation,
        xref="paper", yref="paper",
        x=0.15, y=0.75,
        showarrow=False,
        font=dict(size=12),
    )

    # Save the plot as an HTML file if save_path is provided
    if save_path:
        # Ensure the directory exists
        save_path = os.path.join(save_path, figure_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)


def create_training_plot(df, save_path):

    df = pd.read_csv(df)

    epochs = df.iloc[:,0]
    train_loss = df.iloc[:,1]
    val_loss = df.iloc[:,2]
    test_loss = df.iloc[:,3]

    min_val_loss_epoch = epochs[val_loss.idxmin()]

    # Create a Matplotlib figure and axis
    plt.figure(figsize=(10, 6), dpi=300)  # Adjust the figure size as needed
    plt.plot(epochs, train_loss, label='Train Loss', marker='o', linestyle='-')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o', linestyle='-')
    plt.plot(epochs, test_loss, label='Test Loss', marker='o', linestyle='-')

    plt.axvline(x=min_val_loss_epoch, color='gray', linestyle='--', label=f'Min Validation Epoch ({min_val_loss_epoch})')

    # Customize the plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(False)
    plt.legend()

    # Save the plot in high resolution (adjust file format as needed)
    plt.savefig('{}/loss_vs_epochs.png'.format(save_path), bbox_inches='tight')

    plt.close()



def create_bar_plot(means:tuple, stds:tuple, min:float, max:float, metric:str, save_path:str, tml_algorithm:str, n_folds:int=5):

    plt.figure(figsize=(10, 8), dpi=300)

    bar_width = 0.35

    mean_gnn, mean_tml = means
    std_gnn, std_tml = stds

    folds = list(range(1, n_folds+1))
    index = np.arange(n_folds)

    plt.bar(index, mean_gnn, bar_width, label='GNN Approach', yerr=std_gnn, capsize=5)
    plt.bar(index+bar_width, mean_tml, bar_width, label=f'{tml_algorithm.upper()} Approach', yerr=std_tml, capsize=5)

    plt.ylim(min*.98, max *1.02)
    plt.xlabel('Fold Used as Test Set', fontsize = 34)

    label = 'Mean $R^2$ Value' if metric == 'R2' else f'Mean {metric} Value / eV'
    plt.ylabel(label, fontsize = 34)

    plt.xticks(index + bar_width / 2, list(folds))

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.savefig(os.path.join(save_path, f'{metric}_GNN_vs_TML'), dpi=300, bbox_inches='tight')

    print('Plot {}_GNN_vs_TML has been saved in the directory {}'.format(metric,save_path))

    plt.clf()



def create_violin_plot(data, save_path:str):

    plt.figure(figsize=(10, 8), dpi=300)

    violinplot(data = data, x='Test_Fold', y='Error', hue='Method', split=True, gap=.1, inner="quart", fill=True)

    plt.xlabel('Fold Used as Test Set', fontsize=34)
    plt.ylabel('$Energy_{DFT}-Energy_{predicted}$ / eV', fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    ax= plt.gca()
    ax.get_legend().remove()

    plt.savefig(os.path.join(save_path, f'Error_distribution_GNN_vs_AP_violin_plot'), dpi=300, bbox_inches='tight')

    print('Plot Error_distribution_GNN_vs_AP_violin_plot has been saved in the directory {}'.format(save_path))
    plt.clf()


def create_strip_plot(data, save_path:str):

    plt.figure(figsize=(10, 8), dpi=300)

    stripplot(data = data, x='Test_Fold', y='Error', hue='Method', size=3,  dodge=True, jitter=True, marker='D', alpha=.3)

    plt.xlabel('Fold Used as Test Set', fontsize=34)
    plt.ylabel('$Energy_{DFT}-Energy_{predicted}$ / eV', fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    ax= plt.gca()
    ax.get_legend().remove()

    plt.savefig(os.path.join(save_path, f'Error_distribution_GNN_vs_AP_strip_plot'), dpi=300, bbox_inches='tight')

    print('Plot Error_distribution_GNN_vs_AP_strip_plot has been saved in the directory {}'.format(save_path))
    plt.clf()


def plot_parity_224(df_GNN, df_ap, save_path:str):
    
    lr_GNN = LinearRegression()
    lr_GNN.fit(df_GNN[['DFT_energy(eV)']], df_GNN['Mean_Delta_E'])
    predicted_Delta_E_GNN = lr_GNN.predict(df_GNN[['DFT_energy(eV)']])

    lr_ap = LinearRegression()
    lr_ap.fit(df_ap[['DFT_energy(eV)']], df_ap['Mean_Delta_E'])
    predicted_Delta_E_ap = lr_ap.predict(df_ap[['DFT_energy(eV)']])

    plt.figure(dpi=300)

    plt.scatter(df_GNN['DFT_energy(eV)'], df_GNN['Mean_Delta_E'], label='GNN', c='#1f77b4')
    plt.errorbar(df_GNN['DFT_energy(eV)'], df_GNN['Mean_Delta_E'], yerr=df_GNN['Std_Delta_E'], fmt='o', markersize=5,  capsize=3, c='#1f77b4')
    plt.plot(df_GNN['DFT_energy(eV)'], predicted_Delta_E_GNN,  c='#1f77b4')

    plt.scatter(df_ap['DFT_energy(eV)'], df_ap['Mean_Delta_E'], label = 'Atomistic Potential', c ='#ff7f0e')
    plt.errorbar(df_ap['DFT_energy(eV)'], df_ap['Mean_Delta_E'], yerr=df_ap['Std_Delta_E'], fmt='o', markersize=5,  capsize=3, c= '#ff7f0e')
    plt.plot(df_ap['DFT_energy(eV)'], predicted_Delta_E_ap, c='#ff7f0e')

    plt.plot([df_GNN['DFT_energy(eV)'].min(), df_GNN['DFT_energy(eV)'].max()], [df_GNN['DFT_energy(eV)'].min(), df_GNN['DFT_energy(eV)'].max()], 'k--', label='DFT')

    plt.text(6.5,2, 'MAE GNN: {:.3f}'.format(mean_absolute_error(df_GNN['DFT_energy(eV)'], df_GNN['Mean_Delta_E'])))
    plt.text(6.5,1.5, 'RMSE GNN: {:.3f}'.format(sqrt(mean_squared_error(df_GNN['DFT_energy(eV)'], df_GNN['Mean_Delta_E']))))

    plt.text(6.5,0.5, 'MAE AP: {:.3f}'.format(mean_absolute_error(df_ap['DFT_energy(eV)'], df_ap['Mean_Delta_E'])))
    plt.text(6.5,0, 'RMSE AP: {:.3f}'.format(sqrt(mean_squared_error(df_ap['DFT_energy(eV)'], df_ap['Mean_Delta_E']))))

    plt.xlabel('DFT Calculated $\Delta$E / eV', fontsize=18)
    plt.ylabel('ML predicted $\Delta$E / eV', fontsize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend()
    plt.savefig(os.path.join(save_path, f'parity_plot_224_cells'), dpi=300, bbox_inches='tight')
    plt.clf()


def plot_num_points_effect(means:tuple, stds:tuple, num_points:list, metric:str, save_path:str):

    means_gnn, means_ap = means
    stds_gnn, stds_ap = stds
        
    fig, ax = plt.subplots()

    # Plot mean_MAE vs num_data_points with error bars for std_MAE
    ax.errorbar(num_points, means_gnn, yerr=stds_gnn, label=f'GNN',  capsize=5, color = '#1f77b4')

    # Plot mean_RMSE vs num_data_points with error bars for std_RMSE
    ax.errorbar(num_points, means_ap, yerr=stds_ap, label=f'Atomistic Potential',  capsize=5, color = '#ff7f0e')

    ax.set_xlabel('Number of Data Points', fontsize = 16)

    label = 'Mean $R^2$ Value' if metric == 'R2' else f'Mean {metric} Value / eV'
    ax.set_ylabel(label, fontsize = 16)

    ax.plot(num_points, means_gnn, 'o-', color='#1f77b4',  markersize=8, linewidth=2)
    ax.plot(num_points, means_ap, 'o-', color='#ff7f0e',  markersize=8, linewidth=2)

    #ax.legend()
    plt.grid(False)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.savefig(os.path.join(save_path, f'{metric}_vs_num_points'), dpi=300, bbox_inches='tight')

    print('Plot {}_vs_num_points has been saved in the directory {}'.format(metric,save_path))

    plt.clf()


def plot_diff_distribution(data, save_path, file_name):

    plt.figure(figsize=(8, 8))

    mean_value = np.mean(data)
    median_value = np.median(data)

    kdeplot(data, fill=True, color='skyblue')

    # Add a vertical line at the median and mean
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2)
    plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2)

    # Create custom legend labels
    mean_legend_label = f'Mean deviation: {mean_value:.3f} eV'
    median_legend_label = f'Median deviation: {median_value:.3f} eV'
    total_points_label = f'Attribution Scores for {len(data)} Structures'

    # Labeling and title
    plt.xlabel('$\hat{Y}_{i}-\hat{Y}_{masked}$')
    plt.ylabel('Density')

    plt.legend([total_points_label, mean_legend_label, median_legend_label])

    save_dir = os.path.join(save_path, file_name)

    plt.savefig(save_dir, dpi=300, bbox_inches='tight')

    plt.clf()

    return 'Plot saved in {}'.format(save_dir)
