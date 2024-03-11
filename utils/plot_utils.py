import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import os
from seaborn import violinplot, stripplot, jointplot, barplot
from icecream import ic

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
    plt.xlabel('DFT Calculated Internal Energy (eV)')
    plt.ylabel('GNN Predicted Internal Energy (eV)')
    
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