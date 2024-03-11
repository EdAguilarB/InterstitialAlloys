from options.base_options import BaseOptions
import os
import torch

from data.alloy import carbide


def train_networt_nested_cv():

    print('Initialising interstitial alloy experiment using early stopping')

    # Get hyperparameters
    opt = BaseOptions().parse()

    # Get the current working directory
    current_dir = os.getcwd()

    # Set the device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the dataset
    data = carbide(opt = opt, root=opt.root, filename=opt.filename, max_d=opt.max_d, step=opt.step, name='Mo2C_222')


if __name__ == "__main__":
    train_networt_nested_cv()
    

