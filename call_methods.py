import argparse
from copy import copy, deepcopy
from torch_geometric.loader import DataLoader
from icecream import ic


def make_network(network_name: str, opt: argparse.Namespace, n_node_features: int, n_edge_features:int=None):
    if network_name == "CGC":
        from model.cgc import crysgraphconv
        return crysgraphconv(opt=opt, n_node_features=n_node_features, n_edge_features=n_edge_features)
    else:
        raise ValueError(f"Network {network_name} not implemented")
    

def create_loaders(dataset, opt: argparse.Namespace):

    """
    Creates training, validation and testing loaders for cross validation and
    inner cross validation training-evaluation processes.
    Args:
    dataset: pytorch geometric datset
    batch_size (int): size of batches
    val (bool): whether or not to create a validation set
    folds (int): number of folds to be used
    num points (int): number of points to use for the training and evaluation process

    Returns:
    (tuple): DataLoaders for training, validation and test set

    """

    batch_size = opt.batch_size
    folds = opt.folds


    folds = [[] for _ in range(folds)]
    for data in dataset:
        folds[data.fold-1].append(data)

    for outer in range(len(folds)):
        proxy = copy(folds)
        test_loader = DataLoader(proxy.pop(outer), batch_size=batch_size, shuffle=False)
        for inner in range(len(proxy)):  # length is reduced by 1 here
            proxy2 = copy(proxy)
            val_loader = DataLoader(proxy2.pop(inner), batch_size=batch_size, shuffle=False)
            flatten_training = [item for sublist in proxy2 for item in sublist]  # flatten list of lists
            train_loader = DataLoader(flatten_training, batch_size=batch_size, shuffle=True)
            yield deepcopy((train_loader, val_loader, test_loader))