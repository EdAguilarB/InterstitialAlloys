import argparse
from torch_geometric.data import Dataset
import os
import pandas as pd
import torch

from icecream import ic



class interstitial_alloy(Dataset):


    def __init__(self, opt: argparse.Namespace, root: str, filename: str, max_d: float, step: float =.5) -> None:

        # This variables must be initialized before calling the super().__init__ method as super().__init__ 
        # will call the download and process methods, which require the variables to be initialized to run properly

        #self._name = "BaseDataset"
        self.filename = filename
        self.max_d = max_d
        self.step = step
        self.vor_cut_off = opt.vor_cut_off
        self._opt = opt
        self._root = root
        super().__init__(root = self._root)
        
    @property
    def raw_file_names(self):
        files = os.listdir(os.path.join(self.root, 'raw'))
        return sorted(files)
    
    @property
    def processed_file_names(self):
        return [f'{self._name}_{idx}.pt' for idx in range(len(self.raw_file_names))]
    
    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError
    
    def _get_node_feats(self):
        raise NotImplementedError
    
    def _get_edge_features(self):
        raise NotImplementedError
    
    def _print_dataset_info(self) -> None:
        """
        Prints the dataset info
        """
        print(f"{self._name} dataset has {len(self)} samples")

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):

        material = torch.load(os.path.join(self.processed_dir, 
                                f'{self._name}_{idx}.pt')) 
        return material
    

    
    def _one_h_e(self, x, allowable_set, ok_set=None):

        if x not in allowable_set:
            if ok_set is not None and x == ok_set:
                pass
            else:
                print(x)
        return list(map(lambda s: x == s, allowable_set))