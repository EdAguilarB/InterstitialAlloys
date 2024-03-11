import argparse
import pandas as pd
import torch
from torch_geometric.data import  Data
import numpy as np 
import os
import os.path as osp
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import VoronoiNN
import sys
from data.datasets import interstitial_alloy
from sklearn.model_selection import StratifiedKFold
from icecream import ic

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class carbide(interstitial_alloy):

    def __init__(self, opt:argparse.Namespace, root: str, filename: str, max_d: float, step: float, name:str, include_fold = True) -> None:

        self._include_fold = include_fold
        self._name = name

        if self._include_fold:
            filename = filename[:-4] + '_folds' + filename[-4:]
            try:
                pd.read_csv(os.path.join(root, name, f'{filename}'))
            except:
                self._split_data(root, name, opt.folds)
                

        
        root = os.path.join(root, name)
        
        self.energy = pd.read_csv(os.path.join(root, filename), index_col=0)
        self.energy['dft'] = self.energy['dft']*13.605693122994
        self.min_energy = min(self.energy['dft'])


        super().__init__(opt = opt, root = root, filename = filename, max_d=max_d, step=step)

        
        
    def process(self):

        total_structures = len(self.raw_paths)
        idx = 0
        for raw_path in tqdm(self.raw_paths):

            if idx%100 == 0:
                print('{}/{} structures processed'.format(idx, total_structures))
            # Read data from `raw_path`.

            structure = Structure.from_file(raw_path)
            node_feats = self._get_node_feats(structure)
            adj, edge_feats = self._get_edge(structure)
            file_name = raw_path.split('/')[-1]
            energy, e_norm = self._get_energy(file_name)
            coords = self._get_cords(structure)
            fold = self._get_fold(file_name)

            data = Data(x=node_feats,
                        edge_index=adj,
                        edge_attr=edge_feats,
                        y = e_norm,
                        energy=energy,
                        fold = fold,
                        file_name = file_name,
                        coords = coords)

            torch.save(data, osp.join(self.processed_dir, f'{self._name}_{idx}.pt'))
            idx += 1
            

    def _get_node_feats(self, structure):
        atom_fea = np.vstack([[[1,0] if structure[i].specie.number != 6 else [0,1]
                            for i in range(len(structure))]])
        atom_fea = torch.tensor(atom_fea, dtype=torch.float)
        return atom_fea
    

    def _get_edge(self, structure):

        #gdf = self._GaussianExpansion(vmin=0, vmax=self.max_d, step=self.step)

        vnn = VoronoiNN(cutoff=self.max_d,allow_pathological=True,compute_adj_neighbors=False)
        
        nbr_fea_idx, nbr_fea_t = [], []

        for central_atom in range(len(structure)):

            nbrs = vnn.get_nn_info(structure, central_atom)

            for nbr_info in nbrs: # newer version
                if nbr_info['poly_info']['face_dist']*2 <= self.max_d:
                    nbr_fea_idx.append([central_atom,nbr_info['site_index']])
                    nbr_fea_t.append(nbr_info['poly_info']['solid_angle'])

        nbr_fea_t = self._GaussianExpansion(vmin=0, vmax=self.max_d, step = self.step, v = np.array(nbr_fea_t))
        nbr_fea_idx = np.array(nbr_fea_idx).transpose()
        nbr_fea = np.array(nbr_fea_t)

        return torch.tensor(nbr_fea_idx), torch.tensor(nbr_fea)
    

    def _get_energy(self, file_name):
        e = np.array(self.energy.loc[file_name, 'dft'])
        en_norm = np.array(e - self.min_energy)
        return torch.tensor(e).reshape(1), torch.tensor(en_norm).reshape(1)
    
    def _get_fold(self, file_name):
        fold = self.energy.loc[file_name, 'fold']
        return fold
    
    def _get_cords(self, structure):
        coords = []
        for atom in structure:
            coords.append(atom.coords)
        coords = np.vstack(coords)
        return coords
    
    def _GaussianExpansion(self, vmin, vmax, step, v, var = None):

        assert vmin < vmax
        assert vmax - vmin > step

        filter = np.arange(vmin, vmax+step, step)

        if var is None:
            var = step
        var = var

        return np.exp(-(v[..., np.newaxis] - filter)**2 / var**2)
    

    def _create_folds(num_folds, df):
        """
        splits a dataset in a given quantity of folds

        Args:
        num_folds = number of folds to create
        df = dataframe to be splited

        Returns:
        dataset with new "folds" and "mini_folds" column with information of fold for each datapoint
        """

        # Calculate the number of data points in each fold
        fold_size = len(df) // num_folds
        remainder = len(df) % num_folds

        # Create a 'fold' column to store fold assignments
        fold_column = []

        # Assign folds
        for fold in range(1, num_folds + 1):
            fold_count = fold_size
            if fold <= remainder:
                fold_count += 1
            fold_column.extend([fold] * fold_count)

        # Assign the 'fold' column to the DataFrame
        df['fold'] = fold_column

        return df
    

    def _split_data(self, root, filename, n_folds):

        dataset = pd.read_csv(os.path.join(root, 'raw', f'{filename}'))
        dataset['category'] = dataset['%top'].apply(lambda m: 0 if m < 50 else 1)

        folds = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state=23)

        test_idx = []

        for _, test in folds.split(dataset['ER1'], dataset['category']):
            test_idx.append(test)

        index_dict = {index: list_num for list_num, index_list in enumerate(test_idx) for index in index_list}

        dataset['fold'] = dataset.index.map(index_dict)

        filename = filename[:-4] + '_folds' + filename[-4:]

        dataset.to_csv(os.path.join(root, 'raw', filename))

        print('{}.csv file was saved in {}'.format(filename, os.path.join(root, 'raw')))