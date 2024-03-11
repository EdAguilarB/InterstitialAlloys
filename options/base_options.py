import argparse


class BaseOptions:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        self.parser.add_argument(
            '--experiment_name',
            type=str,
            default='experiment',
            help='name of the experiment',
            ),

        self.parser.add_argument(
            '--filename',
            type=str,
            default='learning.csv',
            help='name of the csv file',
            )

        self.parser.add_argument(
            '--root', 
            type=str, 
            default='data/datasets/rhcaa_learning',
            help='path to the folder containing the csv files',
            )    

        self.parser.add_argument(
            '--filename_TiC',
            type=str,
            default='final_test.csv',
            help='name of the csv file for the final test',
            )
        
        self.parser.add_argument(
            '--root_TiC', 
            type=str, 
            default='data/datasets/rhcaa_final_test',
            help='path to the folder containing the csv files',
            )

        self.parser.add_argument(
            '--filename_final_test',
            type=str,
            default='final_test.csv',
            help='name of the csv file for the final test',
            )
        
        self.parser.add_argument(
            '--root_final_test', 
            type=str, 
            default='data/datasets/rhcaa_final_test',
            help='path to the folder containing the csv files',
            )
        
        self.parser.add_argument(
            '--log_dir_results',
            type=str,
            default='results/',
            help='path to the folder where the results will be saved',
            )
        
        self.parser.add_argument(
            '--mol_cols',
            type=str,
            default=['Ligand', 'substrate', 'boron reagent'],
            help='column names of the reactant and product smiles',
            )
        
        self.parser.add_argument(
            '--folds',
            type=int,
            default=10,
            help='Number of folds',
            )
        
        self.parser.add_argument(
            '--n_classes',
            type=int,
            default=1,
            help='Number of classes',
            )
        
        self.parser.add_argument(
            '--n_convolutions',
            type=int,
            default=2,
            help='Number of convolutions',
            )
        
        self.parser.add_argument(
            '--readout_layers',
            type=int,
            default=2,
            help='Number of readout layers',
            )
        
        self.parser.add_argument(
            '--embedding_dim',
            type=int,
            default=64,
            help='Embedding dimension',
            )
        
        self.parser.add_argument(
            '--global_seed',
            type=int,
            default=123456789,
            help='Global random seed for reproducibility',
            )
        
        self.initialized = True


    def parse(self):
        if not self.initialized:
            self.initialize()
        self._opt = self.parser.parse_args()

        return self._opt