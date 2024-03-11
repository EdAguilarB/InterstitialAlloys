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
            default='energy_descriptor.csv',
            help='name of the csv file',
            )

        self.parser.add_argument(
            '--root', 
            type=str, 
            default='data/datasets/',
            help='path to the folder containing the csv files',
            )    
        
        self.parser.add_argument(
            '--folds',
            type=int,
            default=5,
            help='Number of folds',
            )
        
        self.parser.add_argument(
            '--max_d',
            type=float,
            default=10,
            help='Maximum distance',
            )
        
        self.parser.add_argument(
            '--step',
            type=float,
            default=0.5,
            help='Step',
            )

        self.parser.add_argument(
            '--log_dir_results',
            type=str,
            default='results/',
            help='path to the folder where the results will be saved',
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