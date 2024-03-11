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
            default=3.5,
            help='Maximum distance for neighbour search',
            )
        
        self.parser.add_argument(
            '--step',
            type=float,
            default=0.5,
            help='Gaussian expansion step',
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
            '--embedding_dim',
            type=int,
            default=16,
            help='Embedding dimension',
            )
        
        self.parser.add_argument(
            '--readout_layers',
            type=int,
            default=2,
            help='Number of readout layers',
            )
        
        self.parser.add_argument(
            '--batch_norm',
            type=bool,
            default=True,
            help='Batch normalization',
            )
        
        self.parser.add_argument(
            '--pooling',
            type=str,
            default='gap',
            help='Pooling method',
            )
    
        self.parser.add_argument(
            '--problem_type',
            type=str,
            default='regression',
            help='Type of problem',
            )
        
        self.parser.add_argument(
            '--optimizer',
            type=str,
            default='Adam',
            help='Optimizer',
            )
        
        self.parser.add_argument(
            '--lr',
            type=float,
            default=0.01,
            help='Learning rate',
            )
        
        self.parser.add_argument(
            '--amsgrad',
            type=bool,
            default=True,
            help='Amsgrad for Adam optimizer',
            )
        
        self.parser.add_argument(
            '--scheduler',
            type=str,
            default='ReduceLROnPlateau',
            help='Scheduler',
            )
        
        self.parser.add_argument(
            '--step_size',
            type=int,
            default=7,
            help='Step size for the scheduler',
            )
        
        self.parser.add_argument(
            '--gamma',
            type=float,
            default=0.7,
            help='Gamma for the scheduler',
            )
        
        self.parser.add_argument(
            '--min_lr',
            type=float,
            default=1e-08,
            help='Minimum learning rate for the scheduler',
            )
        
        self.parser.add_argument(
            '--batch_size',
            type=int,
            default=40,
            help='Batch size',
        )
        
        self.parser.add_argument(
            '--epochs',
            type=int,
            default=300,
            help='Number of epochs',
            )
        
        self.parser.add_argument(
            '--early_stopping',
            type=int,
            default=6,
            help='Early stopping',
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