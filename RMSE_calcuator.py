from argparse import ArgumentParser
import numpy as np

def data_args(parser: ArgumentParser):
    """
    Adds data arguments.
    """
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file for calculating RMSE')
    
    parser.add_argument('--ref_data_path', type=str,
                        help='Path to data CSV file as the reference values')
    
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to CSV file where result will be saved')
    
def parse_data_args() -> Namespace:
    parser = ArgumentParser()
    data_args(parser)
    args = parser.parse_args()
    
    return args


args = parsea_data_args()


def read_data(path):
    """
    Read the data. Column 1: sample, Column 2: target value
    """
    f = open(path, 'r')
    lines = f.readlines()
    sample_list = []
    target_value_list = []

    for line in lines[1:]:    # lines[1:] for removing header
        data = line.strip().split(',')
        sample = data[0]
        target_value = data[1]
        sample_list.append(sample)
        target_value_list.append(float(target_value))
    f.close()

    return sample_list, target_value_list
    
def





