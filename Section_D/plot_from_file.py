import torch
import argparse
import metrics as metrics
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--directory_path', type=str, help='Load path of directory')
parser.add_argument('--pred_file_names', type=str, help='Load names of prediction files, names are split by hypens ')
parser.add_argument('--targets_file_names', type=str, help='Load names of target files')
parser.add_argument('--names', type=str, help='Load names for the graph key')


def main():    
    predictions_list = args.pred_file_names.split(",")
    targets_list = args.targets_file_names.split(",")
    keys_list = args.names.split(",")
    print(predictions_list, targets_list, keys_list)
    
    targets, predictions = [],[]
    for target, prediction in zip(targets_list, predictions_list):
        targets.append(np.loadtxt(args.directory_path + target).astype(np.int64))
        predictions.append(np.loadtxt(args.directory_path + prediction, dtype=float))
    metrics.multiplots(predictions, targets, keys_list)

if __name__ == '__main__':
    args = parser.parse_args()
    main() 