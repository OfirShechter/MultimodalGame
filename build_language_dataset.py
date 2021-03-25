import pickle
from random import random
import argparse

import logging

import json
import numpy as np

import torch

parser = argparse.ArgumentParser(description='Process .pkl language outputs to json datasets')
parser.add_argument('--path', metavar='P', type=str, nargs='+', help='path for language pkl')
parser.add_argument('--outpath', metavar='P', type=str, nargs='+', help='output path for language pkl')


def run(path_l, outpath_l):
    for path, outpath in zip(path_l, outpath_l):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # tensors to lists
        for row in data:
            for col, val in row.items():
                if isinstance(val, list):
                    for i, v in enumerate(val):
                        if isinstance(v, torch.Tensor):
                            val[i] = v.tolist()
                if isinstance(val, torch.Tensor):
                    row[col] = val.tolist()
        # dump
        with open(outpath, 'w') as f:
            json.dump(data, f)


if __name__ == '__main__':
    args = parser.parse_args()
    assert len(args.path) == len(args.outpath)

    run(args.path, args.outpath)
