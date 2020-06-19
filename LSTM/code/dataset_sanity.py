__author__ = 'Connor Heaton'

import os
import math
import time
import torch
import argparse

import matplotlib.pyplot as plt

from datasets import PhysicsDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='/home/czh/datasets/george_data', help='Dir where data can be found')
    parser.add_argument('--out', default='../out', help='Directory to put output')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size to use')

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    dataset = PhysicsDataset('train', args.data, args, only_noise=True, undersample=0.25)
    noise_min = float('inf')
    noise_max = float('-inf')
    noise_max_values = []

    for i in range(len(dataset)):
        item = dataset[i]
        data = item['data']
        #print('Item {}- data: {} min: {} max: {}'.format(i, data.shape, torch.min(data), torch.max(data)))
        if torch.min(data) < noise_min:
            noise_min = torch.min(data)

        if torch.max(data) > noise_max:
            noise_max = torch.max(data)

        # data = data.view(-1).tolist()
        noise_max_values.append(torch.max(data).item())

    print('noise min: {} noise max: {}'.format(noise_min, noise_max))

    dataset = PhysicsDataset('train', args.data, args, only_signal=True, undersample=0.25)
    signal_min = float('inf')
    signal_max = float('-inf')
    signal_max_values = []

    for i in range(len(dataset)):
        item = dataset[i]
        data = item['data']
        # print('Item {}- data: {} min: {} max: {}'.format(i, data.shape, torch.min(data), torch.max(data)))
        if torch.min(data) < signal_min:
            signal_min = torch.min(data)

        if torch.max(data) > signal_max:
            signal_max = torch.max(data)

        # data = data.view(-1).tolist()
        signal_max_values.append(torch.max(data).item())

    print('signal min: {} signal max: {}'.format(signal_min, signal_max))

    if not os.path.exists('../figs'):
        os.makedirs('../figs')

    outfile = '../figs/data_values_hist.png'
    plt.hist(noise_max_values, bins=20, color='red', label='Noise', alpha=0.5, range=[0, 124])
    plt.hist(signal_max_values, bins=20, color='blue', label='Signal', alpha=0.5, range=[0, 124])
    plt.legend()
    plt.title('Data Values')
    plt.savefig(outfile, bbox_inches='tight')
    plt.clf()

    print('ty!!')