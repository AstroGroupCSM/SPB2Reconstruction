__author__ = 'Connor Heaton'

import argparse
import datetime
import glob
import time
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'none':
        return None
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_data(datadir):
    start_time = time.time()
    stats = {'min': [], 'mean': [], 'median': [], 'max': []}

    print('Reading from {}...'.format(datadir))
    for raw_file in os.listdir(datadir):
        filepath = os.path.join(datadir, raw_file)
        data = np.load(filepath)

        data_min = np.min(data)
        data_mean = np.mean(data)
        data_median = np.median(data)
        data_max = np.max(data)

        stats['min'].append(data_min)
        stats['mean'].append(data_mean)
        stats['median'].append(data_median)
        stats['max'].append(data_max)
    elapsed_time = time.time() - start_time
    print('Returning data from {0}... elapsed time: {1:.2f}s...'.format(datadir, elapsed_time))

    return stats


def plot_stats(data, label):
    colors = ['red', 'green', 'blue', 'yellow']

    plt.hist(data['min'], bins=25, color=colors[0], alpha=0.5, label='min')
    plt.hist(data['mean'], bins=25, color=colors[1], alpha=0.5, label='mean')
    plt.hist(data['median'], bins=25, color=colors[2], alpha=0.5, label='median')
    plt.hist(data['max'], bins=25, color=colors[3], alpha=0.5, label='max')

    plt.title(label)
    plt.legend()


def plot_signal_strengths(args):
    noise_dir = os.path.join(args.data, 'Noise')
    signal_dir = os.path.join(args.data, 'Signal')
    plt_file = os.path.join(args.out, 'stat_summary.png')

    pool_args = [(noise_dir,), (signal_dir,)]
    print('Creating pool to read data...')
    with Pool(processes=2) as pool:
        stats = pool.starmap(read_data, pool_args)

    print('Noise min: {}'.format(stats[0]['min']))
    print('Noise mean: {}'.format(stats[0]['mean']))
    print('Noise median: {}'.format(stats[0]['median']))
    print('Noise max: {}'.format(stats[0]['max']))

    print('Signal min: {}'.format(stats[1]['min']))
    print('Signal mean: {}'.format(stats[1]['mean']))
    print('Signal median: {}'.format(stats[1]['median']))
    print('Signal max: {}'.format(stats[1]['max']))

    plt.subplot(1, 2, 1)
    plot_stats(stats[0], 'Noise')
    plt.subplot(1, 2, 2)
    plot_stats(stats[1], 'Signal')
    plt.savefig(plt_file, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='/home/datasets/george_data/v2', help='Dir where data can be found')
    parser.add_argument('--out', default='../out', help='Directory to put output')

    args = parser.parse_args()

    plot_signal_strengths(args)