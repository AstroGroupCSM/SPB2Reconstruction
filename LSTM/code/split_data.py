__author__ = 'Connor Heaton'

import os
import numpy as np

from sklearn.model_selection import train_test_split


def partition_files(files, p_train, p_dev, p_test):
    n = len(files)
    idxs = np.array([i for i in range(n)])

    idx_train, idx_test = train_test_split(idxs, train_size=p_train)
    idx_dev, idx_test = train_test_split(idx_test, train_size=p_dev / (p_dev + p_test))

    print('idx_train: {}'.format(idx_train.shape))
    print('idx_dev: {}'.format(idx_dev.shape))
    print('idx_test: {}'.format(idx_test.shape))

    idx_train = list(idx_train)
    idx_dev = list(idx_dev)
    idx_test = list(idx_test)

    train_files = [files[idx] for idx in idx_train]
    dev_files = [files[idx] for idx in idx_dev]
    test_files = [files[idx] for idx in idx_test]

    return train_files, dev_files, test_files


def write_to_file(noise_files, signal_files, out_file):
    data = []

    for noise_file in noise_files:
        d = np.load(noise_file)
        d = np.expand_dims(d, axis=0)
        data.append(d)

    for signal_file in signal_files:
        d = np.load(signal_file)
        d = np.expand_dims(d, axis=0)
        data.append(d)

    data = np.vstack(data)
    print('data shape: {}'.format(data.shape))
    np.save(out_file, data)


if __name__ == '__main__':
    data_dir = '/home/datasets/george_data/v2'
    noise_dir = os.path.join(data_dir, 'Noise')
    signal_dir = os.path.join(data_dir, 'Signal')

    p_train = 0.7
    p_dev = 0.15
    p_test = 0.15

    noise_files = [os.path.join('Noise', f) for f in os.listdir(noise_dir)]
    signal_files = [os.path.join('Signal', f) for f in os.listdir(signal_dir)]

    print('N noise files: {}'.format(len(noise_files)))
    print('N signal files: {}'.format(len(signal_files)))

    print('Partitioning noise files...')
    noise_train_files, noise_dev_files, noise_test_files = partition_files(noise_files, p_train, p_dev, p_test)
    print('Partitioning signal files')
    signal_train_files, signal_dev_files, signal_test_files = partition_files(signal_files, p_train, p_dev, p_test)

    train_files = noise_train_files[:]
    train_files.extend(signal_train_files[:])
    with open(os.path.join(data_dir, 'train_files.txt'), 'w+') as f:
        f.write('\n'.join(train_files))

    dev_files = noise_dev_files[:]
    dev_files.extend(signal_dev_files[:])
    with open(os.path.join(data_dir, 'dev_files.txt'), 'w+') as f:
        f.write('\n'.join(dev_files))

    test_files = noise_test_files[:]
    test_files.extend(signal_test_files[:])
    with open(os.path.join(data_dir, 'test_files.txt'), 'w+') as f:
        f.write('\n'.join(test_files))

    # print('Writing train files to disk...')
    # write_to_file(noise_train_files, signal_train_files, os.path.join(data_dir, 'train_data.npy'))
    #
    # print('Writing dev files to disk...')
    # write_to_file(noise_dev_files, signal_dev_files, os.path.join(data_dir, 'dev_data.npy'))
    #
    # print('Writing test files to disk...')
    # write_to_file(noise_test_files, signal_test_files, os.path.join(data_dir, 'test_data.npy'))

    print('Done!')
