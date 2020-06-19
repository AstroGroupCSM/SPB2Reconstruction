__author__ = 'Connor Heaton'

import os
import json
import glob
import math
import torch
import random

import numpy as np
from collections import defaultdict

from torch.utils.data import Dataset


def read_np_file(filename):
    x = np.load(filename)
    return x


class KeyDependentDict(defaultdict):
    def __init__(self):
        super().__init__(None)  # base class doesn't get a factory
        self.f_of_x = lambda x: np.load(x)  # save f(x)

    def __missing__(self, key):  # called when a default needed
        ret = self.f_of_x(key)  # calculate default value
        self[key] = ret  # and install it in the dict
        return ret


class PhysicsDataset(Dataset):
    def __init__(self, mode, basedir, args, only_noise=False, only_signal=False, undersample=None,
                 force_unbalanced=False, disallow_supersample=False):
        random.seed(16)
        self.mode = mode
        self.basedir = basedir
        self.args = args
        self.balance_data = getattr(self.args, 'balance_data', False) and not force_unbalanced
        self.only_noise = only_noise
        self.only_signal = only_signal
        self.undersample = undersample
        self.supersample = getattr(self.args, 'supersample', False) and not disallow_supersample and self.balance_data
        self.do_frame_swap = getattr(self.args, 'do_frame_swaps', False) and self.mode == 'train'
        self.do_transforms = getattr(self.args, 'do_transforms', False) and self.mode == 'train'
        self.transform_prob = getattr(self.args, 'transform_prob', 0.5)
        self.epoch_to_start_frame_swaps = getattr(self.args, 'epoch_to_start_frame_swaps', -1)
        self.pct_frames_to_swap = getattr(self.args, 'pct_frames_to_swap', 0.1)
        self.normalize_data = getattr(self.args, 'normalize_data', False)
        self.norm_by_frame = getattr(self.args, 'norm_by_frame', False)
        self.only_high_signal = getattr(self.args, 'only_high_signal', False)   # and self.mode == 'train'
        self.high_signal_file = '../listOfEventsWithHighSignal.npy'

        self.high_signal_item_ids = []
        if self.only_high_signal:
            self.high_signal_item_ids = np.load(self.high_signal_file).tolist()

        print('PhysicsDataset.basedir: {}'.format(self.basedir))
        self.items = []

        # self.data_file = os.path.join(self.basedir, '{}_files.txt'.format(self.mode if 'eval' not in self.mode
        #                                                                   else 'dev'))
        #
        # with open(self.data_file, 'r') as f:
        #     for line in f:
        #         line = line.strip()
        #         if self.only_high_signal:
        #             type_dir, base_filename = os.path.split(line)
        #             if type_dir == 'Noise':
        #                 self.items.append(os.path.join(self.basedir, line))
        #             else:
        #                 item_id = int(base_filename[:-4].split('-')[1])
        #                 if item_id in self.high_signal_item_ids:
        #                     self.items.append(os.path.join(self.basedir, line))
        #         else:
        #             self.items.append(os.path.join(self.basedir, line))     # add filepath to items

        if self.mode == 'train':
            v2_filename = os.path.join(self.basedir, 'v2', 'train_files.txt')
            with open(v2_filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if 'Noise' in line:
                        self.items.append(os.path.join(self.basedir, 'v2', line))

            v1_filename = os.path.join(self.basedir, 'v1', 'train_files.txt')
            with open(v1_filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if 'Signal' in line:
                        line = line.split(os.path.sep)[-2:]
                        self.items.append(os.path.join(self.basedir, 'v1', line[0], line[1]))
        elif self.mode == 'dev':
            v2_filename = os.path.join(self.basedir, 'v2', 'dev_files.txt')
            with open(v2_filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if self.only_high_signal:
                        type_dir, base_filename = os.path.split(line)
                        if type_dir == 'Noise':
                            self.items.append(os.path.join(self.basedir, 'v2', line))
                        else:
                            item_id = int(base_filename[:-4].split('-')[1])
                            if item_id in self.high_signal_item_ids:
                                self.items.append(os.path.join(self.basedir, 'v2', line))
                    else:
                        self.items.append(os.path.join(self.basedir, 'v2', line))

        self.noise_files = [f for f in self.items if 'Noise' in f]
        self.signal_files = [f for f in self.items if 'Signal' in f]

        # print('self.noise_files: {}'.format(self.noise_files))
        # print('self.signal_files: {}'.format(self.signal_files))

        self.n_noise_files = len(self.noise_files)
        self.n_signal_files = len(self.signal_files)

        self.label_weights = [1 / self.n_noise_files, 1 / self.n_signal_files]
        self.length = self.n_noise_files + self.n_signal_files

        if self.balance_data and not self.supersample:
            # self.noise_files = [f for f in self.items if 'Noise' in f]
            # self.signal_files = [f for f in self.items if 'Signal' in f]

            min_len = min(len(self.noise_files), len(self.signal_files))
            self.noise_files = self.noise_files[:min_len]
            self.signal_files = self.signal_files[:min_len]
            self.n_noise_files = len(self.noise_files)
            self.n_signal_files = len(self.signal_files)
            all_files = self.noise_files[:]
            all_files.extend(self.signal_files[:])
            self.items = all_files[:]
            self.length = self.n_noise_files + self.n_signal_files
        elif self.supersample:
            self.length = 2 * self.n_signal_files

        random.seed(self.args.seed)
        random.shuffle(self.noise_files)
        random.shuffle(self.signal_files)
        random.shuffle(self.items)

        # self.noise_data = [np.load(nf) for nf in self.noise_files]
        # self.signal_data = [np.load(sf) for sf in self.signal_files]
        self.noise_data = KeyDependentDict()
        self.signal_data = KeyDependentDict()

        # if self.only_noise:
        #     self.items = [f for f in self.items if 'Noise' in f]
        #
        # if self.only_signal:
        #     self.items = [f for f in self.items if 'Signal' in f]

        if self.undersample is not None:
            smpl_idx = int(len(self.items) * self.undersample)
            print('smpl idx: {}'.format(smpl_idx))
            self.items = self.items[:smpl_idx]

        self.CURR_EPOCH = 0
        self.CURR_ITEM = 0
        self.NOISE_IDX = 0
        self.SIGNAL_IDX = 0

        print('*** PhysicsDataset ***')
        print('*** Mode: {} ***'.format(self.mode))
        print('*** N Noise: {} ***'.format(self.n_noise_files))
        print('*** N Signal: {} ***'.format(self.n_signal_files))
        print('*** N Total: {} ***'.format(self.length))
        if self.do_frame_swap:
            print('*** do_frame_swap: {} ***'.format(self.do_frame_swap))
            print('*** epoch_to_start_frame_swaps: {} ***'.format(self.epoch_to_start_frame_swaps))
            print('*** pct_frames_to_swap: {} ***'.format(self.pct_frames_to_swap))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.CURR_ITEM += 1
        if self.CURR_ITEM == self.length:
            self.CURR_ITEM = 0
            self.CURR_EPOCH += 1

        # item_id = -1
        if self.supersample:
            if idx % 2 == 0:    # is a signal idx which will always map 1:1 to signal files
                effective_idx = idx // 2
                item_file = self.signal_files[effective_idx]
                # item_id = os.path.join('Signal', os.path.basename(item_file))
                item_data = self.signal_data[item_file]
                if self.do_frame_swap and self.CURR_EPOCH >= self.epoch_to_start_frame_swaps:
                    swap_file = random.choice(self.noise_files)
                    swap_data = self.noise_data[swap_file]
            else:
                # effective_idx = random.randint(0, self.n_noise_files - 1)
                item_file = self.noise_files[self.NOISE_IDX]
                # item_id = os.path.join('Noise', os.path.basename(item_file))
                item_data = self.noise_data[item_file]
                self.NOISE_IDX += 1
                if self.NOISE_IDX == self.n_noise_files:
                    self.NOISE_IDX = 0
                if self.do_frame_swap and self.CURR_EPOCH >= self.epoch_to_start_frame_swaps:
                    swap_file = random.choice(self.signal_files)
                    swap_data = self.signal_data[swap_file]
        else:
            item_file = self.items[idx]
            item_data = np.load(item_file)

        item_id = os.path.basename(item_file)

        # item_data = np.load(item_file)
        item_data_raw = item_data[:, :, :]
        item_label = np.array([0 if 'Noise' in item_file else 1])

        if self.do_frame_swap and self.CURR_EPOCH >= self.epoch_to_start_frame_swaps and self.supersample:
            # swap_data = np.load(swap_file)
            n_frames_to_swap = int(item_data.shape[0] * self.pct_frames_to_swap)
            frame_idx_to_swap = random.sample(range(0, item_data.shape[0] - 1), n_frames_to_swap)
            item_data[frame_idx_to_swap] = swap_data[frame_idx_to_swap]

        if self.normalize_data:
            # max_val = getattr(self.args, 'norm_max', 124)
            if self.norm_by_frame:
                tmp_data = item_data.reshape(item_data.shape[0], -1)
                max_val = np.max(tmp_data, axis=-1).reshape((-1, 1, 1))
            else:
                max_val = np.max(item_data)

            # item_data = item_data / max_val
            item_data = np.divide(item_data, max_val)

        item_data = torch.from_numpy(item_data)
        item_label = torch.from_numpy(item_label)

        item_transforms = []
        if random.random() >= self.transform_prob and self.do_transforms:   # reverse temporal order of frames
            item_data = torch.flip(item_data, [0])
            item_transforms.append('temporal')

        if random.random() >= self.transform_prob and self.do_transforms:   # swap frames vertically
            item_data = torch.flip(item_data, [1])
            item_transforms.append('vertical')

        if random.random() >= self.transform_prob and self.do_transforms:   # swap frames horizontally
            item_data = torch.flip(item_data, [2])
            item_transforms.append('horizontal')

        if len(item_transforms) > 0:
            item_transforms = '&'.join(item_transforms)
        else:
            item_transforms = 'none'

        out = {'data': item_data, 'label': item_label, 'data_raw': item_data_raw, 'item_id': item_id,
               'item_transforms': item_transforms}

        return out


def read_data_files(file_list, max_val_cutoff=-1):
    data = []

    for file in file_list:
        data_mat = np.load(file)
        x = data_mat.shape[1]
        y = data_mat.shape[2]
        for t in range(data_mat.shape[0]):
            frame_data = data_mat[t, :, :].reshape(x, y)
            frame_max = frame_data.max()
            if frame_max >= max_val_cutoff:
                data.append([frame_data, 0 if 'Noise' in file else 1])

    return data


class PhysicsFramesDataset(Dataset):
    def __init__(self, mode, basedir, args, only_noise=False, only_signal=False, undersample=None,
                 force_unbalanced=False, disallow_supersample=False):
        random.seed(16)
        self.mode = mode
        self.basedir = basedir
        self.args = args
        self.balance_data = getattr(self.args, 'balance_data', False) and not force_unbalanced
        self.only_noise = only_noise
        self.only_signal = only_signal
        self.undersample = undersample
        self.only_high_signal = getattr(self.args, 'only_high_signal', False)  # and self.mode == 'train'
        self.high_signal_file = '../listOfEventsWithHighSignal.npy'
        self.supersample = getattr(self.args, 'supersample', False) and not disallow_supersample and self.balance_data

        self.do_transforms = getattr(self.args, 'do_transforms', False) and self.mode == 'train'
        self.transform_prob = getattr(self.args, 'transform_prob', 0.5)

        self.high_signal_item_ids = []
        if self.only_high_signal:
            self.high_signal_item_ids = np.load(self.high_signal_file).tolist()

        print('PhysicsFramesDataset.basedir: {}'.format(self.basedir))

        self.items = []
        if self.mode == 'train':
            v2_filename = os.path.join(self.basedir, 'v2', 'train_files.txt')
            with open(v2_filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if 'Noise' in line:
                        self.items.append(os.path.join(self.basedir, 'v2', line))

            v1_filename = os.path.join(self.basedir, 'v1', 'train_files.txt')
            with open(v1_filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if 'Signal' in line:
                        line = line.split(os.path.sep)[-2:]
                        self.items.append(os.path.join(self.basedir, 'v1', line[0], line[1]))
        elif self.mode == 'dev':
            v2_filename = os.path.join(self.basedir, 'v2', 'dev_files.txt')
            with open(v2_filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if self.only_high_signal:
                        type_dir, base_filename = os.path.split(line)
                        if type_dir == 'Noise':
                            self.items.append(os.path.join(self.basedir, 'v2', line))
                        else:
                            item_id = int(base_filename[:-4].split('-')[1])
                            if item_id in self.high_signal_item_ids:
                                self.items.append(os.path.join(self.basedir, 'v2', line))
                    else:
                        self.items.append(os.path.join(self.basedir, 'v2', line))

        self.noise_files = [f for f in self.items if 'Noise' in f]
        self.signal_files = [f for f in self.items if 'Signal' in f]

        if self.undersample is not None:
            smpl_idx_noise = int(len(self.noise_files) * self.undersample)
            smpl_idx_signal = int(len(self.signal_files) * self.undersample)
            random.shuffle(self.noise_files)
            random.shuffle(self.signal_files)
            self.noise_files = self.noise_files[:smpl_idx_noise]
            self.signal_files = self.signal_files[:smpl_idx_signal]

        self.n_noise_files = len(self.noise_files)
        self.n_signal_files = len(self.signal_files)

        self.label_weights = [1 / self.n_noise_files, 1 / self.n_signal_files]
        self.length = self.n_noise_files + self.n_signal_files

        if self.balance_data and not self.supersample:
            # self.noise_files = [f for f in self.items if 'Noise' in f]
            # self.signal_files = [f for f in self.items if 'Signal' in f]

            min_len = min(len(self.noise_files), len(self.signal_files))
            self.noise_files = self.noise_files[:min_len]
            self.signal_files = self.signal_files[:min_len]
            self.n_noise_files = len(self.noise_files)
            self.n_signal_files = len(self.signal_files)
            all_files = self.noise_files[:]
            all_files.extend(self.signal_files[:])
            self.items = all_files[:]
            self.length = self.n_noise_files + self.n_signal_files
        elif self.supersample:
            self.length = 2 * self.n_signal_files

        random.seed(self.args.seed)
        random.shuffle(self.noise_files)
        random.shuffle(self.signal_files)
        random.shuffle(self.items)

        # self.noise_data = [np.load(nf) for nf in self.noise_files]
        # self.signal_data = [np.load(sf) for sf in self.signal_files]
        self.noise_data = KeyDependentDict()
        self.signal_data = KeyDependentDict()

        if self.undersample is not None:
            smpl_idx = int(len(self.items) * self.undersample)
            print('smpl idx: {}'.format(smpl_idx))
            self.items = self.items[:smpl_idx]

        self.CURR_EPOCH = 0
        self.CURR_ITEM = 0
        self.NOISE_IDX = 0
        self.SIGNAL_IDX = 0

        print('*** PhysicsFrameDataset ***')
        print('*** Mode: {} ***'.format(self.mode))
        print('*** N Noise: {} ***'.format(self.n_noise_files))
        print('*** N Signal: {} ***'.format(self.n_signal_files))
        print('*** N Total: {} ***'.format(len(self.items)))

        # self.data_file = os.path.join(self.basedir, '{}_files.txt'.format(self.mode if 'eval' not in self.mode
        #                                                                   else 'dev'))
        # self.data_files = []
        #
        # with open(self.data_file, 'r') as f:
        #     for line in f:
        #         line = line.strip()
        #         self.data_files.append(os.path.join(self.basedir, line))     # add filepath to items
        #
        # noise_files = [f for f in self.data_files if 'Noise' in f]
        # signal_files = [f for f in self.data_files if 'Signal' in f]
        #
        # if self.undersample is not None:
        #     smpl_idx_noise = int(len(noise_files) * self.undersample)
        #     smpl_idx_signal = int(len(signal_files) * self.undersample)
        #     random.shuffle(noise_files)
        #     random.shuffle(signal_files)
        #     noise_files = noise_files[:smpl_idx_noise]
        #     signal_files = signal_files[:smpl_idx_signal]
        #
        # noise_items = read_data_files(noise_files)
        # signal_items = read_data_files(signal_files, max_val_cutoff=5)
        # self.label_weights = [1 / len(noise_items), 1 / len(signal_items)]
        # n_noise = len(noise_items)
        # n_signal = len(signal_items)
        #
        # if self.balance_data:
        #     min_len = min(len(noise_items), len(signal_items))
        #     noise_items = noise_items[:min_len]
        #     signal_items = signal_items[:min_len]
        #     n_noise = len(noise_items)
        #     n_signal = len(signal_items)
        #     all_items = noise_items[:]
        #     all_items.extend(signal_items[:])
        #     self.items = all_items[:]
        # else:
        #     self.items.extend(noise_items[:])
        #     self.items.extend(signal_items[:])
        #
        # random.shuffle(self.items)

    def __len__(self):
        # return len(self.items)
        return self.length

    def __getitem__(self, idx):
        self.CURR_ITEM += 1
        if self.CURR_ITEM == self.length:
            self.CURR_ITEM = 0
            self.CURR_EPOCH += 1

        # item_id = -1
        if self.supersample:
            if idx % 2 == 0:  # is a signal idx which will always map 1:1 to signal files
                effective_idx = idx // 2
                item_file = self.signal_files[effective_idx]
                # item_id = os.path.join('Signal', os.path.basename(item_file))
                item_data = self.signal_data[item_file]
            else:
                # effective_idx = random.randint(0, self.n_noise_files - 1)
                item_file = self.noise_files[self.NOISE_IDX]
                # item_id = os.path.join('Noise', os.path.basename(item_file))
                item_data = self.noise_data[item_file]
                self.NOISE_IDX += 1
                if self.NOISE_IDX == self.n_noise_files:
                    self.NOISE_IDX = 0
        else:
            item_file = self.items[idx]
            item_data = np.load(item_file)

        item_id = os.path.basename(item_file)

        # item_data = np.load(item_file)
        # item_data_raw = item_data[:, :, :]
        seq_len = item_data.shape[0]
        item_label = torch.zeros(seq_len, dtype=torch.long) if 'Noise' in item_file \
            else torch.ones(seq_len, dtype=torch.long)
        item_data = torch.from_numpy(item_data)

        item_transforms = []
        if random.random() >= self.transform_prob and self.do_transforms:   # swap frames vertically
            item_data = torch.flip(item_data, [1])
            item_transforms.append('vertical')

        if random.random() >= self.transform_prob and self.do_transforms:   # swap frames horizontally
            item_data = torch.flip(item_data, [2])
            item_transforms.append('horizontal')

        if len(item_transforms) > 0:
            item_transforms = '&'.join(item_transforms)
        else:
            item_transforms = 'none'

        out = {'data': item_data, 'label': item_label, 'item_id': item_id,
               'item_transforms': item_transforms}

        return out



