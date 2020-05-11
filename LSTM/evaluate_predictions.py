__author__ = 'Connor Heaton'


import os
import glob
import math
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from subprocess import call
from collections import defaultdict

from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_preds(predfile):
    labels = []
    logits = []
    preds = []
    with open(predfile, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split(',')

            these_logits = [float(v) for v in line[:2]]
            lbl = float(line[2])

            pred = 0 if these_logits[0] > these_logits[1] else 1

            labels.append(lbl)
            logits.append(these_logits)
            preds.append(pred)

    return labels, logits, preds


def calc_metrics(x):
    tn = x[0, 0]
    fn = x[1, 0]
    fp = x[0, 1]
    tp = x[1, 1]

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)

    acc = (tn + tp) / (tn + fn + fp + tp)

    return tpr, tnr, fnr, fpr, acc


def plot_confusion_matrix(cm, classes, plot_file, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues, save_and_clear=True, xticks=True, yticks=True, colorbar=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools

    y_ticks = []
    for i, lbl in enumerate(classes):
        n_lbl = sum(cm[i, :])
        y_ticks.append('{} (n={})'.format(lbl, n_lbl))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    if colorbar:
        plt.colorbar()
    tick_marks = np.arange(len(classes))
    if xticks:
        plt.xticks(tick_marks, classes, rotation=-75)
    if yticks:
        plt.yticks(tick_marks, y_ticks)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i + 0.45, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=10, rotation=-45)
    if yticks:
        plt.ylabel('True label')
    if xticks:
        plt.xlabel('Predicted label')
    if save_and_clear:
        plt.tight_layout()
        plt.savefig(plot_file, bbox_inches='tight')
        plt.clf()


def plot_combined_confusion_matrices(pred_files, mode, args):
    basedir = args.preds

    if len(pred_files) > 1:
        adj_files = []
        for f in pred_files:
            epoch = f[:-4]
            epoch = int(epoch.split('_')[-1])
            adj_files.append([f, epoch])
        sorted_files = sorted(adj_files, key=lambda x: x[-1])
        pred_files = sorted_files[:]
    else:
        pred_files = [[x, 0] for x in pred_files]

    label_strs = ['Noise', 'Signal']
    fig = plt.figure()

    n_cols = int(math.ceil(len(pred_files) / 2))
    n_rows = 2

    for idx, (pred_file, epoch) in enumerate(pred_files):
        labels, logits, preds = read_preds(pred_file)

        mat = confusion_matrix(labels, preds)
        xticks = False if idx + 1 < n_cols else True
        yticks = True if idx + 1 == 1 or idx + 1 == n_cols + 1 else False
        colorbar = True if (idx + 1) % n_cols == 0 else False
        plt.subplot(n_rows, n_cols, idx + 1)
        plot_confusion_matrix(mat, label_strs, None, normalize=True, save_and_clear=False,
                              title='Epoch {}'.format(epoch), xticks=xticks, yticks=yticks, colorbar=colorbar)

    plt_file = os.path.join(basedir, '{}_confusion_matrix_combined.png'.format(mode))
    # plt.tight_layout(pad=3.0)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.2)
    plt.savefig(plt_file, bbox_inches='tight')
    plt.clf()


def evaluate_mode_preds(pred_files, mode, args):
    basedir = args.preds

    if len(pred_files) > 1:
        adj_files = []
        for f in pred_files:
            epoch = f[:-4]
            epoch = int(epoch.split('_')[-1])
            adj_files.append([f, epoch])
        sorted_files = sorted(adj_files, key=lambda x: x[-1])
        pred_files = sorted_files[:]
    else:
        pred_files = [[x, 0] for x in pred_files]

    colors = ['red', 'blue', 'purple', 'green', 'orange', 'coral', 'black', 'yellow', 'pink', 'cyan', 'lime', 'bisque']
    label_strs = ['Noise', 'Signal']
    metric_strs = ['TPR', 'TNR', 'FNR', 'FPR', 'ACC']
    all_metrics = []

    for pred_file, epoch in pred_files:
        labels, logits, preds = read_preds(pred_file)

        mat = confusion_matrix(labels, preds)
        metrics = calc_metrics(mat)
        all_metrics.append(metrics)

        plt_file = os.path.join(basedir, '{}_confusion_matrix_{}.png'.format(mode, epoch))
        plot_confusion_matrix(mat, label_strs, plt_file, normalize=True)

    if len(all_metrics) == 1:
        tpr, tnr, fnr, fpr, acc = all_metrics[0]
    else:
        tpr, tnr, fnr, fpr, acc = zip(*all_metrics)
    legend_patches = []

    for i, metric_values in enumerate([tpr, tnr, fnr, fpr, acc]):
        metric_values = list(metric_values)
        plt.plot(np.arange(len(metric_values)), metric_values, c=colors[i])
        metric_patch = mpatches.Patch(color=colors[i], label=metric_strs[i])
        legend_patches.append(metric_patch)

    metric_file = os.path.join(basedir, '{}_metrics.png'.format(mode))
    plt.legend(handles=legend_patches, loc='lower right')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.title('Prediction Performance')
    plt.savefig(metric_file, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--preds', default='../saved/')

    args = parser.parse_args()

    # labels = ['Noise', 'Signal']

    for mode in ['train-eval', 'eval', 'test']:
        glob_str = os.path.join(args.preds, '{}_preds_*.csv'.format(mode))
        print('{} glob str:{}'.format(mode, glob_str))
        pred_files = glob.glob(glob_str)
        if len(pred_files) > 0:
            print('\tFiles:\n\t{}'.format('\n\t'.join(pred_files)))

            evaluate_mode_preds(pred_files, mode, args)
            plot_combined_confusion_matrices(pred_files, mode, args)







