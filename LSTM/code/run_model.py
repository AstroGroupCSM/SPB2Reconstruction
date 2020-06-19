__author__ = 'Connor Heaton'

import argparse
import datetime
import torch
import glob
import sys
import os

import numpy as np
import torch.multiprocessing as mp

from runner import Runner


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='../data', help='Dir where data can be found')
    parser.add_argument('--out', default='../out', help='Directory to put output')

    parser.add_argument('--train', default=False, type=str2bool)
    parser.add_argument('--eval', default=False, type=str2bool)
    parser.add_argument('--test', default=False, type=str2bool)

    # model parms
    parser.add_argument('--model_type', default='ConvRNN')
    parser.add_argument('--epochs', default=10, type=int, help='# epochs to train for')
    parser.add_argument('--batch_size', default=48, type=int, help='Batch size to use')
    parser.add_argument('--lr', default=5e-5, type=float, help='Learning rate')

    parser.add_argument('--res_conv_kernel_size', default=3, type=int)
    parser.add_argument('--res_conv_stride', default=1, type=int)
    parser.add_argument('--conv_kernel_size', default=3, type=int)
    parser.add_argument('--conv_stride', default=1, type=int)
    parser.add_argument('--res_pool_kernel_size', default=3, type=int)
    parser.add_argument('--res_pool_stride', default=1, type=int)
    parser.add_argument('--pool_kernel_size', default=3, type=int)
    parser.add_argument('--pool_stride', default=1, type=int)
    parser.add_argument('--conv_dropout_prob', default=0.2, type=float)
    parser.add_argument('--conv_channels', default=1, type=int)
    parser.add_argument('--n_conv_layers', default=12, type=int)

    parser.add_argument('--rnn_dim', default=64, type=int)
    parser.add_argument('--rnn_input_dropout_prob', default=0.30, type=float)
    parser.add_argument('--rnn_output_dropout_prob', default=0.30, type=float)
    parser.add_argument('--linear_projection_dropout_prob', default=0.30, type=float)

    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--n_grad_accum', default=1, type=int)

    parser.add_argument('--l2', default=0.0001, type=float)

    parser.add_argument('--weight_xent_loss', default=False, type=str2bool)
    parser.add_argument('--noise_loss_weight', default=1.0, type=float)
    parser.add_argument('--signal_loss_weight', default=1.0, type=float)

    parser.add_argument('--schedule_weight_xent_loss', default=False, type=str2bool)
    parser.add_argument('--schedule_weight_xent_loss_epoch_start', default=0, type=int)
    parser.add_argument('--schedule_weight_xent_loss_epoch_end', default=10, type=int)
    parser.add_argument('--schedule_weight_xent_loss_noise_start_weight', default=0.9, type=float)
    parser.add_argument('--schedule_weight_xent_loss_noise_final_weight', default=0.5, type=float)

    parser.add_argument('--use_frame_loss', default=False, type=str2bool)
    parser.add_argument('--frame_loss_weight', default=0.25, type=float)
    parser.add_argument('--frame_undersample_p', default=1.0, type=float)

    parser.add_argument('--frame_clf_dropout_p', default=0.2, type=float)

    # data parms
    parser.add_argument('--balance_data', default=False, type=str2bool)
    parser.add_argument('--normalize_data', default=False, type=str2bool)
    parser.add_argument('--norm_max', default=124, type=int)
    parser.add_argument('--norm_by_frame', default=False, type=str2bool)
    parser.add_argument('--supersample', default=False, type=str2bool)
    parser.add_argument('--only_high_signal', default=False, type=str2bool)

    # fun data parms
    parser.add_argument('--do_transforms', default=False, type=str2bool)
    parser.add_argument('--transform_prob', default=0.5, type=float)
    parser.add_argument('--do_frame_swaps', default=False, type=str2bool)
    parser.add_argument('--epoch_to_start_frame_swaps', default=-1, type=int)
    parser.add_argument('--pct_frames_to_swap', default=0.1, type=float)

    # logging parms
    parser.add_argument('--seed', default=16, type=int)
    parser.add_argument('--ckpt_file', default=None)
    parser.add_argument('--ckpt_file_tmplt', default='model_{}e.pt')
    parser.add_argument('--print_every', default=1, type=int)
    parser.add_argument('--summary_every', default=15, type=int)
    parser.add_argument('--eval_every', default=1, type=int)
    parser.add_argument('--arg_out_file', default='args.txt', help='File to write cli args to')
    parser.add_argument('--verbosity', default=0, type=int)

    parser.add_argument('--grad_summary', default=True, type=str2bool)
    parser.add_argument('--grad_summary_every', default=100, type=int)

    # hardware parms
    parser.add_argument('--gpus', default=[0], help='Which GPUs to use', type=int, nargs='+')
    parser.add_argument('--port', default='12345', help='Port to use for DDP')
    parser.add_argument('--on_cpu', default=False, type=str2bool)
    parser.add_argument('--n_data_workers', default=3, help='# threads used to fetch data')

    args = parser.parse_args()

    # post-processing of some args
    args.world_size = len(args.gpus)

    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)

    run_modes = []
    if args.train:
        run_modes.append('train')

        # directories should only be made if training new model
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print('*' * len('* Model time ID: {} *'.format(curr_time)))
        print('* Model time ID: {} *'.format(curr_time))
        print('*' * len('* Model time ID: {} *'.format(curr_time)))

        args.out = os.path.join(args.out, curr_time)
        os.makedirs(args.out)

        args.tb_dir = os.path.join(args.out, 'tb_dir')
        os.makedirs(args.tb_dir)

        args.model_save_dir = os.path.join(args.out, 'models')
        os.makedirs(args.model_save_dir)

        args.arg_out_file = os.path.join(args.out, args.arg_out_file)
        args_d = vars(args)
        with open(args.arg_out_file, 'w+') as f:
            for k, v in args_d.items():
                f.write('{} = {}\n'.format(k, v))
    if args.eval:
        run_modes.append('eval')
    if args.test:
        run_modes.append('test')

    if (args.eval or args.test) and not args.train:
        args.out = os.path.dirname(args.ckpt_file)
        if os.path.basename(args.out) == 'models':
            args.out = os.path.dirname(args.out)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    print('*' * 40)
    print('sys.version: {}'.format(sys.version))
    print('torch.cuda.device_count(): {}'.format(torch.cuda.device_count()))
    print('torch version: {}'.format(torch.__version__))
    print('*' * 40)

    # for each mode, run the model. Hoping to be able to use the same call, just change mode?
    for mode in run_modes:
        print('Creating {} distributed models for {}...'.format(len(args.gpus), mode))
        mp.spawn(Runner, nprocs=len(args.gpus), args=(mode, args))

    print('Finished!')