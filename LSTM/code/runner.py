__author__ = 'Connor Heaton'

import os
import math
import time
import torch

import numpy as np
import torch.distributed as dist

from sklearn.metrics import confusion_matrix

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from models import ConvRNNModel, ConvModel
from datasets import PhysicsDataset, PhysicsFramesDataset
from evaluate_predictions import calc_metrics


def calc_accuracy(logits, labels, sources=None):
    labels = labels.reshape(-1)
    logit_max = np.argmax(logits, axis=-1)

    accuracy = np.mean(labels == logit_max)

    if sources is not None:
        src_accs = []
        unique_sources = list(np.unique(sources))
        sources = np.array(sources) if type(sources) in [list, tuple] else sources
        sources = sources.reshape(-1)

        for src in unique_sources:
            src_idx = np.where(sources == src)
            src_lbls = labels[src_idx]
            src_preds = logit_max[src_idx]

            src_acc = np.mean(src_lbls == src_preds)
            src_accs.append([src, src_acc])

        return accuracy, src_accs

    return accuracy


class Runner(object):
    def __init__(self, gpu, mode, args):
        self.rank = gpu
        self.mode = mode
        self.args = args

        self.lr = getattr(self.args, 'lr', 1e-5)
        self.l2 = getattr(self.args, 'l2', 0.0001)
        self.model_type = getattr(self.args, 'model_type', 'ConvRNN')

        if self.args.on_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.rank))

        print('self.device: {}'.format(self.device))
        print('torch.cuda.device_count(): {}'.format(torch.cuda.device_count()))
        dist.init_process_group('nccl',
                                world_size=len(self.args.gpus),
                                rank=self.rank)

        torch.manual_seed(self.args.seed)

        if self.rank == 0:
            print('Creating model...')

        if self.model_type == 'CNN':
            self.model = ConvModel(self.args, self.device).to(self.device)
        else:
            self.model = ConvRNNModel(self.args, self.device).to(self.device)

        if not self.args.on_cpu:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank],
                                                                   find_unused_parameters=False)

        ckpt_file = None
        if self.mode != 'train':
            # ckpt_file = self.args.out_dir
            if self.args.train:
                ckpt_epoch_offset = 1
                ckpt_file = os.path.join(self.args.model_save_dir,
                                         self.args.ckpt_file_tmplt.format(self.args.epochs - ckpt_epoch_offset))
                while not os.path.exists(ckpt_file) and self.args.epochs - ckpt_epoch_offset >= 0:
                    ckpt_epoch_offset += 1
                    ckpt_file = os.path.join(self.args.model_save_dir,
                                             self.args.ckpt_file_tmplt.format(self.args.epochs - ckpt_epoch_offset))

            else:
                ckpt_file = self.args.ckpt_file

        if ckpt_file is not None:
            if self.rank == 0:
                print('Loading model from checkpoint...')
            map_location = {'cuda:{}'.format(0): 'cuda:{}'.format(gpu_id) for gpu_id in args.gpus}
            state_dict = torch.load(ckpt_file, map_location=map_location)
            self.model.load_state_dict(state_dict,strict=False)  # , strict=False

        if self.rank == 0:
            print('Loading dataset...')

        dataset_start_time = time.time()

        if self.model_type == 'CNN':
            frame_undersample_p = getattr(self.args, 'frame_undersample_p', 0.25)
            frame_undersample_p = None if frame_undersample_p == 1.0 else frame_undersample_p
            self.dataset = PhysicsFramesDataset(self.mode, self.args.data, self.args, undersample=frame_undersample_p)
        else:
            self.dataset = PhysicsDataset(self.mode, self.args.data, self.args)

        if self.rank == 0:
            print('Time to read dataset: {0:.2f}s'.format(time.time() - dataset_start_time))

        if getattr(self.args, 'weight_xent_loss', False):
            self.label_weights = self.dataset.label_weights
        else:
            self.label_weights = [1.0, 1.0]
        # if self.rank == 0:
        #     print('Setting label weights to {}...'.format(self.label_weights))
        # self.model.module.set_label_weights(self.label_weights)

        if self.args.on_cpu:
            data_sampler = None
        else:
            data_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,
                                                                           num_replicas=args.world_size,
                                                                           rank=self.rank,
                                                                           shuffle=True if self.mode == 'train'
                                                                           else False)

        self.data_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=False,
                                      num_workers=self.args.n_data_workers, pin_memory=True, sampler=data_sampler)
        self.n_iters = int(math.ceil(len(self.dataset) / (self.args.batch_size * len(self.args.gpus))))

        self.schedule_weight_xent_loss = getattr(self.args, 'schedule_weight_xent_loss', False) and self.mode == 'train'
        self.schedule_weight_xent_loss_epoch_start = getattr(self.args, 'schedule_weight_xent_loss_epoch_start', 0)
        self.schedule_weight_xent_loss_epoch_end = getattr(self.args, 'schedule_weight_xent_loss_epoch_end', 10)
        self.schedule_weight_xent_loss_noise_start_weight = getattr(self.args,
                                                                    'schedule_weight_xent_loss_noise_start_weight',
                                                                    0.9)
        self.schedule_weight_xent_loss_noise_final_weight = getattr(self.args,
                                                                    'schedule_weight_xent_loss_noise_final_weight',
                                                                    0.5)
        self.iters_w_scheduled_xent = (self.schedule_weight_xent_loss_epoch_end -
                                       self.schedule_weight_xent_loss_epoch_start) * self.n_iters

        self.aux_dataset = None
        self.aux_data_loader = None
        self.aux_n_iters = None

        if self.mode == 'train' and (self.args.eval or self.args.eval_every > 0):
            if self.rank == 0:
                print('Loading aux dataset...')

            if self.model_type == 'CNN':
                frame_undersample_p = getattr(self.args, 'frame_undersample_p', 1.0)
                frame_undersample_p = None if frame_undersample_p == 1.0 else frame_undersample_p
                self.aux_dataset = PhysicsFramesDataset('dev', self.args.data, self.args,
                                                        force_unbalanced=True,
                                                        undersample=frame_undersample_p)
            else:
                self.aux_dataset = PhysicsDataset('dev', self.args.data, self.args, force_unbalanced=True,
                                                  disallow_supersample=True)

            if self.args.on_cpu:
                aux_data_sampler = None
            else:
                aux_data_sampler = torch.utils.data.distributed.DistributedSampler(self.aux_dataset,
                                                                                   num_replicas=args.world_size,
                                                                                   rank=self.rank)
            self.aux_data_loader = DataLoader(self.aux_dataset, batch_size=self.args.batch_size, shuffle=False,
                                              num_workers=self.args.n_data_workers, pin_memory=True,
                                              sampler=aux_data_sampler)
            self.aux_n_iters = int(math.ceil(len(self.aux_dataset) / (self.args.batch_size * len(self.args.gpus))))

        self.summary_writer = None
        if self.rank == 0 and self.mode == 'train':
            self.summary_writer = SummaryWriter(log_dir=self.args.tb_dir)

        self.n_epochs = 1
        if self.mode == 'train':
            self.n_epochs = self.args.epochs

            opt_parms = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = optim.Adam(opt_parms, lr=self.lr, weight_decay=self.l2)
            n_total_iters = self.n_iters * self.n_epochs
            n_warmup_steps = self.args.warmup_proportion * n_total_iters
            if n_warmup_steps > 0:  # get_linear_schedule_with_warmup
                self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                                 num_warmup_steps=n_warmup_steps,
                                                                 num_training_steps=n_total_iters,
                                                                 num_cycles=0.33333)
                # self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                #                                                  num_warmup_steps=n_warmup_steps,
                #                                                  num_training_steps=n_total_iters)
            else:
                self.scheduler = None

        self.run()

    def run(self):
        for epoch in range(self.n_epochs):
            if self.rank == 0:
                print('Performing epoch {} of {}'.format(epoch, self.n_epochs))

            if self.mode == 'train':
                self.model.train()
            else:
                self.model.eval()

            if self.mode == 'eval':
                with torch.no_grad():
                    self.run_one_epoch(epoch, self.mode)
            else:
                self.run_one_epoch(epoch, self.mode)

            if self.mode == 'train':
                if self.rank == 0:
                    print('Saving model...')
                    torch.save(self.model.state_dict(),
                               os.path.join(self.args.model_save_dir, self.args.ckpt_file_tmplt.format(epoch)))
                dist.barrier()

                if self.args.eval_every > 0 and epoch % self.args.eval_every == 0:
                    self.model.eval()
                    with torch.no_grad():
                        self.run_one_epoch(epoch, 'train-eval')

    def run_one_epoch(self, epoch, mode):
        if mode == self.mode:
            dataset = self.data_loader
            n_iters = self.n_iters
            n_samples = len(self.dataset)
        else:
            dataset = self.aux_data_loader
            n_iters = self.aux_n_iters
            n_samples = len(self.aux_dataset)

        accuracies = []
        all_logits = []
        all_labels = []
        all_ids = []
        all_transforms = []
        iter_since_grad_accum = 1
        for batch_idx, batch_data in enumerate(dataset):
            batch_start_time = time.time()

            batch_x = batch_data['data'].float().to(self.device, non_blocking=True)
            batch_y = batch_data['label'].to(self.device, non_blocking=True).view(-1)
            batch_item_ids = batch_data['item_id']
            batch_item_transforms = batch_data['item_transforms']
            if not self.model_type == 'CNN':
                batch_x_raw = batch_data['data_raw'].float().to(self.device, non_blocking=True)
            else:
                batch_x_raw = None
                batch_size = batch_x.shape[0]
                seq_len = batch_x.shape[-3]
                dim_x = batch_x.shape[-2]
                dim_y = batch_x.shape[-1]
                batch_x = batch_x.view(-1, dim_x, dim_y)

                tmp_info = list(zip(batch_item_ids, batch_item_transforms))
                batch_info = []
                for ti in tmp_info:
                    for sl in range(seq_len):
                        batch_info.append(ti)

                batch_item_ids, batch_item_transforms = map(list, zip(*batch_info))

            batch_xent_weight = [1.0, 1.0]
            if self.schedule_weight_xent_loss and mode == 'train':
                if epoch < self.schedule_weight_xent_loss_epoch_start:
                    batch_xent_weight = [self.schedule_weight_xent_loss_noise_start_weight, 1.0]
                elif epoch > self.schedule_weight_xent_loss_epoch_end:
                    batch_xent_weight = [self.schedule_weight_xent_loss_noise_final_weight, 1.0]
                else:
                    iters_into_schedule = (epoch - self.schedule_weight_xent_loss_epoch_start) * n_iters + batch_idx
                    pct_into_schedule = iters_into_schedule / self.iters_w_scheduled_xent
                    pct_schedule_remaining = (self.iters_w_scheduled_xent - iters_into_schedule) / \
                                             self.iters_w_scheduled_xent

                    orig_weight = self.schedule_weight_xent_loss_noise_start_weight * pct_schedule_remaining
                    new_weight = self.schedule_weight_xent_loss_noise_final_weight * pct_into_schedule
                    batch_noise_weight = orig_weight + new_weight
                    batch_xent_weight = [batch_noise_weight, 1.0]

            batch_xent_weight = torch.tensor(batch_xent_weight)

            if batch_idx == 0 and self.rank == 0:
                # print('batch_x: {}'.format(batch_x))
                print('batch_x shape: {}\tmin: {}\tmax: {}'.format(batch_x.shape,
                                                                   torch.min(batch_x),
                                                                   torch.max(batch_x)))
                # print('batch_y: {}'.format(batch_y))
                print('batch_y shape: {}'.format(batch_y.shape))

            batch_outputs = self.model(batch_x, batch_y, x_raw=batch_x_raw, batch_xent_weight=batch_xent_weight)

            loss = batch_outputs[0]
            logits = batch_outputs[1]

            if batch_idx == 0 and self.rank == 0:
                print('logits shape: {}'.format(logits.shape))

            if mode == 'train':
                loss.backward()

            logits = logits.detach().to('cpu').numpy()
            labels = batch_y.to('cpu').numpy()

            batch_acc = calc_accuracy(logits, labels)
            accuracies.append(batch_acc)

            # if not mode == 'test':
            these_labels = list(labels.reshape(-1))
            these_logits = [list(l) for l in list(logits)]

            all_labels.extend(these_labels)
            all_logits.extend(these_logits)
            all_ids.extend(batch_item_ids)
            all_transforms.extend(batch_item_transforms)

            if batch_idx % self.args.grad_summary_every == 0 and self.summary_writer is not None and mode == 'train' \
                    and self.args.grad_summary:
                for name, p in self.model.named_parameters():
                    self.summary_writer.add_histogram('grad/{}'.format(name), p.grad.data,
                                                      (epoch * n_iters) + batch_idx)
                    self.summary_writer.add_histogram('weight/{}'.format(name), p.data, (epoch * n_iters) + batch_idx)

            if batch_idx % self.args.summary_every == 0 and self.summary_writer is not None:
                self.summary_writer.add_scalar('loss/{}'.format(mode), loss, (epoch * n_iters) + batch_idx)
                self.summary_writer.add_scalar('acc/{}'.format(mode), batch_acc, (epoch * n_iters) + batch_idx)
            batch_elapsed_time = time.time() - batch_start_time
            if batch_idx % self.args.print_every == 0 and self.rank == 0:
                print_str = 'Mode: {5} Epoch: {0}/{1} Iter: {2}/{3} Loss: {4:2.4f} Accuracy: {6:3.2f}%'.format(epoch,
                                                                                                               self.n_epochs,
                                                                                                               batch_idx,
                                                                                                               n_iters,
                                                                                                               loss,
                                                                                                               mode,
                                                                                                               batch_acc * 100)

                print_str = '{0} Xent weight: {1}'.format(print_str, batch_xent_weight)
                print_str = '{0} Time: {1:3.2f}s'.format(print_str, batch_elapsed_time)
                if self.rank == 0:
                    print(print_str)

            if iter_since_grad_accum == self.args.n_grad_accum and mode == 'train':
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                iter_since_grad_accum = 1
            else:
                iter_since_grad_accum += 1

        if iter_since_grad_accum > 1 and mode == 'train':
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.rank == 0:
            # avg_acc = np.mean(accuracies)
            all_preds = [0 if logit[0] > logit[1] else 1 for logit in all_logits]
            mat = confusion_matrix(all_labels, all_preds)
            print(mat)
            tpr, tnr, fnr, fpr, acc = calc_metrics(mat)
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('metrics/{}/{}'.format(mode, 'tpr'), tpr, epoch)
                self.summary_writer.add_scalar('metrics/{}/{}'.format(mode, 'tnr'), tnr, epoch)
                self.summary_writer.add_scalar('metrics/{}/{}'.format(mode, 'fnr'), fnr, epoch)
                self.summary_writer.add_scalar('metrics/{}/{}'.format(mode, 'fpr'), fpr, epoch)

            print('*' * 50)
            print('\tEpoch: {}'.format(epoch))
            print('\tMode: {}'.format(mode))
            print('\tAvg acc: {0:3.2f}%'.format(acc * 100))
            print('\tTPR: {0:3.24f}'.format(tpr))
            print('\tTNR: {0:3.24f}'.format(tnr))
            print('\tFNR: {0:3.24f}'.format(fnr))
            print('\tFPR: {0:3.24f}'.format(fpr))
            print('*' * 50)

        if len(all_labels) > 0:
            print('Writing preds to file...')
            if mode == 'eval':
                pred_file = os.path.join(self.args.out, 'eval_preds.csv')
            else:
                pred_file = os.path.join(self.args.out, '{}_preds_{}.csv'.format(mode, epoch))

            with open(pred_file, 'w+') as f:
                for rec_id, trans, rec_logits, lbl in zip(all_ids, all_transforms, all_logits, all_labels):
                    f.write('{},{},{},{}\n'.format(rec_id, trans, ','.join([str(l) for l in rec_logits]), lbl))
