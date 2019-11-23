# -*- coding: utf-8 -*-

from __future__ import print_function, division
import inspect
import torch
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import os
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
import torch
import torch.nn as nn
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time
import pdb

class print_and_plot():
    def __init__(self, opt):
        super(print_and_plot, self).__init__()
        self.since_init = time.time()
        self.since_toc = self.since_init
        self.plot_init = True
        self.loss_record = {}
        self.acc_record = {}
        self.etc_record = {}
        self.cnt_record = []
        self.epoch_record = []
        # self.old_rank1 = 0
        self.map_total = 0
        self.total_best_k = 0
        self.total_best_idx = 0
        self.rank1_total = 0
        self.rank5_total = 0
        self.rank10_total = 0
        self.rank20_total = 0
        self.rank30_total = 0

        self.old_rank1 = []
        self.old_rank5 = []
        self.old_rank10 = []
        self.old_rank20 = []
        self.old_map = []
        self.best_epoch = []
        self.test_record = []
        for k in range(len(opt.evaluate_category)):
            self.old_rank1 += [0]
            self.old_rank5 += [0]
            self.old_rank10 += [0]
            self.old_rank20 += [0]
            self.old_map += [0]
            self.best_epoch += [0]
            self.test_record += [{}]
            self.test_record[k]['CMC'] = [np.zeros((100))]
            self.test_record[k]['MAP'] = [0]
            self.test_record[k]['EPOCH'] = [0]

    def print_info(self, opt, epoch, cnt, cnt_end_data, loss_type, acc_type, etc_type, cnt_cumul):

        epoch_init = epoch + (cnt + 1) / (cnt_end_data + 1)
        epoch_remaining = opt.num_epoch - epoch_init
        time_toc = time.time() - self.since_toc
        time_init = time.time() - self.since_init
        time_remaining = epoch_remaining / epoch_init * time_init
        time_init /= 60
        time_remaining /= 60
        print('[Epoch:{}] [{}/{}] [{}] '.format(epoch + 1, cnt + 1, cnt_end_data + 1, cnt_cumul), end='')
        print('[Tic: {:.0f}m {:.1f}s] [Start: {:.0f}h {:.0f}m, End: {:.0f}h {:.0f}m] '.format(
            time_toc // 60, time_toc % 60, time_init // 60, time_init % 60, time_remaining // 60,
            time_remaining % 60), end = '')
        print('Loss : ', end='')
        for key, value in loss_type.items():
            print('{} ({:.2f}), '.format(key, value), end='')
        print('Etc : ', end='')
        for key, value in etc_type.items():
            print('{} ({:.2f}), '.format(key, value), end='')
        print('Acc : ', end='')
        for key, value in acc_type.items():
            print('A_{} ({:.2f}%), '.format(key, value * 100), end='')
        print('')
        self.since_toc = time.time()

    def record_info(self, opt, phase, epoch, loss_type, acc_type, etc_type, cnt_cumul):

        if phase in opt.phase_train:
            for key, value in loss_type.items():
                try:
                    try:
                        self.loss_record[key].append(value.item())  # if error value.data[0]
                    except:
                        self.loss_record[key].append(value)
                except:
                    try:
                        self.loss_record[key] = [value.item()]
                    except:
                        self.loss_record[key] = [value]
            for key, value in acc_type.items():
                try:
                    try:
                        self.acc_record[key].append(value.item())
                    except:
                        self.acc_record[key].append(value)
                except:
                    try:
                        self.acc_record[key] = [value.item()]
                    except:
                        self.acc_record[key] = [value]
            for key, value in etc_type.items():
                # if value.iscuda: value = value.cpu()
                try:
                    try:
                        self.etc_record[key].append(value.item())
                    except:
                        self.etc_record[key].append(value)
                except:
                    try:
                        self.etc_record[key] = [value.item()]
                    except:
                        self.etc_record[key] = [value]
            self.cnt_record.append(cnt_cumul)
            self.epoch_record.append(epoch + 1)

    def plot_initialization(self, opt):
        if self.plot_init:
            self.plot_init = False

            self.plot_info = {}
            num_r = 5
            num_c = 5
            cnt = 0
            self.plot_info['loss'] = {}
            self.plot_info['loss']['fig'] = plt.figure(num=None, figsize=(20, 20), dpi=400, facecolor='w', edgecolor='k')
            if opt.flag_mean_plot:
                self.plot_info['mean_loss'] = {}
                self.plot_info['mean_loss']['fig'] = plt.figure(num=None, figsize=(20, 20), dpi=400, facecolor='w',
                                                           edgecolor='k')
            for key, value in self.loss_record.items():
                cnt += 1
                self.plot_info['loss'][key] = self.plot_info['loss']['fig'].add_subplot(num_r, num_c, cnt, title=key)
                if opt.flag_mean_plot:
                    self.plot_info['mean_loss'][key] = self.plot_info['mean_loss']['fig'].add_subplot(num_r, num_c, cnt, title=key)

            cnt = 0
            self.plot_info['acc'] = {}
            self.plot_info['acc']['fig'] = plt.figure(num=None, figsize=(20, 20), dpi=400, facecolor='w', edgecolor='k')


            if opt.flag_mean_plot:
                self.plot_info['mean_acc'] = {}
                self.plot_info['mean_acc']['fig'] = plt.figure(num=None, figsize=(20, 20), dpi=400, facecolor='w',
                                                          edgecolor='k')
            for key, value in self.acc_record.items():
                cnt += 1
                self.plot_info['acc'][key] = self.plot_info['acc']['fig'].add_subplot(num_r, num_c, cnt, title=key)
                if opt.flag_mean_plot:
                    self.plot_info['mean_acc'][key] = self.plot_info['mean_acc']['fig'].add_subplot(num_r, num_c, cnt, title=key)

            cnt = 0
            self.plot_info['etc'] = {}
            self.plot_info['etc']['fig'] = plt.figure(num=None, figsize=(20, 20), dpi=400, facecolor='w', edgecolor='k')


            if opt.flag_mean_plot:
                self.plot_info['mean_etc'] = {}
                self.plot_info['mean_etc']['fig'] = plt.figure(num=None, figsize=(20, 20), dpi=400, facecolor='w',
                                                          edgecolor='k')
            for key, value in self.etc_record.items():
                cnt += 1
                self.plot_info['etc'][key] = self.plot_info['etc']['fig'].add_subplot(num_r, num_c, cnt, title=key)
                if opt.flag_mean_plot:
                    self.plot_info['mean_etc'][key] = self.plot_info['mean_etc']['fig'].add_subplot(num_r, num_c, cnt, title=key)


            self.plot_info['test'] = []
            for k in range(len(opt.evaluate_category) + 1):
                self.plot_info['test'] += [{}]
                self.plot_info['test'][k]['fig'] = plt.figure(num=None, figsize=(20, 20), dpi=400, facecolor='w',
                                                              edgecolor='k')
                self.plot_info['test'][k]['MAP'] = self.plot_info['test'][k]['fig'].add_subplot(231, title='MAP')
                self.plot_info['test'][k]['Rank1'] = self.plot_info['test'][k]['fig'].add_subplot(232, title='Rank1')
                self.plot_info['test'][k]['Rank5'] = self.plot_info['test'][k]['fig'].add_subplot(233, title='Rank5')
                self.plot_info['test'][k]['Rank10'] = self.plot_info['test'][k]['fig'].add_subplot(234, title='Rank10')
                self.plot_info['test'][k]['Rank20'] = self.plot_info['test'][k]['fig'].add_subplot(235, title='Rank20')
                self.plot_info['test'][k]['Rank30'] = self.plot_info['test'][k]['fig'].add_subplot(236, title='Rank30')


    def draw_and_save_info(self, opt, epoch):
        time_start = time.time()

        current_epoch = epoch + 1
        folder_name = 'plot_results'
        if not os.path.isdir(os.path.join(opt.save_dir, folder_name)):
            os.mkdir(os.path.join(opt.save_dir, folder_name))
        x_epoch = []
        idx_epoch = []
        for i in range(current_epoch):
            x_epoch.append(i + 1)
            idx = np.argwhere(np.asarray(i + 1) == np.asarray(self.epoch_record))
            idx = idx.flatten()
            idx = idx.tolist()
            idx_epoch.append(idx)
        style_x = 'b-'
        style_xx = 'r-'
        x = 'loss'
        xx = 'mean_' + x

        if opt.flag_mean_plot:
            first_plot_xx = False
            if (len(idx_epoch[-1]) == 1) and (len(idx_epoch) > 1):
                flag_plot_xx = True
                if len(idx_epoch) == 2:
                    first_plot_xx = True
            else:
                flag_plot_xx = False
        else:
            flag_plot_xx = False
        for key, value in self.loss_record.items():
            self.plot_info[x][key].plot(self.cnt_record, self.loss_record[key], style_x, label=key)
            self.plot_info[x][key].title.set_text(key + ' : ' + str(round(self.loss_record[key][-1], 4)))
            if flag_plot_xx:
                mean_val = []
                for i in range(current_epoch):
                    mean_val.append(np.mean(self.loss_record[key][int(idx_epoch[i][0]):int(idx_epoch[i][-1]) + 1]))
                self.plot_info[xx][key].plot(x_epoch[:-1], mean_val[:-1], style_xx, label=key)
                self.plot_info[xx][key].title.set_text(key + ' : ' + str(round(mean_val[-1], 4)))
                if first_plot_xx:
                    self.plot_info[xx][key].legend()
            # if current_epoch == 1:
            #     self.plot_info[x][key].legend()
        self.plot_info[x]['fig'].savefig(os.path.join(opt.save_dir, folder_name, 'plot_' + x + '.jpg'))
        plt.close(self.plot_info[x]['fig'])
        if flag_plot_xx:
            self.plot_info[xx]['fig'].savefig(os.path.join(opt.save_dir, folder_name, 'plot_' + xx + '.jpg'))
            plt.close(self.plot_info[xx]['fig'])

        x = 'acc'
        xx = 'mean_' + x
        for key, value in self.acc_record.items():
            self.plot_info[x][key].plot(self.cnt_record, self.acc_record[key], style_x, label=key)
            self.plot_info[x][key].title.set_text(key + ' : ' + str(round(self.acc_record[key][-1], 4)))
            if flag_plot_xx:
                mean_val = []
                for i in range(current_epoch):
                    mean_val.append(np.mean(self.acc_record[key][int(idx_epoch[i][0]):int(idx_epoch[i][-1]) + 1]))
                self.plot_info[xx][key].plot(x_epoch[:-1], mean_val[:-1], style_xx, label=key)
                self.plot_info[xx][key].title.set_text(key + ' : ' + str(round(mean_val[-1], 4)))
                if first_plot_xx:
                    self.plot_info[xx][key].legend()
            # if current_epoch == 1:
            #     self.plot_info[x][key].legend()
        self.plot_info[x]['fig'].savefig(os.path.join(opt.save_dir, folder_name, 'plot_' + x + '.jpg'))
        plt.close(self.plot_info[x]['fig'])
        if flag_plot_xx:
            self.plot_info[xx]['fig'].savefig(os.path.join(opt.save_dir, folder_name, 'plot_' + xx + '.jpg'))
            plt.close(self.plot_info[xx]['fig'])

        x = 'etc'
        xx = 'mean_' + x
        for key, value in self.etc_record.items():
            self.plot_info[x][key].plot(self.cnt_record, self.etc_record[key], style_x, label=key)
            self.plot_info[x][key].title.set_text(key + ' : ' + str(round(self.etc_record[key][-1], 4)))
            if flag_plot_xx:
                mean_val = []
                for i in range(current_epoch):
                    mean_val.append(np.mean(self.etc_record[key][int(idx_epoch[i][0]):int(idx_epoch[i][-1]) + 1]))
                self.plot_info[xx][key].plot(x_epoch[:-1], mean_val[:-1], style_xx, label=key)
                self.plot_info[xx][key].title.set_text(key + ' : ' + str(round(mean_val[-1], 4)))
                if first_plot_xx:
                    self.plot_info[xx][key].legend()
            # if current_epoch == 1:
            #     self.plot_info[x][key].legend()
        self.plot_info[x]['fig'].savefig(os.path.join(opt.save_dir, folder_name, 'plot_' + x + '.jpg'))
        plt.close(self.plot_info[x]['fig'])
        if flag_plot_xx:
            self.plot_info[xx]['fig'].savefig(os.path.join(opt.save_dir, folder_name, 'plot_' + xx + '.jpg'))
            plt.close(self.plot_info[xx]['fig'])

    def draw_and_save_info_test(self, opt, epoch, k):

        if self.plot_init:
            self.plot_initialization(opt)
            self.plot_init = False

        current_epoch = epoch
        folder_name = 'plot_results'
        if not os.path.isdir(os.path.join(opt.save_dir, folder_name)):
            os.mkdir(os.path.join(opt.save_dir, folder_name))
        x_epoch = []



        if k == 1:
            style_xx = 'm-'
        elif k == 2:
            style_xx = 'b-'
        elif k == 3:
            style_xx = 'g-'
        elif k == 4:
            style_xx = 'k-'
        elif k == 5:
            style_xx = 'y-'
        elif k == 6:
            style_xx = 'c-'
        else:
            style_xx = 'r-'

        best_idx = np.argwhere(np.asarray(self.test_record[k]['EPOCH']) == np.asarray(self.best_epoch[k]))
        best_idx = int(best_idx[0])

        x = 'test'
        rank1 = []
        rank5 = []
        rank10 = []
        rank20 = []
        rank30 = []
        for i in range(len(self.test_record[k]['EPOCH'])):
            try:
                rank1.append(self.test_record[k]['CMC'][i][0])
            except:
                rank1.append(0)
            try:
                rank5.append(self.test_record[k]['CMC'][i][4])
            except:
                rank5.append(0)
            try:
                rank10.append(self.test_record[k]['CMC'][i][9])
            except:
                rank10.append(0)
            try:
                rank20.append(self.test_record[k]['CMC'][i][19])
            except:
                rank20.append(0)
            try:
                rank30.append(self.test_record[k]['CMC'][i][29])
            except:
                rank30.append(0)
        self.plot_info[x][k]['MAP'].plot(self.test_record[k]['EPOCH'], self.test_record[k]['MAP'], style_xx, label='MAP')
        self.plot_info[x][k]['MAP'].title.set_text(
            'MAP : ' + str(round(self.test_record[k]['MAP'][best_idx], 4)) + ' (Ep.' + str(self.best_epoch[k]) + '/' + str(current_epoch) + ')')
        self.plot_info[x][k]['Rank1'].plot(self.test_record[k]['EPOCH'], rank1, style_xx, label='Rank1')
        self.plot_info[x][k]['Rank1'].title.set_text(
            'Rank1 : ' + str(round(rank1[best_idx], 4)) + ' (Ep.' + str(self.best_epoch[k]) + '/' + str(current_epoch) + ')')
        self.plot_info[x][k]['Rank5'].plot(self.test_record[k]['EPOCH'], rank5, style_xx, label='Rank5')
        self.plot_info[x][k]['Rank5'].title.set_text(
            'Rank5 : ' + str(round(rank5[best_idx], 4)) + ' (Ep.' + str(self.best_epoch[k]) + '/' + str(current_epoch) + ')')
        self.plot_info[x][k]['Rank10'].plot(self.test_record[k]['EPOCH'], rank10, style_xx, label='Rank10')
        self.plot_info[x][k]['Rank10'].title.set_text(
            'Rank10 : ' + str(round(rank10[best_idx], 4)) + ' (Ep.' + str(self.best_epoch[k]) + '/' + str(current_epoch) + ')')
        self.plot_info[x][k]['Rank20'].plot(self.test_record[k]['EPOCH'], rank20, style_xx, label='Rank20')
        self.plot_info[x][k]['Rank20'].title.set_text(
            'Rank20 : ' + str(round(rank20[best_idx], 4)) + ' (Ep.' + str(self.best_epoch[k]) + '/' + str(current_epoch) + ')')
        self.plot_info[x][k]['Rank30'].plot(self.test_record[k]['EPOCH'], rank30, style_xx, label='Rank30')
        self.plot_info[x][k]['Rank30'].title.set_text(
            'Rank30 : ' + str(round(rank30[best_idx], 4)) + ' (Ep.' + str(self.best_epoch[k]) + '/' + str(current_epoch) + ')')


        kk = len(opt.evaluate_category)
        self.plot_info[x][kk]['MAP'].plot(self.test_record[k]['EPOCH'], self.test_record[k]['MAP'], style_xx, label='MAP')
        self.plot_info[x][kk]['MAP'].title.set_text(
            'MAP : ' + str(round(self.test_record[k]['MAP'][best_idx], 4)) + ' (Ep.' + str(self.best_epoch[k]) + '/' + str(current_epoch) + ')')
        self.plot_info[x][kk]['Rank1'].plot(self.test_record[k]['EPOCH'], rank1, style_xx, label='Rank1')
        self.plot_info[x][kk]['Rank1'].title.set_text(
            'Rank1 : ' + str(round(rank1[best_idx], 4)) + ' (Ep.' + str(self.best_epoch[k]) + '/' + str(current_epoch) + ')')
        self.plot_info[x][kk]['Rank5'].plot(self.test_record[k]['EPOCH'], rank5, style_xx, label='Rank5')
        self.plot_info[x][kk]['Rank5'].title.set_text(
            'Rank5 : ' + str(round(rank5[best_idx], 4)) + ' (Ep.' + str(self.best_epoch[k]) + '/' + str(current_epoch) + ')')
        self.plot_info[x][kk]['Rank10'].plot(self.test_record[k]['EPOCH'], rank10, style_xx, label='Rank10')
        self.plot_info[x][kk]['Rank10'].title.set_text(
            'Rank10 : ' + str(round(rank10[best_idx], 4)) + ' (Ep.' + str(self.best_epoch[k]) + '/' + str(current_epoch) + ')')
        self.plot_info[x][kk]['Rank20'].plot(self.test_record[k]['EPOCH'], rank20, style_xx, label='Rank20')
        self.plot_info[x][kk]['Rank20'].title.set_text(
            'Rank20 : ' + str(round(rank20[best_idx], 4)) + ' (Ep.' + str(self.best_epoch[k]) + '/' + str(current_epoch) + ')')
        self.plot_info[x][kk]['Rank30'].plot(self.test_record[k]['EPOCH'], rank30, style_xx, label='Rank30')
        self.plot_info[x][kk]['Rank30'].title.set_text(
            'Rank30 : ' + str(round(rank30[best_idx], 4)) + ' (Ep.' + str(self.best_epoch[k]) + '/' + str(current_epoch) + ')')



        if self.rank1_total < max(rank1):
            self.map_total = round(self.test_record[k]['MAP'][best_idx], 4)
            self.rank1_total = round(rank1[best_idx], 4)
            self.rank5_total = round(rank5[best_idx], 4)
            self.rank10_total = round(rank10[best_idx], 4)
            self.rank20_total = round(rank20[best_idx], 4)
            self.rank30_total = round(rank30[best_idx], 4)
            self.total_best_k = k
            self.total_best_idx = self.best_epoch[k]



        # if current_epoch == 1:
        #     self.plot_info[x]['MAP'].legend()
        #     self.plot_info[x]['Rank1'].legend()
        #     self.plot_info[x]['Rank5'].legend()
        #     self.plot_info[x]['Rank10'].legend()
        #     self.plot_info[x]['Rank20'].legend()
        #     self.plot_info[x]['Rank30'].legend()
        self.plot_info[x][k]['fig'].savefig(os.path.join(opt.save_dir, folder_name, 'plot_' + x + '_' + opt.evaluate_category[k] + '.jpg'))
        plt.close(self.plot_info[x][k]['fig'])

        if k == len(opt.evaluate_category) - 1: #last

            self.plot_info[x][kk]['MAP'].title.set_text(
                'MAP : ' + str(self.map_total) + ' (Ep.' + str(
                    self.total_best_idx) + '/' + str(current_epoch) + ')')
            self.plot_info[x][kk]['Rank1'].title.set_text(
                'Rank1 : ' + str(self.rank1_total) + ' (Ep.' + str(self.total_best_idx) + '/' + str(current_epoch) + ')')

            self.plot_info[x][kk]['Rank5'].title.set_text(
                'Rank5 : ' + str(self.rank5_total) + ' (Ep.' + str(self.total_best_idx) + '/' + str(current_epoch) + ')')

            self.plot_info[x][kk]['Rank10'].title.set_text(
                'Rank10 : ' + str(self.rank10_total) + ' (Ep.' + str(self.total_best_idx) + '/' + str(current_epoch) + ')')

            self.plot_info[x][kk]['Rank20'].title.set_text(
                'Rank20 : ' + str(self.rank20_total) + ' (Ep.' + str(self.total_best_idx) + '/' + str(current_epoch) + ')')

            self.plot_info[x][kk]['Rank30'].title.set_text(
                'Rank30 : ' + str(self.rank30_total) + ' (Ep.' + str(self.total_best_idx) + '/' + str(current_epoch) + ')')
            self.plot_info[x][kk]['fig'].savefig(os.path.join(opt.save_dir, folder_name, 'plot_' + x + '_all.jpg'))
            plt.close(self.plot_info[x][kk]['fig'])

        # ------------------ Save
        result = {'loss_record': self.loss_record, 'acc_record': self.acc_record,
                  'etc_record': self.etc_record,
                  'cnt_record': self.cnt_record, 'epoch_record': self.epoch_record}
        scipy.io.savemat(os.path.join(opt.save_dir, 'parameter.mat'), result)
        result_test = {'test_record': self.test_record[k]}
        scipy.io.savemat(os.path.join(opt.save_dir, 'parameter_test_' + opt.evaluate_category[k] + '.mat'), result_test)


    def record_test_result(self, CMC_single, ap_single, epoch, k):
        if CMC_single[0] > self.old_rank1[k]:
            self.old_rank1[k] = CMC_single[0]
            self.old_rank5[k] = CMC_single[4]
            self.old_rank10[k] = CMC_single[9]
            self.old_rank20[k] = CMC_single[19]
            self.old_map[k] = ap_single
            self.best_epoch[k] = epoch
        self.test_record[k]['CMC'].append(CMC_single)
        self.test_record[k]['MAP'].append(ap_single)
        self.test_record[k]['EPOCH'].append(epoch)


##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        # relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]




def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        # if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
        #     os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
        model = torch.load(os.path.join(model_dir, 'vgg16.pth'))
        # vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(model.parameters()[0], vgg.parameters()):
        # for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean)) # subtract mean
    return batch

def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun




    #---------------------------------------# Function (save_network)
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    if not os.path.isdir(os.path.join(opt.save_dir, 'checkpoints')):
        os.mkdir(os.path.join(opt.save_dir, 'checkpoints'))
    save_path = os.path.join(opt.save_dir, 'checkpoints', save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(opt.gpu_ids[0])
    print('+++++++++ Save network: epoch[{}] +++++++++'.format(epoch_label))

#---------------------------------------# Function (isdebugging)
def isdebugging():
  for frame in inspect.stack():
    if frame[1].endswith("pydevd.py"):
      return True
  return False

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()