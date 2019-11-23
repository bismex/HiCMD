
from __future__ import print_function, division
import torch
import math
import torchvision.utils as vutils
import torch.nn.init as init
import time

import inspect
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
import numpy as np
matplotlib.use('agg')
import pdb
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
import yaml
import math
import scipy.io
import sys
from util_etc import *
from util_test import *
from util_train import *
from data_sampler import *
from reIDmodel import *
from reIDmodel_others import *
from random_erasing import *
from shutil import copyfile
version =  torch.__version__


def __write_images(opt, image_outputs, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    display_image_num = image_outputs[0].size(0)
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    torch.cat([images[:, -1] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, file_name, nrow=1)

    if opt.save_all_images or opt.save_row_images or opt.save_col_images:
        dot_loc = -4
        w = image_tensor.size(-1)
        h = image_tensor.size(-2)
        all_w = image_grid.size(-1)
        all_h = image_grid.size(-2)
        num_row = round(all_h / h)
        num_col = round(all_w / w)
        idx_row = [[(j * h) + i for i in range(h)] for j in range(num_row)]
        idx_col = [[(j * w) + i for i in range(w)] for j in range(num_col)]
    if opt.save_all_images:
        # cnt = 0
        for i in range(len(idx_row)):
            image_grid2 = image_grid[:,idx_row[i],:]
            for j in range(len(idx_col)):
                # cnt += 1
                file_name2 = file_name[:dot_loc] + '_all_(' + str(i + 1) + ',' + str(j + 1) + ')'+ file_name[dot_loc:]
                vutils.save_image(image_grid2[:, :, idx_col[j]]/255.0, file_name2, nrow=1)
    if opt.save_row_images:
        for i in range(len(idx_row)):
            file_name2 = file_name[:dot_loc] + '_row_(' + str(i + 1) + ')' + file_name[dot_loc:]
            vutils.save_image(image_grid[:,idx_row[i],:]/255.0, file_name2, nrow=1)
    if opt.save_col_images:
        for i in range(len(idx_col)):
            file_name2 = file_name[:dot_loc] + '_col_(' + str(i + 1) + ')', file_name[dot_loc:]
            vutils.save_image(image_grid[:,:,idx_col[i]]/255.0, file_name2, nrow=1)


def write_2images(opt, image_outputs, image_directory, postfix):
    if not os.path.isdir(image_directory):
        os.mkdir(image_directory)
    n = len(image_outputs)
    for i in range(n):
        __write_images(opt, image_outputs[i], '{}/{}_exp{}.png'.format(image_directory, postfix, i+1))

    # __write_images(opt, image_outputs[0:n//2], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
    # __write_images(opt, image_outputs[n//2:n], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations,img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    write_one_row_html(html_file, iterations, '%s/gen_b2a_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()



def pop_after_string(full_name, local_name, pop_on = True):
    list_str = list(full_name)
    pop_str = list_str.pop(full_name.find(local_name) + len(local_name))
    if pop_on:
        full_name = ''.join(list_str)
    return full_name, pop_str


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


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))



def opt_settings(opt):

    #---------------------------------------# Common setting
    opt.phase_train = 'train'
    if opt.train_all:
        opt.phase_train = opt.phase_train + '_all'
        opt.val_epoch = 0
    if opt.test_only:
        opt.phase_exp = ['test']
    else:
        opt.phase_exp = [opt.phase_train, 'test']
    opt.test_on = True
    opt.phase_data = []
    if opt.phase_train in opt.phase_exp:
        opt.phase_data.append(opt.phase_train)
    if 'val' in opt.phase_exp:
        opt.phase_data.append('val')
    if 'test' in opt.phase_exp:
        opt.phase_data.append('gallery')
        opt.phase_data.append('query')
        if opt.test_multi:
            opt.phase_data.append('multi-query')

    #---------------------------------------# Train setting
    if opt.phase_train in opt.phase_exp:
        opt.save_dir = os.path.join('./model', opt.data_name, opt.name_output)
        cnt = 0
        while os.path.isfile(opt.save_dir + '/train.py'):
            cnt += 1
            add_name = opt.name_output + '_' + str(cnt)
            opt.save_dir = os.path.join('./model', opt.data_name, add_name)
            print('Folder "{}" already exists, change name to {}'.format(opt.save_dir, os.path.join('./model', opt.data_name, add_name)))
        if not os.path.isdir(os.path.join('./model', opt.data_name)):
            os.mkdir(os.path.join('./model', opt.data_name))
        if not os.path.isdir(opt.save_dir):
            os.mkdir(opt.save_dir)
        sys.stdout = Logger(os.path.join(opt.save_dir, 'log.txt'))
    if isdebugging():
        opt.pin_memory = False
        opt.train_workers = 0
        opt.test_workers = 0
    else:
        opt.pin_memory = True
    opt.set_batchsize = {}
    opt.set_shuffle = {}
    opt.set_workers = {}
    opt.set_droplast = {}
    for x in opt.phase_data:
        if x in opt.phase_train:
            opt.set_batchsize[x] = opt.train_batchsize
            opt.set_shuffle[x] = True
            opt.set_workers[x] = opt.train_workers
            if opt.skip_last_batch:
                opt.set_droplast[x] = True
            else:
                opt.set_droplast[x] = False
        elif x in ['val', 'gallery', 'query', 'multi-query']:
            opt.set_batchsize[x] = opt.test_batchsize
            opt.set_shuffle[x] = False
            opt.set_workers[x] = opt.test_workers
            opt.set_droplast[x] = False


    #---------------------------------------# Check GPU
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >=0:
            gpu_ids.append(gid)
    torch.cuda.set_device(gpu_ids[0])

    print('===> [Check gpu] {}'.format(gpu_ids))
    opt.gpu_ids = gpu_ids

    cudnn.benchmark = True

    return opt

def data_settings(opt):
    #---------------------------------------# Transform data

    transform_train_list = []
    transform_train_list = transform_train_list + [transforms.Resize((opt.h,opt.w), interpolation=3)]
    transform_train_list = transform_train_list + [transforms.Pad(opt.pad)] if opt.pad > 0 else transform_train_list
    transform_train_list = transform_train_list + [transforms.RandomCrop((opt.h,opt.w))] if opt.pad > 0 else transform_train_list
    transform_train_list = transform_train_list + [transforms.RandomHorizontalFlip()] if opt.flip else transform_train_list
    transform_train_list = transform_train_list + [transforms.ToTensor()]
    transform_train_list = transform_train_list + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    transform_val_list = []
    transform_val_list = transform_val_list + [transforms.Resize(size=(opt.h,opt.w),interpolation=3)]
    transform_val_list = transform_val_list + [transforms.ToTensor()]
    transform_val_list = transform_val_list + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    if opt.all_erasing_p>0:
        transform_train_list = transform_train_list +  [RandomErasing(probability = opt.all_erasing_p, mean=[0.0, 0.0, 0.0])]
    if opt.color_jitter:
        transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list
    data_transforms = {}
    for x in opt.phase_data:
        if x == opt.phase_train:
            data_transforms[x] = transforms.Compose(transform_train_list)
        else:
            data_transforms[x] = transforms.Compose(transform_val_list)
    print('===> [Transform data] ' + str(transform_train_list))
    data_info = {}
    # if opt.test_tsne and opt.test_on:
    image_datasets_train_tsne = datasets.ImageFolder(os.path.join(opt.data_dir, opt.data_name, opt.phase_train), data_transforms['query'])
    cnt = 0
    for i in range(len(image_datasets_train_tsne.targets)):
        if image_datasets_train_tsne.targets[i] < opt.test_tsne_num:
            cnt += 1
    sampler = DummySampler(image_datasets_train_tsne)
    sampler.num_samples = cnt
    # sampler.num_samples = len(image_datasets_train_tsne.targets)
    dataloaders_train_tsne = torch.utils.data.DataLoader(image_datasets_train_tsne, batch_size=opt.set_batchsize['query'], shuffle=opt.set_shuffle['query'],
                                              num_workers=opt.set_workers['query'], sampler = sampler, pin_memory=opt.pin_memory, drop_last=opt.set_droplast['query'])
    data_info['train_tsne_cam'], data_info['train_tsne_label'], data_info['train_tsne_modal'] = get_attribute(opt.data_flag, image_datasets_train_tsne.imgs, flag = opt.type_domain_label)

    train_label_all = data_info['train_tsne_label'] # all labels
    train_modal_all = data_info['train_tsne_modal']
    train_cam_all = data_info['train_tsne_cam']

    data_info['train_tsne_cam'] = data_info['train_tsne_cam'][:cnt]
    data_info['train_tsne_label'] = data_info['train_tsne_label'][:cnt]
    data_info['train_tsne_modal'] = data_info['train_tsne_modal'][:cnt]

    # else:
    #     dataloaders_train_tsne = []
    #---------------------------------------# Load data
    since = time.time()
    image_datasets = {x: datasets.ImageFolder(os.path.join(opt.data_dir, opt.data_name, x), data_transforms[x]) for x in opt.phase_data}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.set_batchsize[x], shuffle=opt.set_shuffle[x],
                                                  num_workers=opt.set_workers[x], pin_memory=opt.pin_memory, drop_last=opt.set_droplast[x]) for x in opt.phase_data}
    if opt.test_on:
        data_info['gallery_cam'], data_info['gallery_label'], data_info['gallery_modal'] = get_attribute(opt.data_flag, image_datasets['gallery'].imgs, flag = opt.type_domain_label)
        data_info['query_cam'], data_info['query_label'], data_info['query_modal'] = get_attribute(opt.data_flag, image_datasets['query'].imgs, flag = opt.type_domain_label)
        if opt.test_multi:
            data_info['mquery_cam'], data_info['mquery_label'], data_info['mquery_modal'] = get_attribute(opt.data_flag, image_datasets['multi-query'].imgs, flag = opt.type_domain_label)
    if opt.phase_train in opt.phase_exp:
        opt.dataset_sizes = {x: len(image_datasets[x]) for x in opt.phase_data}
        class_names = image_datasets[opt.phase_train].classes
        opt.nclasses = len(class_names)
        # inputs, classes = next(iter(dataloaders[opt.phase_train]))
    opt.use_gpu = torch.cuda.is_available()
    print(' ------------------------------')
    print(' Dataset {} statistics:'.format(opt.data_name))
    print(' ------------------------------')
    print(' subset   | # ids | # images')
    print(' ------------------------------')
    if opt.phase_train in opt.phase_exp:
        if opt.cross_reid:
            train_cam, train_label, train_modal = get_attribute(opt.data_flag, image_datasets[opt.phase_train].imgs, flag = opt.type_domain_label)
            cross_flag = {}
            cross_flag['thermal'] = int(0)
            cross_flag['visual'] = int(1)
            cross_idx = {}
            cross_modal = {}
            cross_cam = {}
            cross_label = {}
            cross_idx['thermal'] = [i for i, x in enumerate(train_modal) if x == cross_flag['thermal']]
            cross_idx['visual'] = [i for i, x in enumerate(train_modal) if x == cross_flag['visual']]
            cross_modal['thermal'] = [train_modal[x] for x in cross_idx['thermal']]
            cross_modal['visual'] = [train_modal[x] for x in cross_idx['visual']]
            cross_cam['thermal'] = [train_cam[x] for x in cross_idx['thermal']]
            cross_cam['visual'] = [train_cam[x] for x in cross_idx['visual']]
            cross_label['thermal'] = [train_label[x] for x in cross_idx['thermal']]
            cross_label['visual'] = [train_label[x] for x in cross_idx['visual']]
            print(' Visible  | {:5d} | {:8d}'.format(len(np.unique(cross_label['visual'])), train_modal.count(cross_flag['visual'])))
            print(' Thermal  | {:5d} | {:8d}'.format(len(np.unique(cross_label['thermal'])), train_modal.count(cross_flag['thermal'])))
        else:
            print(' Training | {:5d} | {:8d}'.format(opt.nclasses, len(image_datasets[opt.phase_train])))
        print(' ------------------------------')
    if 'test' in opt.phase_exp:
        print(' Query    | {:5d} | {:8d}'.format(len(np.unique(data_info['query_label'])), len(data_info['query_label'])))
        print(' Gallery  | {:5d} | {:8d}'.format(len(np.unique(data_info['gallery_label'])), len(data_info['gallery_label'])))
        print(' ------------------------------')
        print(' Data Loading Time:\t {:.3f}'.format(round(time.time()-since, 3)))
        print(' ------------------------------')
    # if (opt.samp_pos + opt.samp_neg) > 0:
    if not opt.test_only:
        old_train_dataloader = dataloaders[opt.phase_train]
    else:
        old_train_dataloader = []
    for x in opt.phase_data:
        if not x in ['query', 'gallery']:
            image_datasets[x] = PosNegSampler(os.path.join(opt.data_dir, opt.data_name, x), data_transforms[x], data_flag = opt.data_flag,
                                              name_samping = opt.name_samping, num_pos = opt.samp_pos, num_neg = opt.samp_neg, opt=opt)
            dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.set_batchsize[x], shuffle=opt.set_shuffle[x],
                                                            num_workers=opt.set_workers[x], pin_memory=opt.pin_memory, drop_last=opt.set_droplast[x])

    train_v_idx_first = []
    train_t_idx_first = []
    train_v_idx_last = []
    train_t_idx_last = []

    train_modal_all = np.asarray(train_modal_all)
    train_cam_all = np.asarray(train_cam_all)

    if opt.test_IR2_flag:  # IR2
        if opt.type_domain_label == 1:  # 0:RGB, 1:IR, 2:IR2
            cam_idx = 2
        elif opt.type_domain_label == 2:  # 0,1:RGB, 2:IR, 3:IR2
            cam_idx = 3
        elif opt.type_domain_label == 3:  # 0,1,3,4:RGB, 2:IR, 5:IR2
            cam_idx = 5
    else: # IR
        if opt.type_domain_label == 1:  # 0:RGB, 1:IR, 2:IR2
            cam_idx = 1
        elif opt.type_domain_label == 2:  # 0,1:RGB, 2:IR, 3:IR2
            cam_idx = 2
        elif opt.type_domain_label == 3:  # 0,1,3,4:RGB, 2:IR, 5:IR2
            cam_idx = 2


    for i in range(len(np.unique(train_label_all))):
        idx = np.where(train_label_all == np.unique(train_label_all)[i])[0]
        train_v_idx_first.append(idx[np.where(train_modal_all[idx] == 1)[0][0]])
        train_v_idx_last.append(idx[np.where(train_modal_all[idx] == 1)[0][-1 - opt.visual_last_idx ]])
        if opt.data_name == 'SYSU' and opt.type_domain_label > 0:
            common_idx = idx[[train_modal_all[idx] == 0][0] * [train_cam_all[idx] == cam_idx][0]]
            if len(common_idx) > 0:
                train_t_idx_first.append(common_idx[0])
                train_t_idx_last.append(common_idx[-1 - opt.visual_last_idx])
        else:
            train_t_idx_first.append(idx[np.where(train_modal_all[idx] == 0)[0][0]])
            train_t_idx_last.append(idx[np.where(train_modal_all[idx] == 0)[0][-1 - opt.visual_last_idx]])

    opt.num_draw_samples_idx_train_a = opt.num_draw_samples_idx_train_a.split(',')
    opt.num_draw_samples_idx_train_a = [int(train_v_idx_first[int(opt.num_draw_samples_idx_train_a[i])]) for i in range(len(opt.num_draw_samples_idx_train_a))]

    opt.num_draw_samples_idx_train_b = opt.num_draw_samples_idx_train_b.split(',')
    opt.num_draw_samples_idx_train_b = [int(train_t_idx_first[int(opt.num_draw_samples_idx_train_b[i])]) for i in range(len(opt.num_draw_samples_idx_train_b))]


    # opt.num_draw_samples_train = min(opt.num_draw_samples_train, len(train_v_idx_first), len(train_t_idx_first))


    train_display_images_a = torch.stack([dataloaders_train_tsne.dataset[opt.num_draw_samples_idx_train_a[i]][0] for i in range(len(opt.num_draw_samples_idx_train_a))])
    train_display_images_b = torch.stack([dataloaders_train_tsne.dataset[opt.num_draw_samples_idx_train_b[i]][0] for i in range(len(opt.num_draw_samples_idx_train_b))])
    train_display_images_a_pos = dataloaders_train_tsne.dataset[train_v_idx_last[opt.visual_pos_idx]][0]
    train_display_images_b_pos = dataloaders_train_tsne.dataset[train_t_idx_last[opt.visual_pos_idx]][0]
    train_display_images_a_neg = dataloaders_train_tsne.dataset[train_v_idx_last[opt.visual_neg_idx]][0]
    train_display_images_b_neg = dataloaders_train_tsne.dataset[train_t_idx_last[opt.visual_neg_idx]][0]


    gallery_lables = np.array(data_info['gallery_label'])
    gallery_modals = np.array(data_info['gallery_modal'])
    query_labels = np.array(data_info['query_label'])
    query_modals = np.array(data_info['query_modal'])
    gallery_cams = np.array(data_info['gallery_cam'])
    query_cams = np.array(data_info['query_cam'])

    test_v_idx = np.where(gallery_modals == 1)[0]
    if len(test_v_idx) > 0:
        gallery_is_v = True
    else:
        gallery_is_v = False

    train_v_idx_first = []
    train_t_idx_first = []
    train_v_idx_last = []
    train_t_idx_last = []
    for i in range(len(np.unique(gallery_lables))):
        idx = np.where(gallery_lables == np.unique(gallery_lables)[i])[0]
        # gallery
        if gallery_is_v:
            train_v_idx_first.append(idx[np.where(gallery_modals[idx] == 1)[0][0]])
            train_v_idx_last.append(idx[np.where(gallery_modals[idx] == 1)[0][-1 - opt.visual_last_idx]])
        else: # thermal

            if opt.data_name == 'SYSU' and opt.type_domain_label > 0:
                common_idx = idx[[gallery_modals[idx] == 0][0] * [gallery_cams[idx] == cam_idx][0]]
                if len(common_idx) > 0:
                    train_t_idx_first.append(common_idx[0])
                    train_t_idx_last.append(common_idx[-1 - opt.visual_last_idx])
            else:
                train_t_idx_first.append(idx[np.where(gallery_modals[idx] == 0)[0][0]])
                train_t_idx_last.append(idx[np.where(gallery_modals[idx] == 0)[0][-1 - opt.visual_last_idx]])

    for i in range(len(np.unique(query_labels))):
        idx = np.where(query_labels == np.unique(query_labels)[i])[0]
        # query
        if gallery_is_v: # thermal

            if opt.data_name == 'SYSU' and opt.type_domain_label > 0:
                common_idx = idx[[query_modals[idx] == 0][0] * [query_cams[idx] == cam_idx][0]]
                if len(common_idx) > 0:
                    train_t_idx_first.append(common_idx[0])
                    train_t_idx_last.append(common_idx[-1 - opt.visual_last_idx])
            else:
                train_t_idx_first.append(idx[np.where(query_modals[idx] == 0)[0][0]])
                train_t_idx_last.append(idx[np.where(query_modals[idx] == 0)[0][-1 - opt.visual_last_idx]])



        else:
            train_v_idx_first.append(idx[np.where(query_modals[idx] == 1)[0][0]])
            train_v_idx_last.append(idx[np.where(query_modals[idx] == 1)[0][-1 - opt.visual_last_idx]])

    opt.num_draw_samples_idx_test_a = opt.num_draw_samples_idx_test_a.split(',')
    opt.num_draw_samples_idx_test_a = [int(train_v_idx_first[int(opt.num_draw_samples_idx_test_a[i])]) for i in range(len(opt.num_draw_samples_idx_test_a))]

    opt.num_draw_samples_idx_test_b = opt.num_draw_samples_idx_test_b.split(',')
    opt.num_draw_samples_idx_test_b = [int(train_t_idx_first[int(opt.num_draw_samples_idx_test_b[i])]) for i in range(len(opt.num_draw_samples_idx_test_b))]


    if gallery_is_v:
        test_display_images_a = torch.stack(
            [dataloaders['gallery'].dataset[opt.num_draw_samples_idx_test_a[i]][0] for i in range(len(opt.num_draw_samples_idx_test_a))])
        test_display_images_b = torch.stack(
            [dataloaders['query'].dataset[opt.num_draw_samples_idx_test_b[i]][0] for i in range(len(opt.num_draw_samples_idx_test_b))])
        test_display_images_a_pos = dataloaders['gallery'].dataset[train_v_idx_last[opt.visual_pos_idx]][0]
        test_display_images_b_pos = dataloaders['query'].dataset[train_t_idx_last[opt.visual_pos_idx]][0]
        test_display_images_a_neg = dataloaders['gallery'].dataset[train_v_idx_last[opt.visual_neg_idx]][0]
        test_display_images_b_neg = dataloaders['query'].dataset[train_t_idx_last[opt.visual_neg_idx]][0]
    else:
        test_display_images_a = torch.stack(
            [dataloaders['query'].dataset[opt.num_draw_samples_idx_test_a[i]][0] for i in range(len(opt.num_draw_samples_idx_test_a))])
        test_display_images_b = torch.stack(
            [dataloaders['gallery'].dataset[opt.num_draw_samples_idx_test_b[i]][0] for i in range(len(opt.num_draw_samples_idx_test_b))])
        test_display_images_a_pos = dataloaders['query'].dataset[train_v_idx_last[opt.visual_pos_idx]][0]
        test_display_images_b_pos = dataloaders['gallery'].dataset[train_t_idx_last[opt.visual_pos_idx]][0]
        test_display_images_a_neg = dataloaders['query'].dataset[train_v_idx_last[opt.visual_neg_idx]][0]
        test_display_images_b_neg = dataloaders['gallery'].dataset[train_t_idx_last[opt.visual_neg_idx]][0]

    data_sample = {'train_a':train_display_images_a, 'train_b':train_display_images_b, \
                   'test_a' : test_display_images_a, 'test_b' : test_display_images_b, \
                   'train_a_pos':train_display_images_a_pos, 'train_b_pos':train_display_images_b_pos,\
                   'test_a_pos':test_display_images_a_pos, 'test_b_pos':test_display_images_b_pos, \
                   'train_a_neg':train_display_images_a_neg, 'train_b_neg':train_display_images_b_neg,\
                   'test_a_neg':test_display_images_a_neg, 'test_b_neg':test_display_images_b_neg}


    if not opt.test_only:
        new_epoch = opt.num_cnt / (math.floor(len(dataloaders[opt.phase_train].dataset.imgs) / opt.train_batchsize))
        opt.num_epoch = 100
        if new_epoch < opt.num_epoch:
            print('num_epoch is changed from {} to {}'.format(opt.num_epoch, math.ceil(new_epoch)))
            opt.num_epoch = math.ceil(new_epoch)
        opt.total_cnt = math.ceil(new_epoch) * math.floor(
            len(dataloaders[opt.phase_train].dataset.imgs) / opt.train_batchsize)


    return dataloaders, dataloaders_train_tsne, old_train_dataloader, data_info, data_sample, opt


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

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)
