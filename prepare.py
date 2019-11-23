import os
import argparse
import time
import sys
from shutil import copyfile
import cv2
from PIL import Image
parser = argparse.ArgumentParser(description='Prepare')
# parser.add_argument('--load_dir',default='../../DB/reid/Market-1501-v15.09.15/',type=str, help='load dir path')
# parser.add_argument('--save_dir',default='../../DB/baseline_DB/Market_tmp/',type=str, help='save dir path')
parser.add_argument('--dataset_flag',default=8,type=int, help='flag for preparing dataset (0:Market1501 / 5:RegDB / 6:SYSU)')
opt = parser.parse_args()

import numpy as np
from scipy import misc

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def format_check(name, list):
    format_err = True
    for i in range(len(list)):
        if name == list[i]:
            format_err = False
            continue
    return format_err


def copy_file(footnote, load_name, save_name, save_flag):
    load_path = os.path.join(opt.load_dir, load_name)
    if os.path.isdir(load_path):
        cnt = 0
        save_path = os.path.join(opt.save_dir, save_name)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        for root, dirs, files in os.walk(load_path, topdown=True):
            for name in files:
                if format_check(name[-3:], opt.im_format):
                    print(' ---- (format error) : ' + name)
                    continue
                ID  = name.split('_')
                load_image_path = os.path.join(load_path, name)
                save_image_folder = os.path.join(save_path, ID[0])
                save_image_path = os.path.join(save_image_folder, name)
                if not os.path.isdir(save_image_folder):
                    os.mkdir(save_image_folder)
                    first_image_flag = True
                else:
                    first_image_flag = False
                    if os.path.isfile(save_image_path):
                        sys.stdout.write('\r' + footnote + ' ' + name_data + ' [' + load_name + '->' + save_name + '] ---- File already exists')
                        continue
                if save_flag == 'all':
                    cnt += 1
                    copyfile(load_image_path, save_image_path)
                if save_flag == 'all-1':
                    if not first_image_flag:
                        cnt += 1
                        copyfile(load_image_path, save_image_path)
                    else:
                        continue
                if save_flag == '1':
                    if first_image_flag:
                        cnt += 1
                        copyfile(load_image_path, save_image_path)
                    else:
                        continue

                sys.stdout.write('\r' + footnote + ' ' + name_data + ' [' + load_name + '->' + save_name + '] {}/{}'.format(cnt, len(files)))


def copy_file_regDB(footnote, load_name, save_name, save_flag, load_idx):

    image_list = os.path.join(opt.load_dir, 'idx', load_idx + '.txt')

    with open(image_list) as f:
        data_file_list = open(image_list, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    load_path = os.path.join(opt.load_dir, load_name)
    if os.path.isdir(load_path):
        cnt = 0
        save_path = os.path.join(opt.save_dir, save_name)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        old_name = -100
        for i in range(len(file_image)):
            load_image_path = os.path.join(opt.load_dir, file_image[i])
            save_image_folder = os.path.join(save_path, '{:03d}'.format(file_label[i]))
            name = file_image[i].split('/')
            name = load_name[0] + '_' + name[2]
            new_name = file_label[i]
            if new_name == old_name:
                first_image_flag = False
            else:
                first_image_flag = True
            old_name = new_name
            save_image_path = os.path.join(save_image_folder, name)
            if not os.path.isdir(save_image_folder):
                os.mkdir(save_image_folder)
            else:
                if os.path.isfile(save_image_path):
                    sys.stdout.write('\r' + footnote + ' ' + name_data + ' [' + load_name + '->' + save_name + '] ---- File already exists')
                    continue
            if save_flag == 'all':
                cnt += 1
                copyfile(load_image_path, save_image_path)
            if save_flag == 'all-1':
                if not first_image_flag:
                    cnt += 1
                    copyfile(load_image_path, save_image_path)
                else:
                    continue
            if save_flag == '1':
                if first_image_flag:
                    cnt += 1
                    copyfile(load_image_path, save_image_path)
                else:
                    continue
            sys.stdout.write('\r' + footnote + ' ' + name_data + ' [' + load_name + '->' + save_name + '] {}/{}'.format(cnt,
                                                                                                           len(file_image)))


def copy_file_regDB_I2I(footnote, load_name, save_name, load_idx):

    image_list = os.path.join(opt.load_dir, 'idx', load_idx + '.txt')

    with open(image_list) as f:
        data_file_list = open(image_list, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    all_name = []
    load_path = os.path.join(opt.load_dir, load_name)
    if os.path.isdir(load_path):
        cnt = 0
        save_path = os.path.join(opt.save_dir, save_name)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        old_name = -100
        for i in range(len(file_image)):
            load_image_path = os.path.join(opt.load_dir, file_image[i])
            save_image_folder = save_path
            name = file_image[i].split('/')
            name = name[2]
            all_name.append(name)
            save_image_path = os.path.join(save_image_folder, name)
            # img = cv2.imread(load_image_path, cv2.IMREAD_GRAYSCALE)
            # cv2.imwrite(save_image_path, img)
            copyfile(load_image_path, save_image_path)
            sys.stdout.write('\r' + footnote + ' ' + name_data + ' [' + load_name + '->' + save_name + '] {}/{}'.format(cnt,
                                                                                                           len(file_image)))

    name_list = 'list_' + save_name + '.txt'
    f = open(os.path.join(opt.save_dir, name_list), 'w')
    for i in range(len(all_name)):
        data = "{}\n".format(all_name[i])
        f.write(data)
    f.close()

def copy_file_SYSU(footnote, load_name, load_modal, save_name, load_idx):



    image_list = os.path.join(opt.load_dir, 'exp', load_idx + '_id.txt')
    with open(image_list, "r") as file:
        file_lines = file.readlines()
    id_line = file_lines[0]
    all_ids = ["%04d" % int(i) for i in id_line.split(",")]


    for x in load_name:
        load_name_local = 'cam' + str(x)
        load_path = os.path.join(opt.load_dir, load_name_local)
        print('Load : {}'.format(load_path))
        if os.path.isdir(load_path):
            cnt = 0
            save_path = os.path.join(opt.save_dir, save_name)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            for i in range(len(all_ids)):
                cnt += 1
                load_image_folder = os.path.join(load_path, all_ids[i])
                save_image_folder = os.path.join(save_path, '{}'.format(all_ids[i]))
                if not os.path.isdir(save_image_folder):
                    os.mkdir(save_image_folder)
                for root, dirs, files in os.walk(load_image_folder, topdown=True):
                    for name in files:
                        if format_check(name[-3:], opt.im_format):
                            print(' ---- (format error) : ' + name)
                            continue
                        load_image_path = os.path.join(load_image_folder, name)
                        name = load_modal[0] + '_' + load_name_local + '_l' + all_ids[i] + '_' + name
                        save_image_path = os.path.join(save_image_folder, name)
                        copyfile(load_image_path, save_image_path)
                        sys.stdout.write('\r' + footnote + ' ' + name_data + ' [' + load_name_local + '->' + save_name + '] {}/{}'.format(cnt,len(all_ids)))


def copy_file_SYSU_I2I(footnote, load_name, save_name, load_idx):



    image_list = os.path.join(opt.load_dir, 'exp', load_idx + '_id.txt')
    with open(image_list, "r") as file:
        file_lines = file.readlines()
    id_line = file_lines[0]
    all_ids = ["%04d" % int(i) for i in id_line.split(",")]

    all_name = []
    for x in load_name:
        load_name_local = 'cam' + str(x)
        load_path = os.path.join(opt.load_dir, load_name_local)
        print('Load : {}'.format(load_path))
        if os.path.isdir(load_path):
            cnt = 0
            save_path = os.path.join(opt.save_dir, save_name)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            for i in range(len(all_ids)):
                cnt += 1
                load_image_folder = os.path.join(load_path, all_ids[i])
                save_image_folder = save_path
                # save_image_folder = os.path.join(save_path, '{}'.format(all_ids[i]))
                # if not os.path.isdir(save_image_folder):
                #     os.mkdir(save_image_folder)
                for root, dirs, files in os.walk(load_image_folder, topdown=True):
                    for name in files:
                        if format_check(name[-3:], opt.im_format):
                            print(' ---- (format error) : ' + name)
                            continue
                        load_image_path = os.path.join(load_image_folder, name)
                        name = all_ids[i] + '_' + load_name_local + '_' + name
                        save_image_path = os.path.join(save_image_folder, name)
                        all_name.append(name)
                        img = cv2.imread(load_image_path, cv2.IMREAD_GRAYSCALE)
                        cv2.imwrite(save_image_path, img)
                        # copyfile(load_image_path, save_image_path)
                        sys.stdout.write('\r' + footnote + ' ' + name_data + ' [' + load_name_local + '->' + save_name + '] {}/{}'.format(cnt,len(all_ids)))

    name_list = 'list_' + save_name + '.txt'
    f = open(os.path.join(opt.save_dir, name_list), 'w')
    for i in range(len(all_name)):
        data = "{}\n".format(all_name[i])
        f.write(data)
    f.close()


#-----------------------------------------

def check_file():
    opt.im_format = ['jpg', 'png', 'bmp']
    if not os.path.isdir(opt.load_dir):
        print('please change the opt.load_dir')
    if not os.path.isdir(opt.save_dir):
        os.mkdir(opt.save_dir)
    print('Load dir: ' + opt.load_dir)
    print('Save dir: ' + opt.save_dir)

if opt.dataset_flag == 0:
    name_data = 'Market1501'
    opt.load_dir = '../../DB/reid/Market-1501-v15.09.15/'
    opt.save_dir = '../../DB/baseline_DB/Market_lite/'
    check_file()
    copy_file('(1/6)', 'query', 'query', 'all')
    copy_file('(2/6)', 'gt_bbox', 'multi-query', 'all')
    copy_file('(3/6)', 'bounding_box_test', 'gallery', 'all')
    copy_file('(4/6)', 'bounding_box_train', 'train_all', 'all')
    copy_file('(5/6)', 'bounding_box_train', 'train', 'all-1')
    copy_file('(6/6)', 'bounding_box_train', 'val', '1')

elif opt.dataset_flag == 5:
    name_data = 'RegDB'
    opt.load_dir = '../../DB/crossreid/RegDB/'
    for trial in range(1, 11):
        opt.save_dir = '../../DB/baseline_DB/RegDB_{:02d}_tmp/'.format(trial)
        check_file()
        print('\n{} dataset ... trial : {}'.format(name_data, trial))
        copy_file_regDB('(1/8)', 'Visible', 'query', 'all', 'test_visible_' + str(trial))
        copy_file_regDB('(2/8)', 'Thermal', 'gallery', 'all', 'test_thermal_' + str(trial))
        copy_file_regDB('(3/8)', 'Visible', 'train_all', 'all',  'train_visible_' + str(trial))
        copy_file_regDB('(4/8)', 'Visible', 'train', 'all-1', 'train_visible_' + str(trial))
        copy_file_regDB('(5/8)', 'Visible', 'val', '1', 'train_visible_' + str(trial))
        copy_file_regDB('(6/8)', 'Thermal', 'train_all', 'all', 'train_thermal_' + str(trial))
        copy_file_regDB('(7/8)', 'Thermal', 'train', 'all-1', 'train_thermal_' + str(trial))
        copy_file_regDB('(8/8)', 'Thermal', 'val', '1', 'train_thermal_' + str(trial))

elif opt.dataset_flag == 6:
    name_data = 'SYSU'
    opt.load_dir = '../../DB/crossreid/SYSU/SYSU-MM01/'
    opt.save_dir = '../../DB/baseline_DB/SYSU/'
    check_file()
    copy_file_SYSU('(1/6)', [3, 6], 'Thermal', 'query', 'test')
    copy_file_SYSU('(2/6)', [1, 2, 4, 5], 'Visible', 'gallery', 'test')
    copy_file_SYSU('(3/6)', [3, 6], 'Thermal', 'train', 'train')
    copy_file_SYSU('(4/6)', [1, 2, 4, 5], 'Visible', 'train', 'train')
    copy_file_SYSU('(5/6)', [3, 6], 'Thermal', 'val', 'val')
    copy_file_SYSU('(6/6)', [1, 2, 4, 5], 'Visible', 'val', 'val')

elif opt.dataset_flag == 7: # for I2I
    name_data = 'RegDB'
    opt.load_dir = '../../DB/crossreid/RegDB/'
    opt.save_dir = '../../DB/baseline_DB/RegDB_I2I/'
    check_file()
    copy_file_regDB_I2I('(1/4)', 'Visible', 'testA', 'test_visible_' + str(1))
    copy_file_regDB_I2I('(2/4)', 'Thermal', 'testB', 'test_thermal_' + str(1))
    copy_file_regDB_I2I('(3/4)', 'Visible', 'trainA', 'train_visible_' + str(1))
    copy_file_regDB_I2I('(4/4)', 'Thermal', 'trainB', 'train_thermal_' + str(1))




elif opt.dataset_flag == 8: # for I2I
    name_data = 'SYSU'
    opt.load_dir = '../../DB/crossreid/SYSU/SYSU-MM01/'
    opt.save_dir = '../../DB/baseline_DB/SYSU_I2I_gray/'
    check_file()
    copy_file_SYSU_I2I('(1/4)', [3, 6], 'testB', 'test')
    copy_file_SYSU_I2I('(2/4)', [1, 2, 4, 5], 'testA', 'test')
    copy_file_SYSU_I2I('(3/4)', [3, 6], 'trainB', 'train')
    copy_file_SYSU_I2I('(4/4)', [1, 2, 4, 5], 'trainA', 'train')


elif opt.dataset_flag == 9: # for I2I
    name_data = 'SYSU'
    opt.load_dir = '../../DB/crossreid/SYSU/SYSU-MM01/'
    opt.save_dir = '../../DB/baseline_DB/SYSU_indoor_I2I_gray/'
    check_file()
    copy_file_SYSU_I2I('(1/4)', [3], 'testB', 'test')
    copy_file_SYSU_I2I('(2/4)', [1, 2], 'testA', 'test')
    copy_file_SYSU_I2I('(3/4)', [3], 'trainB', 'train')
    copy_file_SYSU_I2I('(4/4)', [1, 2], 'trainA', 'train')


elif opt.dataset_flag == 10: # for I2I
    name_data = 'SYSU'
    opt.load_dir = '../../DB/crossreid/SYSU/SYSU-MM01/'
    opt.save_dir = '../../DB/baseline_DB/SYSU_outdoor_I2I_gray/'
    check_file()
    copy_file_SYSU_I2I('(1/4)', [6], 'testB', 'test')
    copy_file_SYSU_I2I('(2/4)', [4, 5], 'testA', 'test')
    copy_file_SYSU_I2I('(3/4)', [6], 'trainB', 'train')
    copy_file_SYSU_I2I('(4/4)', [4, 5], 'trainA', 'train')