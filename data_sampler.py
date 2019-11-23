from torchvision import datasets
import os
import numpy as np
import random
import torch

class PosNegSampler(datasets.ImageFolder):

    def __init__(self, root, transform, data_flag = 1, name_samping = 'RAND', num_pos = 4, num_neg = 0, opt = ''):
        super(PosNegSampler, self).__init__(root, transform)
        self.cams, self.real_labels, self.modals = get_attribute(data_flag, self.samples, flag = 0)



        self.num_pos = num_pos
        self.num_neg = num_neg
        self.name_samping = name_samping
        self.opt = opt

    def _get_cam_id(self, path):
        camera_id = []
        filename = os.path.basename(path)
        camera_id = filename.split('c')[1][0]
        #camera_id = filename.split('_')[2][0:2]
        return int(camera_id)-1

    def _get_pair_pos_sample(self, index):

        pos_index = np.argwhere(np.asarray(self.real_labels) == np.asarray(self.real_labels[index]))
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index) # delete index

        modal = self.modals[index]
        cross_index = []
        for i in range(len(pos_index)):
            if modal != self.modals[pos_index[i]]:
                cross_index.append(pos_index[i])

        flag_IR_pos_same_cam = False

        if modal == 0: # 0:IR
            IR_pivot_idx = index
            RGB_pivot_idx = int(cross_index[np.random.permutation(len(cross_index))[0]])
        else:
            IR_pivot_idx = int(cross_index[np.random.permutation(len(cross_index))[0]])
            RGB_pivot_idx = index

        IR_pivot_cam = self.cams[IR_pivot_idx]

        IR_same_cam_idx = np.argwhere(np.asarray(self.modals) == np.asarray(self.modals[IR_pivot_idx])).flatten()
        IR_same_cam_idx = IR_same_cam_idx[np.random.permutation(len(IR_same_cam_idx))]
        # IR_pivot_cam = self.cams[IR_pivot_idx]

        # other IR pivot (same cam, different ID)
        IR_pivot_idx_all = []
        IR_pivot_label_all = []
        IR_pivot_idx_all.append(IR_pivot_idx)
        IR_pivot_label_all.append(self.real_labels[IR_pivot_idx])
        cnt = 0
        is_find = False
        if self.opt.pos_mini_batch > 1:
            while not is_find:
                selected_idx = int(IR_same_cam_idx[cnt])
                if not self.real_labels[selected_idx] in IR_pivot_label_all:
                    IR_pivot_idx_all.append(selected_idx)
                    IR_pivot_label_all.append(self.real_labels[selected_idx])
                if len(IR_pivot_idx_all) == self.opt.pos_mini_batch:
                    is_find = True
                cnt += 1

        # find IR pos/neg sample
        selected_pos_index = []
        selected_pos_path = []
        for k in range(len(IR_pivot_idx_all)):
            one_set = []
            selected_idx = IR_pivot_idx_all[k]
            pos_index = np.argwhere(np.asarray(self.real_labels) == np.asarray(self.real_labels[IR_pivot_idx_all[k]])).flatten()
            pos_index = np.setdiff1d(pos_index, index)
            pos_index = pos_index[np.random.permutation(len(pos_index))]
            if_find = False
            cnt = 0
            cnt_yes = 0
            pos_same_cam = [IR_pivot_idx_all[k]]
            if k == 0:
                pos_diff_modal = [RGB_pivot_idx]
            else:
                pos_diff_modal = []
            while cnt_yes != 2:
                if (self.modals[pos_index[cnt]] != self.modals[selected_idx]): # diff modal
                    if len(pos_diff_modal) < self.opt.samp_pos:
                        pos_diff_modal.append(int(pos_index[cnt]))
                        if len(pos_diff_modal) == self.opt.samp_pos:
                            cnt_yes += 1
                elif self.modals[pos_index[cnt]] == self.modals[selected_idx]: # same modal
                    if len(pos_same_cam) < self.opt.samp_pos:
                        pos_same_cam.append(int(pos_index[cnt]))
                        if len(pos_same_cam) == self.opt.samp_pos:
                            cnt_yes += 1
                cnt += 1
            one_set.extend(pos_diff_modal)
            one_set.extend(pos_same_cam)
            selected_pos_index.extend(one_set)

        for i in range(len(selected_pos_index)):
            selected_pos_path.append(self.samples[selected_pos_index[i]][0])
        # for i in range(len(selected_pos_index)):
        #     print('modal: {}, ID: {}, cam: {}'.format(self.modals[selected_pos_index[i]], self.real_labels[selected_pos_index[i]],
        #                                             self.cams[selected_pos_index[i]]))

        return selected_pos_path, selected_pos_index



    def _get_pos_sample(self, index):

        pos_index = np.argwhere(np.asarray(self.real_labels) == np.asarray(self.real_labels[index]))
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index) # delete index
        # same label: pos_index

        modal = self.modals[index]
        cam = self.cams[index]
        mono_index = []
        cross_index = []
        for i in range(len(pos_index)):
            selected_index = pos_index[i]
            selected_modal = self.modals[selected_index]
            if modal == selected_modal:
                mono_index.append(selected_index)
            else:
                cross_index.append(selected_index)


        if 'P_RAND' in self.name_samping: # [n]
            num_mono = self.num_pos
            num_cross = 0
            mono_index = pos_index
        elif 'P_MONO' in self.name_samping: # [n/0]
            num_mono = self.num_pos
            num_cross = 0
        elif 'P_CROSS' in self.name_samping: # [0/n]
            num_mono = 0
            num_cross = self.num_pos
        elif 'P_MULTI1' in self.name_samping: # [n/1] (n>=2)
            num_mono = self.num_pos - 1
            num_cross = 1
        elif 'P_MULTI2' in self.name_samping: # [1/n] (n>=2)
            num_mono = 1
            num_cross = self.num_pos - 1

        if (num_mono < 0) or (num_cross < 0):
            print('please check sampling num_pos')
            assert False


        rand_mono = np.random.permutation(len(mono_index))
        selected_pos_path = []
        selected_pos_index = []
        for i in range(num_mono):
           t = i % len(rand_mono)
           tmp_index = mono_index[rand_mono[t]]
           selected_pos_index.append(tmp_index)
           selected_pos_path.append(self.samples[tmp_index][0])

        rand_cross = np.random.permutation(len(cross_index))
        for i in range(num_cross):
           t = i % len(rand_cross)
           tmp_index = cross_index[rand_cross[t]]
           selected_pos_index.append(tmp_index)
           selected_pos_path.append(self.samples[tmp_index][0])


        return selected_pos_path, selected_pos_index

    def _get_pair_neg_sample(self, pos_label, pos_cam):

        used_label = list(set(pos_label.tolist()))
        rand_idx = np.random.permutation(len(self.real_labels))

        IR_idx_all = []
        RGB_idx_all = []
        cnt = 0

        is_find = False
        while not is_find:
            selected_idx = int(rand_idx[cnt])
            selected_label = self.real_labels[selected_idx]
            if not self.real_labels[selected_idx] in used_label:

                if self.modals[selected_idx] == 1: # RGB
                    if len(RGB_idx_all) < self.opt.neg_mini_batch:
                        RGB_idx_all.append(selected_idx)
                        used_label.append(selected_label)
                elif self.modals[selected_idx] == 0: # IR
                    if len(IR_idx_all) < self.opt.neg_mini_batch:
                        IR_idx_all.append(selected_idx)
                        used_label.append(selected_label)

            if (len(RGB_idx_all) == self.opt.neg_mini_batch) and (len(IR_idx_all) == self.opt.neg_mini_batch):
                is_find = True
            cnt += 1
        selected_neg_index = []
        selected_neg_index.extend(RGB_idx_all)
        selected_neg_index.extend(IR_idx_all)

        selected_neg_path = []
        for i in range(len(selected_neg_index)):
            selected_neg_path.append(self.samples[selected_neg_index[i]][0])
        # for i in range(len(selected_neg_index)):
        #     print('modal: {}, ID: {}, cam: {}'.format(self.modals[selected_neg_index[i]], self.real_labels[selected_neg_index[i]],
        #                                             self.cams[selected_neg_index[i]]))

        return selected_neg_path, selected_neg_index


    def _get_neg_sample(self, index):

        neg_index = np.argwhere(np.asarray(self.real_labels) != np.asarray(self.real_labels[index]))
        neg_index = neg_index.flatten()

        modal = self.modals[index]
        mono_index = []
        cross_index = []
        for i in range(len(neg_index)):
            selected_index = neg_index[i]
            selected_modal = self.modals[selected_index]
            if modal == selected_modal:
                mono_index.append(selected_index)
            else:
                cross_index.append(selected_index)

        if 'N_RAND' in self.name_samping: # [n]
            num_mono = self.num_neg
            num_cross = 0
            mono_index = neg_index
        elif 'N_MONO' in self.name_samping: # [n/0]
            num_mono = self.num_neg
            num_cross = 0
        elif 'N_CROSS' in self.name_samping: # [0/n]
            num_mono = 0
            num_cross = self.num_neg
        elif 'N_MULTI1' in self.name_samping: # [n/1] (n>=2)
            num_mono = self.num_neg - 1
            num_cross = 1
        elif 'N_MULTI2' in self.name_samping: # [1/n] (n>=2)
            num_mono = 1
            num_cross = self.num_neg - 1

        if (num_mono < 0) or (num_cross < 0):
            print('please check sampling num_neg')
            assert False

        rand_mono = np.random.permutation(len(mono_index))
        selected_neg_path = []
        selected_neg_index = []
        for i in range(num_mono):
            t = i % len(rand_mono)
            tmp_index = mono_index[rand_mono[t]]
            selected_neg_index.append(tmp_index)
            selected_neg_path.append(self.samples[tmp_index][0])

        rand_cross = np.random.permutation(len(cross_index))
        for i in range(num_cross):
            t = i % len(rand_cross)
            tmp_index = cross_index[rand_cross[t]]
            selected_neg_index.append(tmp_index)
            selected_neg_path.append(self.samples[tmp_index][0])

        return selected_neg_path, selected_neg_index



    def __getitem__(self, index):
        ori_path, order = self.samples[index]
        real_label = self.real_labels[index]
        cam = self.cams[index]
        modal = self.modals[index]
        attribute = {'order':order, 'label':real_label, 'cam':cam, 'modal':modal}
        ori = self.loader(ori_path)
        if self.transform is not None:
            ori = self.transform(ori)

        if self.num_pos > 0:
            if 'P_PAIR' in self.name_samping:
                pos_path, pos_index = self._get_pair_pos_sample(index)
            else:
                pos_path, pos_index = self._get_pos_sample(index)
            pos_cam = []
            pos_modal = []
            pos_order = []
            pos_label = []
            for i in range(len(pos_index)):
                pos_cam.append(self.cams[pos_index[i]])
                pos_modal.append(self.modals[pos_index[i]])
                pos_order.append(self.samples[pos_index[i]][1])
                pos_label.append(self.real_labels[pos_index[i]])

            pos_image = [0 for _ in range(len(pos_index))]
            for i in range(len(pos_index)):
                pos_image[i] = self.loader(pos_path[i])

            if self.transform is not None:
                for i in range(len(pos_index)):
                    pos_image[i] = self.transform(pos_image[i])

            if self.target_transform is not None:
                pass
                # label_t = self.target_transform(label_t)

            c,h,w = pos_image[0].shape
            for i in range(len(pos_index)):
                pos_image[i] = pos_image[i].view(1,c,h,w)
            pos = pos_image[0]
            for i in range(len(pos_index)-1):
                pos = torch.cat((pos, pos_image[i+1]), 0)
            pos_order = torch.as_tensor(pos_order)
            pos_label = torch.as_tensor(pos_label)
            pos_cam = torch.as_tensor(pos_cam)
            pos_modal = torch.as_tensor(pos_modal)
            attribute_pos = {'order':pos_order, 'label':pos_label, 'cam':pos_cam, 'modal':pos_modal}
        else:
            pos = []
            attribute_pos = {}



        # opt.neg_mini_batch
        if self.num_neg > 0:
            if 'N_PAIR' in self.name_samping:
                neg_path, neg_index = self._get_pair_neg_sample(pos_label, pos_cam[self.opt.samp_pos].item())
            else:
                neg_path, neg_index = self._get_neg_sample(index)

            neg_cam = []
            neg_modal = []
            neg_order = []
            neg_label = []
            for i in range(len(neg_index)):
                neg_cam.append(self.cams[neg_index[i]])
                neg_modal.append(self.modals[neg_index[i]])
                neg_order.append(self.samples[neg_index[i]][1])
                neg_label.append(self.real_labels[neg_index[i]])

            neg_image = [0 for _ in range(len(neg_index))]
            for i in range(len(neg_index)):
                neg_image[i] = self.loader(neg_path[i])

            if self.transform is not None:
                for i in range(len(neg_index)):
                    neg_image[i] = self.transform(neg_image[i])

            if self.target_transform is not None:
                pass
                # label_t = self.target_transform(label_t)

            c,h,w = neg_image[0].shape
            for i in range(len(neg_index)):
                neg_image[i] = neg_image[i].view(1,c,h,w)
            neg = neg_image[0]
            for i in range(len(neg_index)-1):
                neg = torch.cat((neg, neg_image[i+1]), 0)
            neg_order = torch.as_tensor(neg_order)
            neg_label = torch.as_tensor(neg_label)
            neg_cam = torch.as_tensor(neg_cam)
            neg_modal = torch.as_tensor(neg_modal)
            attribute_neg = {'order':neg_order, 'label':neg_label, 'cam':neg_cam, 'modal':neg_modal}
        else:
            neg = []
            attribute_neg = {}

        # pos = torch.cat((pos0.view(1,c,h,w), pos1.view(1,c,h,w), pos2.view(1,c,h,w), pos3.view(1,c,h,w)), 0)
        return ori, pos, neg, attribute, attribute_pos, attribute_neg

def get_attribute(data_flag, img_samples, flag):
    cams = []
    labels = []
    modals = []
    for path, idx in img_samples:
        labels.append(get_real_label(path, data_flag))
        cams.append(get_cam(path, data_flag))
        modals.append(gel_modal(path, data_flag))

    cams = np.asarray(cams)

    if flag == 1:  # [1,2,4,5]->0, 3->1, 6->2
        change_set = [[1, 0], [2, 0], [4, 0], [5, 0], [3, 1], [6, 2]]
    elif flag == 2:  # [1,2]->0, [4,5]->1, 3->2, 6->3
        change_set = [[1, 0], [2, 0], [4, 1], [5, 1], [3, 2], [6, 3]]
    elif flag == 3:  # X = X - 1
        change_set = [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]]
    elif flag == 0:  # [1,2,4,5]->1, [3,6]->0
        change_set = [[1, 1], [2, 1], [3, 0], [4, 1], [5, 1], [6, 0]]
    for i in range(len(change_set)):
        cams[np.where(cams == change_set[i][0])[0]] = int(change_set[i][1])
    cams = list(cams)

    return cams, labels, modals

def get_cam(path, flag):
    filename = os.path.basename(path)
    if flag == 1:  # Market1501
        return int(filename.split('c')[1][0])
    elif flag == 5: # RegDB
        if filename[0] == 'T':  # Thermal : 0
            return int(0)
        else:
            return int(1)
    elif flag == 6: # SYSU
        return int(filename[filename.find('cam')+3])

def get_real_label(path, flag):
    filename = os.path.basename(path)
    if flag == 1:  # Market1501
        label = filename[0:4]
        if label[0:2] == '-1':
            return int(-1)
        else:
            return int(label)
    elif flag == 5: # RegDB
        return int(path.split('/')[-2])
    elif flag == 6: # SYSU
        return int(path.split('/')[-2])


def gel_modal(path, flag):
    filename = os.path.basename(path)
    if flag == 1:  # Market1501
        return int(0)
    elif flag == 5: # RegDB
        if filename[0] == 'T':  # Thermal : 0
            return int(0)
        else:
            return int(1)
    elif flag == 6: # SYSU
        if filename[0] == 'T':  # Thermal : 0
            return int(0)
        else:
            return int(1)