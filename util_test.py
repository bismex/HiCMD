
from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('agg')
import time
import os
import math
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data.sampler import Sampler
import scipy.io
from sklearn.decomposition import PCA
from util_etc import *
from util_train import *
from shutil import copyfile
import cv2
version =  torch.__version__

def sample_n(X, y, z, n):

    if len(y) == 1:
        y = y.reshape(-1)
    if len(z) == 1:
        z = z.reshape(-1)
    n = len(np.unique(y)) if len(np.unique(y)) < n else n
    y_label = np.sort(np.unique(y))[:n]
    idx_all = []
    for i in range(n):
        idx = np.argwhere(np.asarray(y_label[i]) == np.asarray(y))
        idx = idx.flatten()
        idx_all.extend(idx)

    if type(y) is list:
        y = np.array(y)
    if type(z) is list:
        z = np.array(z)

    new_X = X[idx_all, :]
    new_y = y[idx_all]
    new_z = z[idx_all]
    return new_X, new_y, new_z

def dim_reduction_and_draw(TSNE_or_PCA_or_Isomap, X, y, z, n, n_components, output_file):

    dpi_tsne = 400
    h_tsne = 14
    w_tsne = 18
    alpha_tsne = 0.6
    s_tsne = 160

    X, y, z = sample_n(X, y, z, n)
    idx_0 = np.argwhere(np.asarray(z) == np.asarray(0))
    idx_0 = idx_0.flatten() # thermal
    idx_1 = np.argwhere(np.asarray(z) == np.asarray(1))
    idx_1 = idx_1.flatten() # visual

    new_X = TSNE_or_PCA_or_Isomap(n_components=n_components).fit_transform(X)

    X_0 = new_X[idx_0]
    X_1 = new_X[idx_1]
    new_X = np.concatenate((X_0, X_1), axis=0)
    y_0 = y[idx_0]
    y_1 = y[idx_1]
    new_y = y.copy()
    if max(y) > len(np.unique(y))-1:
        real_label = np.unique(new_y)
        for j in range(len(real_label)):
            idx_label = np.where(y == real_label[j])[0]
            new_y[idx_label] = j

    new_y[idx_1] += max(new_y[idx_0]) + 1
    y_tmp_0 = new_y[idx_0]
    y_tmp_1 = new_y[idx_1]
    new_y = np.concatenate((y_tmp_0, y_tmp_1), axis=0) # label (0~39) 400 개
    new_y_modal = new_y.copy()
    new_y_modal[idx_0] = 0
    new_y_modal[idx_1] = 1

    for j in range(2):

        f = plt.figure(num=None, figsize=(w_tsne, h_tsne), dpi=dpi_tsne)
        unique_y_0 = np.unique(y_0)
        unique_y_1 = np.unique(y_1)

        df = pd.DataFrame(new_X, columns=['x', 'y'])
        df['category'] = new_y
        legend_0 = {n: c for n, c in zip(range(len(unique_y_0)), ['Thermal-' + str(unique_y_0[i]) for i in range(len(unique_y_0))])}
        legend_1 = {n+len(legend_0): c for n, c in zip(range(len(unique_y_1)), ['Visual-' + str(unique_y_1[i]) for i in range(len(unique_y_1))])}

        legend_all = {}
        legend_all.update(legend_0)
        legend_all.update(legend_1)
        df['category'] = df['category'].map(legend_all)
        markers_0 = ["o" for i in range(len(unique_y_0))]
        markers_1 = ["X" for i in range(len(unique_y_1))]
        # filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
        markers = []
        markers.extend(markers_0)
        markers.extend(markers_1)
        # cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        cmap_0 = sns.color_palette("hls", n_colors = len(unique_y_0))
        cmap_1 = sns.color_palette("hls", n_colors = len(unique_y_1))
        cmap_all = cmap_0
        cmap_all.extend(cmap_1)
        if j == 1:
            cmap_binary = sns.color_palette("hls", n_colors = 2)
            for k in range(len(unique_y_0)+len(unique_y_1)):
                if k >= len(unique_y_0):
                    cmap_all[k] = cmap_binary[0]
                else:
                    cmap_all[k] = cmap_binary[1]

        # cmap_all: _ColorPalette, tuple 3개씩 30개, s_tsne 160, alpha_tsne 0.6, markers 30개
        sns.scatterplot(data = df, x='x', y='y', palette=cmap_all, s = s_tsne, hue='category', style= 'category', alpha=alpha_tsne, markers = markers)
        # sns.scatterplot(data = df, x='x', y='y', palette=cmap_all, s = 150, style= 'category', alpha=0.4, markers = markers)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.) # 1.0 -> 0.95 (right)
        plt.title(output_file)
        plt.axis('off')
        if j == 0:
            f.savefig(output_file)
        else:
            f.savefig(output_file[:-4] + '_b' + output_file[-4:])

    plt.close('all')
    return 0


def evaluate_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)

        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]

        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])

        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return new_all_cmc, mAP



def evaluate_regdb(distmat, q_pids, g_pids, max_rank=20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32) # 010011011 <

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2 * np.ones(num_g).astype(np.int32)

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

#---------------------------------------# Function (load_network)
def load_network(load_path, network):
    network.load_state_dict(torch.load(load_path))
    return network


def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = 2. - 2 * original_dist   # change the cosine similarity metric to euclidean similarity metric
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    #initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition( original_dist, range(1,k1+1) )

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh( initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh( initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)

    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist

def opt_test_settings(opt):
    if opt.test_only:
        config_path = os.path.join(opt.test_dir, 'opts.yaml')
        opt.save_dir = opt.test_dir
        opt.load_dir = opt.test_dir

        if not os.path.isdir(os.path.join(opt.save_dir, 'test')):
            os.mkdir(os.path.join(opt.save_dir, 'test'))
        opt.save_dir = os.path.join(opt.save_dir, 'test')

        if not os.path.isdir(os.path.join(opt.save_dir, opt.test_name)):
            os.mkdir(os.path.join(opt.save_dir, opt.test_name))
        opt.save_dir = os.path.join(opt.save_dir, opt.test_name)

        with open(config_path, 'r') as stream:
            config = yaml.load(stream)
        opt.dataset_sizes = config['dataset_sizes']
        opt.nclasses = config['nclasses']
        opt.G_input_dim = config['G_input_dim']
        opt.num_epoch = 1
    if opt.test_on:
        opt.etc = '_(ms' + opt.test_ms + ')'
        print('We use the scale: %s' % opt.test_ms)

    if not opt.test_only:
        if not os.path.isfile(opt.save_dir + '/train.py'):
            copyfile('./train.py', opt.save_dir + '/train.py')  # record every run
            copyfile('./trainer.py', opt.save_dir + '/trainer.py')
            copyfile('./set_option.py', opt.save_dir + '/set_option.py')
        with open('%s/opts.yaml' % opt.save_dir, 'w') as fp:  # save opts
            yaml.dump(vars(opt), fp, default_flow_style=False)
        print('===> [Save train and model file]')
        print(vars(opt))

    return opt


def evaluate_reid(qf,ql,qc,gf,gl,gc):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]


    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # Note that there are two kinds of images we do not consider as right-matching images.
    # Junk_index1 is the index of mis-detected images, which contain the body parts.
    # Junk_index2 is the index of the images, which are of the same identity in the same cameras.
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    # We can use the function compute_mAP to obtain the final result. In this function, we will ignore the junk_index.
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated

def test_show_image(opt, epoch, query_label, query_cam, query_path, query_feature, gallery_label, gallery_cam, gallery_path, gallery_feature, gallery_feature_raw, query_feature_raw, k):


    if opt.flag_reverse_figure:
        query_label_tmp = query_label
        query_path_tmp = query_path
        query_feature_tmp = query_feature
        gallery_label_tmp = gallery_label
        gallery_path_tmp = gallery_path
        gallery_feature_tmp = gallery_feature

        query_label = gallery_label_tmp
        query_path = gallery_path_tmp
        query_feature = gallery_feature_tmp

        gallery_label = query_label_tmp
        gallery_path = query_path_tmp
        gallery_feature = query_feature_tmp


    # Show image
    if opt.test_figure:

        num_query = opt.test_show_row_num
        num_gallery = opt.test_show_col_num
        idx_query_all = []
        exist_label = []
        for i in range(len(query_label)):
            tmp = np.argwhere(np.asarray(query_label[i]) == np.asarray(exist_label))
            if not len(tmp) > 0:
                idx_query_all.append(i)
                exist_label = query_label[i]
        if opt.flag_all_figure:
            num_case = int(np.floor(len(idx_query_all) / opt.test_show_row_num))
        else:
            num_case = 1
        for l in range(num_case):

            idx_query = idx_query_all[l*opt.test_show_row_num:(l+1)*opt.test_show_row_num]
            num_query = len(idx_query) if len(idx_query) < num_query else num_query

            row_feat = query_feature[idx_query].cpu()
            col_feat = gallery_feature.cpu()
            distmat = np.matmul(row_feat, np.transpose(col_feat))
            _, rank = distmat.sort(dim=1, descending=True)
            rank = rank[:,0:num_gallery]

            row_label = query_label[idx_query]
            col_label = gallery_label[rank]
            row_label = row_label.reshape(len(row_label), -1)
            label_logical = row_label == col_label


            mat_path = []
            mat_title = []
            mat_color = []
            for i in range(num_query):
                row_path = []
                row_title = ['Query ({})'.format(int(row_label[i]))]
                row_color = ['black']
                row_path.append(query_path[idx_query[i]][0].replace(' ',''))
                for j in range(num_gallery):
                    row_path.append(gallery_path[rank[i, j]][0].replace(' ',''))
                    # row_title.append('R@{} ({})'.format(j+1, int(col_label[i, j])))
                    row_title.append('R@{}'.format(j+1))
                    if label_logical[i, j]:
                        row_color.append('green')
                    else:
                        row_color.append('red')
                    # row_path.replace(' ','')
                mat_path.append(row_path)
                mat_title.append(row_title)
                mat_color.append(row_color)


            fig = plt.figure(figsize=(15, 20), dpi=300)
            cnt = 0
            img_pivot = cv2.imread(mat_path[0][0], 1)
            for i in range(num_query):
                for j in range(num_gallery + 1):
                    cnt += 1
                    ax = plt.subplot(num_query, num_gallery + 1, cnt)
                    ax.axis('off')
                    img = cv2.imread(mat_path[i][j], cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (200, 400))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                    ax.set_title(mat_title[i][j], color=mat_color[i][j])

            if not os.path.isdir(os.path.join(opt.save_dir, 'figure_results')):
                os.mkdir(os.path.join(opt.save_dir, 'figure_results'))
            if opt.test_only:
                fig.savefig(os.path.join(opt.save_dir, 'figure_results', 'test_results_final_{}_({}to{}).png'.format(opt.evaluate_category[k], l*num_query, (l+1)*num_query-1)))
            else:
                fig.savefig(os.path.join(opt.save_dir, 'figure_results', 'test_results_{}_{}_({}to{}).png'.format(str(epoch), opt.evaluate_category[k], l*num_query, (l+1)*num_query-1)))
            plt.close('all')



def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

class DummySampler(Sampler):
    def __init__(self, data):
        self.num_samples = len(data)

    def __iter__(self):
        # print ('\tcalling Sampler:__iter__')
        return iter(range(self.num_samples))

    def __len__(self):
        # print ('\tcalling Sampler:__len__')
        return self.num_samples




def extract_feature(opt, trainer, dataloaders, type_name, modals, cams):
    str_ms = opt.test_ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    modals_set = []
    for i in range(math.ceil(len(dataloaders.dataset.imgs) / dataloaders.batch_size)):
        modals_set.append(modals[i * dataloaders.batch_size:(i+1)*dataloaders.batch_size])
    cams_set = []
    for i in range(math.ceil(len(dataloaders.dataset.imgs) / dataloaders.batch_size)):
        cams_set.append(cams[i * dataloaders.batch_size:(i+1)*dataloaders.batch_size])

    features_all = []
    features_RAM_all = []

    for cnt, data in enumerate(dataloaders):  # Iterate over data.
        img, label = data
        b, c, h, w = img.size()
        if ((cnt + 1) % opt.cnt_test_print_loss == 0) or (cnt == len(dataloaders)-1):
            print('Extract {} feature..{}/{}'.format(type_name, (cnt + 1), len(dataloaders)))

        ff_all = []
        cnt_first = 0
        input_img = Variable(img.cuda())
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            for scale in ms:
                cnt_first += 1
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)

                if opt.test_RAM:
                    feature, feature_RAM = trainer.forward(opt, input_img, modals_set[cnt], cams_set[cnt])
                else:
                    feature, _ = trainer.forward(opt, input_img, modals_set[cnt], cams_set[cnt])

                if cnt_first == 1:
                    ff_all = feature
                else:
                    for k in range(len(feature)):
                        ff_all[k] += feature[k]


        # norm feature
        if opt.test_norm:
            ff_all_tmp = ff_all
            ff_all = []
            for k in range(len(ff_all_tmp)):
                ff = ff_all_tmp[k]
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))
                ff_all += [ff]

            if cnt == 0:
                features = torch.FloatTensor().cuda()

        if cnt == 0:
            for k in range(len(ff_all)):
                if opt.test_gpu:
                    features_all += [torch.FloatTensor().cuda()]
                else:
                    features_all += [torch.FloatTensor()]

            if opt.test_RAM:
                for k in range(len(feature_RAM)):
                    if opt.test_gpu:
                        features_RAM_all += [torch.FloatTensor().cuda()]
                    else:
                        features_RAM_all += [torch.FloatTensor()]

        for k in range(len(ff_all)):
            if opt.test_gpu:
                features_all[k] = torch.cat((features_all[k],ff_all[k].data), 0)
            else:
                features_all[k] = torch.cat((features_all[k],ff_all[k].data.cpu()), 0)

        if opt.test_RAM:
            for k in range(len(feature_RAM)):
                if opt.test_gpu:
                    if opt.test_RAM:
                        features_RAM_all[k] = torch.cat((features_RAM_all[k], feature_RAM[k].data), 0)
                else:
                    if opt.test_RAM:
                        features_RAM_all[k] = torch.cat((features_RAM_all[k], feature_RAM[k].data.cpu()), 0)
        # print(features.shape)


    return features_all, features_RAM_all


def evaluate_result(opt, epoch, result, result_RAM, result_multi, save_path, k):

    query_feature = torch.FloatTensor(result['query_f'])
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    query_cam = result['query_cam']
    query_label = result['query_label']
    query_path = result['query_path']
    gallery_cam = result['gallery_cam']
    gallery_label = result['gallery_label']
    gallery_path = result['gallery_path']
    if (type(query_cam) == list):
        query_cam = np.asarray(query_cam)
        query_label = np.asarray(query_label)
        gallery_cam = np.asarray(gallery_cam)
        gallery_label = np.asarray(gallery_label)
    else:
        query_cam = query_cam[0]
        query_label = query_label[0]
        gallery_cam = gallery_cam[0]
        gallery_label = gallery_label[0]


    add_name = ''
    if opt.eval_rerank: # rerank
        add_name += '(rerank)'
        q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
        q_q_dist = np.dot(query_feature, np.transpose(query_feature))
        g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
        since = time.time()
        re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        time_elapsed = time.time() - since
        print('Reranking complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if opt.data_flag == 5: # for RegDB
        if opt.eval_rerank:
            distmat = -re_rank
        else:
            distmat = np.matmul(query_feature, np.transpose(gallery_feature))
        CMC, ap = evaluate_regdb(-distmat, query_label, gallery_label, max_rank = opt.num_print_rank)


    # This is just for checking performance trends. In the case of SYSU, the evaluation is done with the official matlab code in the matlab folder.
    elif opt.data_flag == 6: # for SYSU
        if opt.eval_rerank:
            distmat = -re_rank
        else:
            distmat = np.matmul(query_feature, np.transpose(gallery_feature))
        CMC, ap = evaluate_sysu(-distmat, query_label, gallery_label, query_cam, gallery_cam, max_rank = opt.num_print_rank)
    else: # for general reid
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        query_feature = query_feature.cuda()
        gallery_feature = gallery_feature.cuda()
        for i in range(len(query_label)):
            if opt.eval_rerank:
                ap_tmp, CMC_tmp = evaluate_rerank(re_rank[i, :], query_label[i], query_cam[i], gallery_label, gallery_cam)
            else:
                ap_tmp, CMC_tmp = evaluate_reid(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
        CMC = CMC.float()
        CMC /= len(query_label) #average CMC
        ap /= len(query_label)
        CMC = CMC.numpy()

    CMC *= 100
    ap *= 100

    CMC = list(CMC)
    if len(CMC) < 100:
        for i in range(100-len(CMC)):
            CMC.append(CMC[-1])
    CMC = tuple(CMC)

    f = open(save_path + add_name + '_test_result.txt', 'w')
    print('[{}] Single-query [Rank@1 : {:.2f}] [Rank@5 : {:.2f}] [Rank@10 : {:.2f}] [Rank@20 : {:.2f}] [mAP : {:.2f}]'.format(opt.evaluate_category[k], CMC[0],CMC[4],CMC[9],CMC[19],ap))
    f.write('[Rank@1 : {:.2f}] [Rank@5 : {:.2f}] [Rank@10 : {:.2f}] [Rank@20 : {:.2f}] [mAP : {:.2f}]\n'.format(CMC[0],CMC[4],CMC[9],CMC[19],ap))
    for i in range(0, min(len(CMC), opt.num_print_rank-1)):
        f.write('Rank{}:{:.4f}\n'.format(i+1, CMC[i]))
    f.close()
    CMC_single = CMC
    ap_single = ap

    # multiple-query
    if 'mquery_f' in result_multi:
        mquery_feature = torch.FloatTensor(result_multi['mquery_f'])
        mquery_cam = result_multi['mquery_cam']
        mquery_label = result_multi['mquery_label']
        mquery_feature = mquery_feature.cuda()
        if (type(mquery_cam) == list):
            mquery_cam = np.asarray(mquery_cam)
            mquery_label = np.asarray(mquery_label)
        else:
            mquery_cam = mquery_cam[0]
            mquery_label = mquery_label[0]
    if 'mquery_f' in result_multi and not opt.eval_rerank:
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        for i in range(len(query_label)):
            mquery_index1 = np.argwhere(mquery_label==query_label[i])
            mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
            mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
            mq = torch.mean(mquery_feature[mquery_index,:], dim=0)
            ap_tmp, CMC_tmp = evaluate_reid(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
            if CMC_tmp[0]==-1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            #print(i, CMC_tmp[0])
        CMC = CMC.float()
        CMC /= len(query_label) #average CMC
        CMC *= 100
        CMC = CMC.numpy()
        ap /= len(query_label)
        ap *= 100
        f = open(save_path + '_test_multi_result.txt', 'w')
        print('Multi-query [Rank@1 : {:.2f}] [Rank@5 : {:.2f}] [Rank@10 : {:.2f}] [Rank@20 : {:.2f}] [mAP : {:.2f}]'.format(CMC[0],CMC[4],CMC[9],CMC[19],ap))
        f.write('[Rank@1 : {:.2f}] [Rank@5 : {:.2f}] [Rank@10 : {:.2f}] [Rank@20 : {:.2f}] [mAP : {:.2f}]\n'.format(CMC[0],CMC[4],CMC[9],CMC[19],ap))
        for i in range(0, min(len(CMC), opt.num_print_rank-1)):
            f.write('Rank{}:{:.4f}\n'.format(i+1, CMC[i]))
        f.close()

    time_start = time.time()

    gallery_feature_raw = []
    query_feature_raw = []
    if opt.test_RAM:
        for j in range(len(result_RAM)):
            gallery_feature_raw += [result_RAM[j]['gallery_f_raw']]
            query_feature_raw += [result_RAM[j]['query_f_raw']]
    test_show_image(opt, epoch, query_label, query_cam, query_path, query_feature, gallery_label, gallery_cam, gallery_path, gallery_feature, gallery_feature_raw, query_feature_raw, k)
    time_elapsed = time.time() - time_start
    if opt.test_RAM or opt.test_figure:
        print('Show figures in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60), end='')

    if opt.test_hist:  # histogram (intra-class, inter-class)

        if not os.path.isdir(os.path.join(opt.save_dir, 'hist_results')):
            os.mkdir(os.path.join(opt.save_dir, 'hist_results'))
        save_path = os.path.join(opt.save_dir, 'hist_results', 'net_{}_hist_{}.png'.format(str(epoch + 1), opt.evaluate_category[k]))
        evaluate_visual(opt, distmat.numpy(), query_label, gallery_label, query_cam, gallery_cam, save_path)


    return CMC_single, ap_single


def evaluate_rerank(score,ql,qc,gl,gc):
    index = np.argsort(score)  #from small to large
    #index = index[::-1]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp



def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def extract_test_features(opt, trainer, dataloaders, data_info):
    if opt.test_only:
        # load_path = opt.test_dir + opt.test_name + '.pth'
        # trainer = load_network(load_path, trainer)

        opt.resume_dir = os.path.join(opt.test_dir, 'checkpoints')
        opt.resume_name = opt.test_name
        trainer.cnt_cumul = trainer.resume(opt)

        trainer = trainer.eval()

        if opt.use_gpu:
            trainer = trainer.cuda()
    else:
        trainer = trainer.eval()
    with torch.no_grad():
        gallery_feature, gallery_feature_raw = \
            extract_feature(opt, trainer, dataloaders['gallery'],'gallery', data_info['gallery_modal'], data_info['gallery_cam'])
        query_feature, query_feature_raw = \
            extract_feature(opt, trainer, dataloaders['query'], 'query', data_info['query_modal'], data_info['query_cam'])
        if opt.test_multi:
            mquery_feature, _ = extract_feature(opt, trainer, dataloaders['multi-query'], 'mquery', data_info['query_modal'], data_info['query_cam'])

    result = []
    for k in range(len(gallery_feature)):
        result += [{'gallery_f': gallery_feature[k].numpy(), 'gallery_label': data_info['gallery_label'],
                  'gallery_cam': data_info['gallery_cam'], 'gallery_path': dataloaders['gallery'].dataset.imgs,
                  'gallery_modal': data_info['gallery_modal'], 'query_modal': data_info['query_modal'],
                  'query_f': query_feature[k].numpy(), 'query_label': data_info['query_label'],
                  'query_cam': data_info['query_cam'], 'query_path': dataloaders['query'].dataset.imgs}]

    result_RAM = []
    for k in range(len(gallery_feature_raw)):
        result_RAM += [{'gallery_f_raw': gallery_feature_raw[k], 'query_f_raw': query_feature_raw[k]}]
    result_multi = []
    if opt.test_multi:
        for k in range(len(mquery_feature)):
            result_multi += [{'mquery_f': mquery_feature[k].numpy(), 'mquery_label': data_info['mquery_label'],
                            'mquery_cam': data_info['mquery_cam']}]

    return result, result_RAM, result_multi

def save_test_features(opt, epoch, result, result_RAM, result_multi, k):
    if not os.path.isdir(os.path.join(opt.save_dir, 'test_results')):
        os.mkdir(os.path.join(opt.save_dir, 'test_results'))
    if opt.phase_train in opt.phase_exp:
        save_filename = 'net_{}'.format(str(epoch))
        save_path = os.path.join(opt.save_dir, 'test_results', save_filename + '_' + opt.data_name + opt.etc + '_' + opt.evaluate_category[k])
    else:
        save_path = os.path.join(opt.save_dir, 'test_results', opt.test_name + '_' + opt.data_name + opt.etc + '_' + opt.evaluate_category[k])
    scipy.io.savemat(save_path + '_test.mat', result)
    if opt.test_multi:
        scipy.io.savemat(save_path + '_test_multi.mat', result_multi)
    if opt.test_RAM:
        if k == 0:
            for j in range(len(result_RAM)):
                scipy.io.savemat(save_path + '_test_RAM.mat', result_RAM[j])

    return save_path

def draw_tsne_visualization(opt, epoch, result, feat_tsne, data_info, k):
    time_start = time.time()

    feat = np.concatenate((result['gallery_f'], result['query_f']), axis=0)
    label = np.concatenate((result['gallery_label'], result['query_label']), axis=0)
    modal = np.concatenate((result['gallery_modal'], result['query_modal']), axis=0)

    if not os.path.isdir(os.path.join(opt.save_dir, 'tsne_results')):
        os.mkdir(os.path.join(opt.save_dir, 'tsne_results'))
    TSNE_name = os.path.join(opt.save_dir, 'tsne_results', 'TSNE_test_{}_{}_{}.png'.format(str(epoch), opt.evaluate_category[k], str(opt.test_tsne_num)))
    # PCA_name = os.path.join(opt.save_dir, 'tsne_results', 'PCA_test_{}_{}.png'.format(str(epoch + 1), opt.evaluate_category[k]))
    dim_reduction_and_draw(TSNE, feat, label, modal, opt.test_tsne_num, 2, TSNE_name)
    # dim_reduction_and_draw(PCA, feat, label, modal, opt.test_tsne_num, 2, PCA_name)

    label = data_info['train_tsne_label']
    modal = data_info['train_tsne_modal']

    TSNE_name = os.path.join(opt.save_dir, 'tsne_results', 'TSNE_train_{}_{}_{}.png'.format(str(epoch), opt.evaluate_category[k], str(opt.test_tsne_num)))
    # PCA_name = os.path.join(opt.save_dir, 'tsne_results', 'PCA_train_{}_{}.png'.format(str(epoch + 1), opt.evaluate_category[k]))
    dim_reduction_and_draw(TSNE, feat_tsne, label, modal, opt.test_tsne_num, 2, TSNE_name)
    # dim_reduction_and_draw(PCA, feat, label, modal, opt.test_tsne_num, 2, PCA_name)
    time_elapsed = time.time() - time_start
    print('Performing Tsne/PCA in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60), end='')


def evaluate_visual(opt, distmat, q_pids, g_pids, q_camids, g_camids, save_path):
    num_q, num_g = distmat.shape

    inter_class_List = []
    intra_class_List = []

    for i in range(num_q):
        # get query pid and camid
        q_pid = q_pids[i]
        q_camid = q_camids[i]
        feat_list = distmat[i, :]

        id_same = (g_pids == q_pid).astype(np.int)
        cam_same = (g_camids == q_camid).astype(np.int)
        id_diff = 1 - (id_same)
        cam_diff = 1 - (cam_same)

        # diff CAM & diff ID
        inter_class_feat = feat_list * cam_diff * id_diff
        remove = 1 - ((inter_class_feat == 0)).astype(np.int)
        inter_class_feat = inter_class_feat[np.where(remove == 1)]
        inter_class_List.extend(inter_class_feat)

        # diff CAM & same ID
        intra_class_feat = feat_list * cam_diff * id_same
        remove = 1 - ((intra_class_feat == 0)).astype(np.int)
        intra_class_feat = intra_class_feat[np.where(remove == 1)]
        intra_class_List.extend(intra_class_feat)

    if opt.test_hist_all:
        flag1 = [False, True, True]
        flag2 = [False, False, True]

        for k in range(3):



            intra_class_List_tmp = intra_class_List.copy()
            inter_class_List_tmp = inter_class_List.copy()

            if flag1[k]:  # cos-sim to dist
                for j in range(len(intra_class_List)):
                    intra_class_List_tmp[j] = math.acos(intra_class_List_tmp[j])

                for j in range(len(inter_class_List)):
                    inter_class_List_tmp[j] = math.acos(inter_class_List_tmp[j])

            fig = plt.figure(figsize=(6, 6), dpi=200)
            ax = fig.add_subplot(111)
            if flag2[k]:
                plt.xlim(opt.test_hist_grid_min, opt.test_hist_grid_max)

            sns.distplot(inter_class_List_tmp, bins=100, hist=True, norm_hist=True, kde=False, ax=ax, color='g', label='Inter-class')
            sns.distplot(intra_class_List_tmp, bins=100, hist=True, norm_hist=True, kde=False, ax=ax, color='r', label='Intra-class')
            if flag1[k]:
                ax.set_xlabel('Feature distance')
            else:
                ax.set_xlabel('Feature similarity')
            ax.set_ylabel('Frequency')
            plt.legend(loc='upper left')
            # plt.show()
            fig.savefig(save_path[:-4] + '_' + str(k) + save_path[-4:])
            plt.close('all')


    else:

        if opt.test_hist_dist:  # cos-sim to dist
            for j in range(len(intra_class_List)):
                intra_class_List[j] = math.acos(intra_class_List[j])

            for j in range(len(inter_class_List)):
                inter_class_List[j] = math.acos(inter_class_List[j])

        fig = plt.figure(figsize=(6, 6), dpi=200)
        ax = fig.add_subplot(111)
        if opt.test_hist_grid:
            plt.xlim(opt.test_hist_grid_min, opt.test_hist_grid_max)

        sns.distplot(inter_class_List, bins=100, hist=True, norm_hist=True, kde=False, ax=ax, color='g', label='Inter-class')
        sns.distplot(intra_class_List, bins=100, hist=True, norm_hist=True, kde=False, ax=ax, color='r', label='Intra-class')
        if opt.test_hist_dist:
            ax.set_xlabel('Feature distance')
        else:
            ax.set_xlabel('Feature similarity')
        ax.set_ylabel('Frequency')
        plt.legend(loc='upper left')
        # plt.show()
        fig.savefig(save_path)
        plt.close('all')


if __name__ == "__main__":
    import numpy as np

    dist_mat = np.random.rand(50, 500)
    q_pids = np.random.choice(10, 500)
    g_pids = np.random.choice(10, 500)
    q_camids = np.random.choice(2, 500)
    g_camids = np.random.choice(2, 500)
    print(q_camids.shape)
    evaluate_visual(dist_mat, q_pids, g_pids, q_camids, g_camids)
