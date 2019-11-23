from __future__ import print_function, division
import torch.nn as nn
import matplotlib
matplotlib.use('agg')
import scipy.io
from util_test import *
from util_train import *
from data_sampler import *
from reIDmodel import *
from util_etc import *
import copy
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.optim import lr_scheduler

from networks import Discriminator, ResidualEncoder,  ResidualDecoder, MLP
from util_train import weights_init, get_model_list, vgg_preprocess, load_vgg16
from random_erasing import RandomErasing

class HICMD(nn.Module):
    def __init__(self, opt):
        super(HICMD, self).__init__()

        # Initialization
        self.old_loss_type = {}
        self.old_acc_type = {}
        self.old_etc_type = {}
        self.loss_type = {}
        self.acc_type = {}
        self.etc_type = {}
        self.cnt_cumul = 0
        self.loss_type['TOTAL'] = 0
        self.loss_type['TOT_D'] = 0
        self.loss_type['TOT_G'] = 0
        self.loss_type['TOT_ID'] = 0
        self.zero_grad_G = True
        self.zero_grad_D = True
        self.zero_grad_ID = True
        self.cnt_batch_G = 0
        self.cnt_batch_D = 0
        self.cnt_batch_ID = 0
        opt.old_apply_pos_cnt = opt.apply_pos_cnt

        # Discriminator
        dis_params = []
        self.dis_RGB = Discriminator(opt)
        dis_params += list(self.dis_RGB.parameters())
        self.dis_IR = copy.deepcopy(self.dis_RGB)
        dis_params += list(self.dis_IR.parameters())

        # Generator (attribute encoder)
        gen_params = []
        self.gen_RGB = nn.Module()
        self.gen_IR = nn.Module()
        self.gen_RGB.enc_att = ft_resnet(depth = opt.G_style_resnet_depth, pretrained = opt.G_style_pretrained, stride = opt.stride, \
                                         partpool = opt.G_style_partpool, pooltype = opt.G_style_typepool, input_channel = opt.G_input_dim)
        gen_params += list(self.gen_RGB.enc_att.parameters())
        self.gen_IR.enc_att = copy.deepcopy(self.gen_RGB.enc_att)
        gen_params += list(self.gen_IR.enc_att.parameters())

        # Generator (prototype encoder)
        bottleneck_dim = opt.G_dim
        self.gen_RGB.enc_pro = ResidualEncoder(n_downsample = opt.G_n_downsamp, n_res = opt.G_n_residual, input_dim = opt.G_input_dim, \
                                                bottleneck_dim = bottleneck_dim, norm = opt.G_enc_res_norm, activ = opt.G_act, pad_type = opt.G_pad_type, \
                                                tanh = opt.G_tanh, res_type = opt.G_res_type, enc_type = opt.G_enc_type, flag_ASPP = opt.G_ASPP, \
                                                init = opt.G_init, w_lrelu = opt.G_w_lrelu)
        gen_params += list(self.gen_RGB.enc_pro.parameters())
        self.gen_IR.enc_pro = copy.deepcopy(self.gen_RGB.enc_pro)
        gen_params += list(self.gen_IR.enc_pro.parameters())

        # Decoder
        self.num_mlp_layer = 4
        input_dim = round(self.gen_RGB.enc_att.output_dim / self.num_mlp_layer)
        self.flag_mlp = 2
        self.gen_RGB.dec = ResidualDecoder(n_upsample=opt.G_n_downsamp, n_res=opt.G_n_residual, input_dim=self.gen_RGB.enc_pro.output_dim, \
                                           output_dim=opt.G_input_dim, dropout=opt.G_dropout, res_norm=opt.G_dec_res_norm, activ=opt.G_act, \
                                           pad_type=opt.G_pad_type, res_type=opt.G_res_type, non_local=opt.G_non_local, \
                                           dec_type=opt.G_dec_type, init=opt.G_init, w_lrelu = opt.G_w_lrelu, \
                                           mlp_input = input_dim, mlp_output = 2 * self.gen_RGB.enc_pro.output_dim, mlp_dim = opt.G_mlp_dim, \
                                           mlp_n_blk = opt.G_mlp_n_blk, mlp_norm = 'none', mlp_activ = opt.G_act)
        gen_params += list(self.gen_RGB.dec.parameters())
        self.gen_IR.dec = self.gen_RGB.dec


        # attribute indexing
        dim = self.gen_RGB.enc_att.output_dim
        self.att_pose_idx = []
        for i in range(opt.G_n_residual):
            for j in range(round(dim/opt.G_n_residual*opt.att_pose_ratio)):
                self.att_pose_idx.append(j + i * round(dim/opt.G_n_residual))
        self.att_illum_idx = [i for i in range(dim) if not i in self.att_pose_idx]
        self.att_pose_dim = len(self.att_pose_idx)
        self.att_illum_dim = len(self.att_illum_idx)

        self.att_style_idx = []
        for i in range(opt.G_n_residual):
            for j in range(round(dim / opt.G_n_residual * opt.att_style_ratio)):
                self.att_style_idx.append(j + i * round(dim / opt.G_n_residual))
        self.att_ex_idx = [i for i in range(dim) if not i in self.att_style_idx]

        self.att_pose_idx = []
        for i in range(opt.G_n_residual):
            for j in range(round(
                    len(self.att_ex_idx) / opt.G_n_residual * opt.att_pose_ratio)):
                self.att_pose_idx.append(self.att_ex_idx[j + i * round(
                    len(self.att_ex_idx) / opt.G_n_residual)])
        self.att_illum_idx = [self.att_ex_idx[i] for i in
                                    range(len(self.att_ex_idx)) if
                                    not self.att_ex_idx[i] in self.att_pose_idx]

        self.att_style_dim = len(self.att_style_idx)
        self.att_ex_dim = len(self.att_ex_idx)
        self.att_pose_dim = len(self.att_pose_idx)
        self.att_illum_dim = len(self.att_illum_idx)

        opt.att_pose_idx = self.att_pose_idx
        opt.att_illum_idx = self.att_illum_idx
        opt.att_style_idx = self.att_style_idx
        opt.att_ex_idx = self.att_ex_idx

        # prototype backbone
        id_dim = 0
        input_dim = self.gen_RGB.enc_pro.output_dim
        self.backbone_pro = ft_resnet2(input_dim = input_dim, depth = opt.backbone_pro_resnet_depth, stride = opt.stride,\
                                               max_num_conv = opt.backbone_pro_max_num_conv, max_ouput_dim = opt.backbone_pro_max_ouput_dim, \
                                               pretrained = opt.backbone_pro_pretrained, partpool = opt.backbone_pro_partpool, pooltype = opt.backbone_pro_typepool)
        id_dim += self.backbone_pro.output_dim
        id_dim += self.att_style_dim

        if id_dim == 0:
            assert(False)

        # additional fc_layers (for CE, TRIP)
        self.combine_weight = ft_weight()
        self.id_classifier = ft_classifier(input_dim = id_dim, class_num = opt.nclasses, droprate = opt.droprate, fc1 = opt.fc1_channel, fc2 = opt.fc2_channel, \
                                         bnorm = opt.bnorm, ID_norm = opt.ID_norm, ID_act = opt.ID_act, w_lrelu = opt.w_lrelu, return_f = True)
        all_params = [{'params': list(self.id_classifier.parameters()), 'lr': opt.lr_backbone}]
        all_params.append({'params': list(self.backbone_pro.parameters()), 'lr': opt.lr_backbone * opt.backbone_pro_lr_ratio})
        all_params.append({'params': list(self.combine_weight.parameters()), 'lr': opt.lr_backbone * opt.combine_weight_lr_ratio})

        # Optimizer and scheduler
        self.id_optimizer = optim.SGD(all_params, weight_decay=opt.weight_decay_bb, momentum=opt.momentum,
                                      nesterov=opt.flag_nesterov)
        self.dis_optimizer = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                              lr=opt.lr_dis, betas=(opt.beta1, opt.beta2),
                                              weight_decay=opt.weight_decay_dis)
        self.gen_optimizer = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                              lr=opt.lr_gen, betas=(opt.beta1, opt.beta2),
                                              weight_decay=opt.weight_decay_gen)

        self.id_scheduler = lr_scheduler.StepLR(self.id_optimizer, step_size=opt.step_size_bb, gamma=opt.gamma_bb)
        self.dis_scheduler = lr_scheduler.StepLR(self.dis_optimizer, step_size=opt.step_size_dis, gamma=opt.gamma_dis)
        self.gen_scheduler = lr_scheduler.StepLR(self.gen_optimizer, step_size=opt.step_size_gen, gamma=opt.gamma_gen)

        # set GPU
        if opt.use_gpu:
            self.id_classifier = self.id_classifier.cuda()
            self.backbone_pro = self.backbone_pro.cuda()
            self.combine_weight = self.combine_weight.cuda()
            self.gen_RGB = self.gen_RGB.cuda()
            self.gen_IR = self.gen_IR.cuda()
            self.dis_RGB = self.dis_RGB.cuda()
            self.dis_IR = self.dis_IR.cuda()
            print('===> [Start training]' + '-' * 30)
            self.CE_criterion = nn.CrossEntropyLoss()
            self.NLL_criterion = nn.NLLLoss()
            self.etc_type['D_lr'] = self.dis_optimizer.param_groups[0]['lr']
            self.etc_type['G_lr'] = self.gen_optimizer.param_groups[0]['lr']
            self.etc_type['ID_lr'] = self.id_optimizer.param_groups[0]['lr']

        self.edge = to_edge
        self.single = to_gray(False)
        self.rand_erasing = RandomErasing(probability=opt.CE_erasing_p, mean=[0.0, 0.0, 0.0])

    def go_train(self, data, opt, phase, cnt, epoch):

        # Batch sampler
        if phase in opt.phase_train:
            self.cnt_cumul += 1
        pred_labels = f = pf = nf = []  # initialize
        _, pos, neg, attribute, attribute_pos, attribute_neg, labels = self.data_variable(opt, data)
        self.is_pos = True if len(pos) > 0 else False
        self.is_neg = True if len(neg) > 0 else False

        pos_all = pos[0]
        pos_label_all = attribute_pos['order'][0]
        pos_cam_all = attribute_pos['cam'][0]
        pos_modal_all = attribute_pos['modal'][0]
        num_pos = pos_all.size(0)

        pos_a1_idx = [x for x in range(num_pos) if x % 4 == 0]
        pos_a2_idx = [x for x in range(num_pos) if x % 4 == 1]
        pos_b1_idx = [x for x in range(num_pos) if x % 4 == 2]
        pos_b2_idx = [x for x in range(num_pos) if x % 4 == 3]

        x_a1 = pos_all[pos_a1_idx]
        x_a2 = pos_all[pos_a2_idx]
        x_b1 = pos_all[pos_b1_idx]
        x_b2 = pos_all[pos_b2_idx]

        labels_a1 = pos_label_all[pos_a1_idx]
        labels_a2 = pos_label_all[pos_a2_idx]
        labels_b1 = pos_label_all[pos_b1_idx]
        labels_b2 = pos_label_all[pos_b2_idx]

        modals_a1 = pos_modal_all[pos_a1_idx]
        modals_a2 = pos_modal_all[pos_a2_idx]
        modals_b1 = pos_modal_all[pos_b1_idx]
        modals_b2 = pos_modal_all[pos_b2_idx]

        cams_a1 = pos_cam_all[pos_a1_idx]
        cams_a2 = pos_cam_all[pos_a2_idx]
        cams_b1 = pos_cam_all[pos_b1_idx]
        cams_b2 = pos_cam_all[pos_b2_idx]

        if self.is_neg:
            neg_all = neg[0]
            neg_label_all = attribute_neg['order'][0]
            neg_cam_all = attribute_neg['cam'][0]
            neg_modal_all = attribute_neg['modal'][0]
            num_neg = neg_all.size(0)
            neg_a_idx = [x for x in range(num_neg) if x < opt.neg_mini_batch]
            neg_b_idx = [x for x in range(num_neg) if x >= opt.neg_mini_batch]
            neg_a = neg_all[neg_a_idx]
            neg_b = neg_all[neg_b_idx]
            self.labels_neg_a = neg_label_all[neg_a_idx]
            self.labels_neg_b = neg_label_all[neg_b_idx]
        else:
            neg_a = []
            neg_b = []

        case_a1 = 'RGB'
        case_a2 = 'RGB'
        case_b1 = 'IR'
        case_b2 = 'IR'

        if self.cnt_cumul > 1:
            if self.cnt_cumul < opt.cnt_warmI2I: # only I2I
                opt.apply_pos_cnt = opt.warm_apply_pos_cnt
            elif self.cnt_cumul < opt.cnt_warmI2I + opt.cnt_warmID: # only ID
                opt.apply_pos_cnt = opt.warm_apply_pos_cnt
            else:
                opt.apply_pos_cnt = opt.old_apply_pos_cnt
        else:
            opt.old_apply_pos_cnt = opt.apply_pos_cnt

        # Same-modality (not used)
        if (opt.apply_pos_cnt > 0) or (opt.cnt_initialize_pos >= self.cnt_cumul):
            if (self.cnt_cumul != 1 and (self.cnt_cumul % opt.apply_pos_cnt == 0)) or (opt.cnt_initialize_pos >= self.cnt_cumul):
                self.labels_a = labels_a1
                self.labels_b = labels_a2
                self.modals_a = modals_a1
                self.modals_b = modals_a2
                self.cams_a = cams_a1
                self.cams_b = cams_a2
                self.case_a = case_a1
                self.case_b = case_a2
                self.dis_update(x_a1, x_a2, opt, phase)
                self.gen_update(x_a1, x_a2, neg_a, neg_b, opt, phase)

                self.labels_a = labels_b1
                self.labels_b = labels_b2
                self.modals_a = modals_b1
                self.modals_b = modals_b2
                self.cams_a = cams_b1
                self.cams_b = cams_b2
                self.case_a = case_b1
                self.case_b = case_b2
                self.dis_update(x_b1, x_b2, opt, phase)
                self.gen_update(x_b1, x_b2, neg_a, neg_b, opt, phase)

        # Cross-modality
        if (opt.cnt_initialize_pos < self.cnt_cumul):
            self.labels_a = labels_a1
            self.labels_b = labels_b1
            self.modals_a = modals_a1
            self.modals_b = modals_b1
            self.cams_a = cams_a1
            self.cams_b = cams_b1
            self.case_a = case_a1
            self.case_b = case_b1
            self.dis_update(x_a1, x_b1, opt, phase)
            self.gen_update(x_a1, x_b1, neg_a, neg_b, opt, phase)

        for key, value in self.loss_type.items():
            self.old_loss_type[key] = self.loss_type[key]
        for key, value in self.acc_type.items():
            self.old_acc_type[key] = self.acc_type[key]
        for key, value in self.etc_type.items():
            self.old_etc_type[key] = self.etc_type[key]
        if opt.flag_synchronize:
            torch.cuda.synchronize()

    def dis_update(self, x_a, x_b, opt, phase):

        # Update discriminator
        if self.case_a == 'RGB':
            self.dis_a = self.dis_RGB
            self.gen_a = self.gen_RGB
        elif self.case_a == 'IR':
            self.dis_a = self.dis_IR
            self.gen_a = self.gen_IR
        else:
            assert(False)

        if self.case_b == 'RGB':
            self.dis_b = self.dis_RGB
            self.gen_b = self.gen_RGB
        elif self.case_b == 'IR':
            self.dis_b = self.dis_IR
            self.gen_b = self.gen_IR
        else:
            assert(False)

        if self.cnt_cumul > 1:
            if self.cnt_cumul < opt.cnt_warmI2I: # only I2I
                Do_dis_update = True
            elif self.cnt_cumul < opt.cnt_warmI2I + opt.cnt_warmID: # only ID
                Do_dis_update = False
            else:
                Do_dis_update = True
        else:
            Do_dis_update = True

        if Do_dis_update:
            if self.zero_grad_D:
                self.dis_optimizer.zero_grad()
                self.zero_grad_D = False

            with torch.no_grad():
                if opt.D_input_dim == 1:
                    Gx_a = self.single(x_a.clone())
                    Gx_b = self.single(x_b.clone())
                else:
                    Gx_a = x_a.clone()
                    Gx_b = x_b.clone()

                Gx_a_raw = Gx_a.clone()
                Gx_b_raw = Gx_b.clone()

                c_a = self.gen_a.enc_pro(Gx_a)
                c_b = self.gen_b.enc_pro(Gx_b)

                s_a = self.gen_a.enc_att(Gx_a)
                s_b = self.gen_b.enc_att(Gx_b)

                s_a2 = s_a.clone()
                s_b2 = s_b.clone()
                s_a, s_b = change_two_index(s_a, s_b, self.att_style_idx, self.att_ex_idx)

                x_ba = self.gen_a.dec(c_b, s_a, self.gen_a.enc_pro.output_dim)
                x_ab = self.gen_b.dec(c_a, s_b, self.gen_b.enc_pro.output_dim)


            self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), Gx_a_raw)  # fake, real image [b x 1 x 64 x 32] matrix LSGAN
            self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), Gx_b_raw)
            self.loss_dis_total = (opt.w_gan * self.loss_dis_a + opt.w_gan * self.loss_dis_b)


            s_a3_add = s_a2.clone()
            s_b3_add = s_b2.clone()
            # pose change
            tmp = self.att_style_idx.copy()
            tmp.extend(self.att_pose_idx)
            s_a3_add, s_b3_add = change_two_index(s_a3_add, s_b3_add, tmp, self.att_illum_idx) # modality remain

            x_ba3 = self.gen_a.dec(c_b, s_a3_add, self.gen_a.enc_pro.output_dim) # (ID, pose) b, (modality) a
            x_ab3 = self.gen_b.dec(c_a, s_b3_add, self.gen_b.enc_pro.output_dim) # (ID, pose) a, (modality) b

            self.loss_dis_a3 = self.dis_a.calc_dis_loss(x_ba3.detach(), Gx_a_raw)
            self.loss_dis_b3 = self.dis_b.calc_dis_loss(x_ab3.detach(), Gx_b_raw)
            self.loss_dis_total += opt.w_gan * (self.loss_dis_a3 + self.loss_dis_b3)

            self.loss_dis_total.backward(retain_graph=False) #*****GPU memory --> 40Mb / ?/ -60Mb
            self.dis_optimizer.step() #*****GPU memory --> 100Mb / 0
            self.zero_grad_D = True

            self.loss_type['D_a'] = opt.w_gan * self.loss_dis_a.item()  # scalar
            self.loss_type['D_b'] = opt.w_gan * self.loss_dis_b.item()
            self.loss_type['TOT_D'] = self.loss_dis_total.item()

        else:
            try:
                self.loss_type['D_a'] = self.old_loss_type['D_a']
                self.loss_type['D_b'] = self.old_loss_type['D_b']
                self.loss_type['TOT_D'] = self.old_loss_type['TOT_D']
            except:
                self.loss_type['D_a'] = 0
                self.loss_type['D_b'] = 0
                self.loss_type['TOT_D'] = 0

        if self.case_a == 'RGB':
            self.dis_RGB = self.dis_a
            self.gen_RGB = self.gen_a
        elif self.case_a == 'IR':
            self.dis_IR = self.dis_a
            self.gen_IR = self.gen_a
        else:
            assert(False)

        if self.case_b == 'RGB':
            self.dis_RGB = self.dis_b
            self.gen_RGB = self.gen_b
        elif self.case_b == 'IR':
            self.dis_IR = self.dis_b
            self.gen_IR = self.gen_b
        else:
            assert(False)


    def gen_update(self, x_a, x_b, neg_a, neg_b, opt, phase):


        if self.case_a == 'RGB':
            self.dis_a = self.dis_RGB
            self.gen_a = self.gen_RGB
        elif self.case_a == 'IR':
            self.dis_a = self.dis_IR
            self.gen_a = self.gen_IR
        else:
            assert(False)


        if self.case_b == 'RGB':
            self.dis_b = self.dis_RGB
            self.gen_b = self.gen_RGB
        elif self.case_b == 'IR':
            self.dis_b = self.dis_IR
            self.gen_b = self.gen_IR
        else:
            assert(False)

        if self.cnt_cumul > 1:
            if self.cnt_cumul < opt.cnt_warmI2I: # only I2I
                Do_gen_update = True
                Do_id_update = False
            elif self.cnt_cumul < opt.cnt_warmI2I + opt.cnt_warmID: # only ID
                Do_gen_update = False
                Do_id_update = True
            else:
                Do_gen_update = True
                Do_id_update = True
        else:
            Do_gen_update = True
            Do_id_update = True


        ##########################################################################
        # ID-PIG (ID-preserving Person Image Generation)
        ##########################################################################
        if Do_gen_update or Do_id_update:
            if opt.G_input_dim == 1:
                Gx_a = self.single(x_a.clone())
                Gx_b = self.single(x_b.clone())
            else:
                Gx_a = x_a
                Gx_b = x_b

            Gx_a_raw = Gx_a.clone()
            Gx_b_raw = Gx_b.clone()

            if self.zero_grad_G:
                self.gen_optimizer.zero_grad()
                self.zero_grad_G = False

            c_a = self.gen_a.enc_pro(Gx_a)
            c_b = self.gen_b.enc_pro(Gx_b)
            s_a = self.gen_a.enc_att(Gx_a)
            s_b = self.gen_b.enc_att(Gx_b)
            s_a_id = s_a.clone()
            s_b_id = s_b.clone()

            s_a2 = s_a.clone()
            s_b2 = s_b.clone()
            s_a, s_b = change_two_index(s_a, s_b, self.att_style_idx, self.att_ex_idx)

            x_ba = self.gen_a.dec(c_b, s_a, self.gen_a.enc_pro.output_dim)
            x_a_recon = self.gen_a.dec(c_a, s_a, self.gen_a.enc_pro.output_dim)

            x_ab = self.gen_b.dec(c_a, s_b, self.gen_b.enc_pro.output_dim)
            x_b_recon = self.gen_b.dec(c_b, s_b, self.gen_b.enc_pro.output_dim)

            x_ba_raw = x_ba.clone()
            x_ab_raw = x_ab.clone()

            if Do_gen_update:
                c_b_recon = self.gen_a.enc_pro(x_ba)
                c_a_recon = self.gen_b.enc_pro(x_ab)
                s_a_recon = self.gen_a.enc_att(x_ba)
                s_b_recon = self.gen_b.enc_att(x_ab)
                s_a_recon_id = s_a_recon.clone()
                s_b_recon_id = s_b_recon.clone()

                if opt.w_cycle_x > 0:
                    x_aba = self.gen_a.dec(c_a_recon, s_a, self.gen_a.enc_pro.output_dim)
                    x_bab = self.gen_b.dec(c_b_recon, s_b, self.gen_b.enc_pro.output_dim)

                # reconstruction loss (same)
                self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, Gx_a_raw)
                self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, Gx_b_raw)
                self.loss_gen_recon_x = opt.w_recon_x * (self.loss_gen_recon_x_a + self.loss_gen_recon_x_b)
                self.loss_type['G_rec_x'] = self.loss_gen_recon_x.item() if self.loss_gen_recon_x != 0 else 0

                # reconstruction loss (cross)
                self.loss_gen_cross_x_ab = self.recon_criterion(x_ab_raw, Gx_b_raw)
                self.loss_gen_cross_x_ba = self.recon_criterion(x_ba_raw, Gx_a_raw)
                self.loss_gen_cross_x = opt.w_cross_x * (self.loss_gen_cross_x_ab + self.loss_gen_cross_x_ba)
                self.loss_type['G_cross_x'] = self.loss_gen_cross_x.item() if self.loss_gen_cross_x != 0 else 0

                # reconstruction loss (cycle)
                if opt.w_cycle_x > 0:
                    self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, Gx_a_raw)
                    self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, Gx_b_raw)
                else:
                    self.loss_gen_cyc_x_a = 0
                    self.loss_gen_cyc_x_b = 0
                self.loss_gen_cyc_x = opt.w_cycle_x * (self.loss_gen_cyc_x_a + self.loss_gen_cyc_x_b)
                self.loss_type['G_cyc_x'] = self.loss_gen_cyc_x.item() if self.loss_gen_cyc_x != 0 else 0

                # attribute code reconstruction loss
                att_style_s_a = torch_gather(s_a_id, self.att_style_idx, 1)
                att_ex_s_a = torch_gather(s_a_id, self.att_ex_idx, 1) if self.att_illum_dim != 0 else None
                att_style_s_b = torch_gather(s_b_id, self.att_style_idx, 1)
                att_ex_s_b = torch_gather(s_b_id, self.att_ex_idx, 1) if self.att_illum_dim != 0 else None
                att_style_s_a_recon = torch_gather(s_a_recon_id, self.att_style_idx, 1)
                att_ex_s_a_recon = torch_gather(s_a_recon_id, self.att_ex_idx, 1) if self.att_illum_dim != 0 else None
                att_style_s_b_recon = torch_gather(s_b_recon_id, self.att_style_idx, 1)
                att_ex_s_b_recon = torch_gather(s_b_recon_id, self.att_ex_idx, 1) if self.att_illum_dim != 0 else None

                self.loss_gen_recon_s = self.recon_criterion(att_ex_s_a, att_ex_s_a_recon)
                self.loss_gen_recon_s += self.recon_criterion(att_ex_s_b, att_ex_s_b_recon)
                self.etc_type['S_remain'] = self.loss_gen_recon_s.item()

                self.loss_gen_recon_s2 = self.recon_criterion(att_style_s_b, att_style_s_a_recon)
                self.loss_gen_recon_s2 += self.recon_criterion(att_style_s_a, att_style_s_b_recon)
                self.etc_type['S_ID'] = self.loss_gen_recon_s2.item()
                self.loss_gen_recon_s += self.loss_gen_recon_s2
                self.loss_gen_recon_s *= opt.w_recon_s
                self.loss_type['G_rec_s'] = self.loss_gen_recon_s.item() if self.loss_gen_recon_s != 0 else 0

                # KLD loss (attribute code to gaussian distribution)
                self.loss_gen_s_a_kl = self.compute_kl(
                    torch_gather(s_a, self.att_ex_idx, 1)) if opt.w_style_kl != 0 else 0
                self.loss_gen_s_b_kl = self.compute_kl(
                    torch_gather(s_b, self.att_ex_idx, 1)) if opt.w_style_kl != 0 else 0
                self.loss_gen_kl = opt.w_style_kl * (self.loss_gen_s_a_kl + self.loss_gen_s_b_kl)
                self.loss_type['style_kl'] = self.loss_gen_kl.item() if self.loss_gen_kl != 0 else 0

                # GAN loss
                self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba_raw)
                self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab_raw)
                self.loss_gen_adv = opt.w_gan * (self.loss_gen_adv_a + self.loss_gen_adv_b)
                self.loss_type['G_adv'] = self.loss_gen_adv.item() if self.loss_gen_adv != 0 else 0
                s_a3_add = s_a2.clone()
                s_b3_add = s_b2.clone()

                tmp = self.att_style_idx.copy()
                tmp.extend(self.att_pose_idx)
                s_a3_add, s_b3_add = change_two_index(s_a3_add, s_b3_add, tmp, self.att_illum_idx)  # only modality change

                x_ba3 = self.gen_a.dec(c_b, s_a3_add, self.gen_a.enc_pro.output_dim)  # (ID, pose) b, (modality) a
                x_ab3 = self.gen_b.dec(c_a, s_b3_add, self.gen_b.enc_pro.output_dim)  # (ID, pose) a, (modality) b
                self.loss_gen_adv_a3 = self.dis_a.calc_gen_loss(x_ba3)
                self.loss_gen_adv_b3 = self.dis_b.calc_gen_loss(x_ab3)
                self.loss_gen_adv += opt.w_gan * (self.loss_gen_adv_a3 + self.loss_gen_adv_b3)

                # total ID-PIG loss
                self.loss_gen_total = self.loss_gen_recon_x + self.loss_gen_cyc_x + self.loss_gen_adv + \
                                      self.loss_gen_recon_s + self.loss_gen_kl + self.loss_gen_cross_x

                if opt.HFL_ratio > 0:
                    self.loss_gen_total *= (1.0 - opt.HFL_ratio)

                self.loss_type['TOT_G'] = self.loss_gen_total.item()

            else:
                try:
                    self.loss_type['G_adv'] = self.old_loss_type['G_adv']
                    self.loss_type['G_rec_x'] = self.old_loss_type['G_rec_x']
                    self.loss_type['G_cyc_x'] = self.old_loss_type['G_cyc_x']
                    self.loss_type['G_rec_s'] = self.old_loss_type['G_rec_s']
                    self.loss_type['G_style_kl'] = self.old_loss_type['G_style_kl']
                    self.loss_type['G_cross_x'] = self.old_loss_type['G_cross_x']
                    self.loss_type['TOT_G'] = self.old_loss_type['TOT_G']
                except:
                    self.loss_type['G_adv'] = 0
                    self.loss_type['G_rec_x'] = 0
                    self.loss_type['G_cyc_x'] = 0
                    self.loss_type['G_rec_s'] = 0
                    self.loss_type['G_style_kl'] = 0
                    self.loss_type['G_cross_x'] = 0
                    self.loss_type['TOT_G'] = 0

        ##########################################################################
        # HFL (Hierarchical Feature Learning)
        ##########################################################################

        if Do_id_update:

            if self.zero_grad_ID:
                self.id_optimizer.zero_grad()  ##################### Add
                self.zero_grad_ID = False

            # Compute prototype and attribute codes (for alternate sampling)
            x_a_re_id = self.to_re(Gx_a_raw.clone())
            x_b_re_id = self.to_re(Gx_b_raw.clone())
            x_a_re_id = x_a_re_id.detach()
            x_b_re_id = x_b_re_id.detach()

            c_a_id = self.gen_a.enc_pro(x_a_re_id)
            c_b_id = self.gen_b.enc_pro(x_b_re_id)
            s_a_id = self.gen_a.enc_att(x_a_re_id)
            s_b_id = self.gen_b.enc_att(x_b_re_id)

            x_ab_re_id = self.to_re(x_ab_raw.clone())
            x_ba_re_id = self.to_re(x_ba_raw.clone())
            x_ab_re_id = x_ab_re_id.detach()
            x_ba_re_id = x_ba_re_id.detach()

            c_a_recon_id = self.gen_b.enc_pro(x_ab_re_id)
            c_b_recon_id = self.gen_a.enc_pro(x_ba_re_id)
            s_a_recon_id = self.gen_a.enc_att(x_ba_re_id)
            s_b_recon_id = self.gen_b.enc_att(x_ab_re_id)

            c_a_id = self.backbone_pro(c_a_id)
            s_a_id_raw = s_a_id.clone()
            s_a_id = torch_gather(s_a_id, self.att_style_idx, 1)

            c_a_id_norm = c_a_id.div(torch.norm(c_a_id, p=2, dim=1, keepdim=True).expand_as(c_a_id))
            s_a_id_norm = s_a_id.div(torch.norm(s_a_id, p=2, dim=1, keepdim=True).expand_as(s_a_id))

            self.etc_type['combine_w'] = self.combine_weight.multp.item()
            c_a_id_norm *= min(self.combine_weight.multp, 1.0)
            s_a_id_norm *= max((1.0 - self.combine_weight.multp), 0.0)

            c_b_id = self.backbone_pro(c_b_id)
            s_b_id = torch_gather(s_b_id, self.att_style_idx, 1)

            c_b_id_norm = c_b_id.div(torch.norm(c_b_id, p=2, dim=1, keepdim=True).expand_as(c_b_id))
            s_b_id_norm = s_b_id.div(torch.norm(s_b_id, p=2, dim=1, keepdim=True).expand_as(s_b_id))

            c_b_id_norm *= min(self.combine_weight.multp, 1.0)
            s_b_id_norm *= max((1.0 - self.combine_weight.multp), 0.0)

            c_a_recon_id = self.backbone_pro(c_a_recon_id)
            s_a_recon_id = torch_gather(s_a_recon_id, self.att_style_idx, 1)

            c_a_recon_id_norm = c_a_recon_id.div(torch.norm(c_a_recon_id, p=2, dim=1, keepdim=True).expand_as(c_a_recon_id))
            s_a_recon_id_norm = s_a_recon_id.div(torch.norm(s_a_recon_id, p=2, dim=1, keepdim=True).expand_as(s_a_recon_id))

            c_a_recon_id_norm *= min(self.combine_weight.multp, 1.0)
            s_a_recon_id_norm *= max((1.0 - self.combine_weight.multp), 0.0)

            c_b_recon_id = self.backbone_pro(c_b_recon_id)
            s_b_recon_id = torch_gather(s_b_recon_id, self.att_style_idx, 1)

            c_b_recon_id_norm = c_b_recon_id.div(torch.norm(c_b_recon_id, p=2, dim=1, keepdim=True).expand_as(c_b_recon_id))
            s_b_recon_id_norm = s_b_recon_id.div(torch.norm(s_b_recon_id, p=2, dim=1, keepdim=True).expand_as(s_b_recon_id))

            c_b_recon_id_norm *= min(self.combine_weight.multp, 1.0)
            s_b_recon_id_norm *= max((1.0 - self.combine_weight.multp), 0.0)


            if self.is_neg:

                if opt.G_input_dim == 1:
                    Gy_a = self.single(neg_a.clone())
                    Gy_b = self.single(neg_b.clone())
                else:
                    Gy_a = neg_a.clone()
                    Gy_b = neg_b.clone()

                y_a_re_id = self.to_re(Gy_a.clone())
                y_b_re_id = self.to_re(Gy_b.clone())

                y_a_re_id = y_a_re_id.detach()
                y_b_re_id = y_b_re_id.detach()

                c_a_neg_id = self.gen_a.enc_pro(y_a_re_id)
                c_b_neg_id = self.gen_b.enc_pro(y_b_re_id)
                s_a_neg_id = self.gen_a.enc_att(y_a_re_id)
                s_b_neg_id = self.gen_b.enc_att(y_b_re_id)

                c_a_neg_id = self.backbone_pro(c_a_neg_id)
                c_b_neg_id = self.backbone_pro(c_b_neg_id)

                s_a_neg_id = torch_gather(s_a_neg_id, self.att_style_idx, 1)
                s_b_neg_id = torch_gather(s_b_neg_id, self.att_style_idx, 1)

                c_a_neg_id_norm = c_a_neg_id.div(torch.norm(c_a_neg_id, p=2, dim=1, keepdim=True).expand_as(c_a_neg_id))
                s_a_neg_id_norm = s_a_neg_id.div(torch.norm(s_a_neg_id, p=2, dim=1, keepdim=True).expand_as(s_a_neg_id))
                c_b_neg_id_norm = c_b_neg_id.div(torch.norm(c_b_neg_id, p=2, dim=1, keepdim=True).expand_as(c_b_neg_id))
                s_b_neg_id_norm = s_b_neg_id.div(torch.norm(s_b_neg_id, p=2, dim=1, keepdim=True).expand_as(s_b_neg_id))

                self.etc_type['combine_w'] = self.combine_weight.multp.item()
                c_a_neg_id_norm *= min(self.combine_weight.multp, 1.0)
                s_a_neg_id_norm *= max((1.0 - self.combine_weight.multp), 0.0)
                c_b_neg_id_norm *= min(self.combine_weight.multp, 1.0)
                s_b_neg_id_norm *= max((1.0 - self.combine_weight.multp), 0.0)

            c_all = torch.Tensor().cuda().type(c_a_id_norm.dtype)
            s_all = torch.Tensor().cuda().type(s_a_id_norm.dtype)
            label_all = torch.Tensor().cuda().type(self.labels_a.dtype)
            idx_all = []

            pivot_idx_ce = [1, 2, 3, 4, 5, 6, 7, 8]  # base 1,2,3,4 [5,6] 7,8,9,10
            pivot_idx_trip1 = [1, 3, 7]
            pivot_idx_trip2 = [2, 4, 8]
            target_idx_trip1 = [1, 2, 3, 4, 5, 6, 7, 8]
            target_idx_trip2 = [1, 2, 3, 4, 5, 6, 7, 8]

            samp_idx = pivot_idx_ce.copy()
            samp_idx.extend(pivot_idx_trip1.copy())
            samp_idx.extend(pivot_idx_trip2.copy())
            samp_idx.extend(target_idx_trip1.copy())
            samp_idx.extend(target_idx_trip2.copy())
            samp_idx = list(set(samp_idx))


            if 1 in samp_idx: # pure a [a]
                c_all = torch.cat((c_all, c_a_id_norm), dim=0)  # original
                s_all = torch.cat((s_all, s_a_id_norm), dim=0)
                label_all = torch.cat((label_all, self.labels_a), dim=0)
                idx_all.extend([1]*len(self.labels_a))

            if 2 in samp_idx: # pure b [b]
                c_all = torch.cat((c_all, c_b_id_norm), dim=0)  # original
                s_all = torch.cat((s_all, s_b_id_norm), dim=0)
                label_all = torch.cat((label_all, self.labels_b), dim=0)
                idx_all.extend([2]*len(self.labels_b))

            if 3 in samp_idx: # x_ba (c_b_recon, s_a_recon) [b] a'
                c_all = torch.cat((c_all, c_b_recon_id_norm), dim=0)  # original
                s_all = torch.cat((s_all, s_a_recon_id_norm), dim=0)
                label_all = torch.cat((label_all, self.labels_b), dim=0)
                idx_all.extend([3]*len(self.labels_b))

            if 4 in samp_idx:  # x_ab (c_a_recon, s_b_recon) [a] b'
                c_all = torch.cat((c_all, c_a_recon_id_norm), dim=0)  # original
                s_all = torch.cat((s_all, s_b_recon_id_norm), dim=0)
                label_all = torch.cat((label_all, self.labels_a), dim=0)
                idx_all.extend([4]*len(self.labels_a))

            if self.is_neg:
                if 5 in samp_idx: # neg a [neg_a]
                    c_all = torch.cat((c_all, c_a_neg_id_norm), dim=0)  # original
                    s_all = torch.cat((s_all, s_a_neg_id_norm), dim=0)
                    label_all = torch.cat((label_all, self.labels_neg_a), dim=0)
                    idx_all.extend([5]*len(self.labels_neg_a))

                if 6 in samp_idx: # neg b [neg_b]
                    c_all = torch.cat((c_all, c_b_neg_id_norm), dim=0)  # original
                    s_all = torch.cat((s_all, s_b_neg_id_norm), dim=0)
                    label_all = torch.cat((label_all, self.labels_neg_b), dim=0)
                    idx_all.extend([6]*len(self.labels_neg_b))

            if 7 in samp_idx: # a_uni_comb (a - ba) [a-b] a''
                c_all = torch.cat((c_all, c_a_id_norm, c_b_recon_id_norm), dim=0)  # original
                s_all = torch.cat((s_all, s_a_recon_id_norm, s_a_id_norm), dim=0)
                label_all = torch.cat((label_all, self.labels_a, self.labels_b), dim=0)
                idx_all.extend([7]*len(self.labels_a)*2)

            if 8 in samp_idx: # b_uni_comb (b - ab) [b-a] b''
                c_all = torch.cat((c_all, c_b_id_norm, c_a_recon_id_norm), dim=0)  # original
                s_all = torch.cat((s_all, s_b_recon_id_norm, s_b_id_norm), dim=0)
                label_all = torch.cat((label_all, self.labels_b, self.labels_a), dim=0)
                idx_all.extend([8]*len(self.labels_a)*2)

            if 9 in samp_idx: # b_cross_comb (b - a+ba) [b-a] a'''
                c_all = torch.cat((c_all, c_b_id_norm, c_b_id_norm, c_a_recon_id_norm, c_a_recon_id_norm), dim=0)  # original
                s_all = torch.cat((s_all, s_a_id_norm, s_a_recon_id_norm, s_a_id_norm, s_a_recon_id_norm), dim=0)
                label_all = torch.cat((label_all, self.labels_b, self.labels_b, self.labels_a, self.labels_a), dim=0)
                idx_all.extend([9]*len(self.labels_a)*4)

            if 10 in samp_idx: # a_cross_comb (a - b+ab) [b-a] b'''
                c_all = torch.cat((c_all, c_a_id_norm, c_a_id_norm, c_b_recon_id_norm, c_b_recon_id_norm), dim=0)  # original
                s_all = torch.cat((s_all, s_b_id_norm, s_b_recon_id_norm, s_b_id_norm, s_b_recon_id_norm), dim=0)
                label_all = torch.cat((label_all, self.labels_a, self.labels_a, self.labels_b, self.labels_b), dim=0)
                idx_all.extend([10]*len(self.labels_a)*4)

            if 11 in samp_idx:
                c_all = torch.cat((c_all, c_a_id_norm, c_b_id_norm), dim=0)  # original
                s_all = torch.cat((s_all, s_a_id_norm, s_b_id_norm), dim=0)
                label_all = torch.cat((label_all, self.labels_a, self.labels_a), dim=0)
                idx_all.extend([11] * (len(self.labels_a) * 2))

                if self.is_neg:
                    c_all = torch.cat((c_all, c_a_neg_id_norm, c_b_neg_id_norm), dim=0)  # original
                    s_all = torch.cat((s_all, s_a_neg_id_norm, s_b_neg_id_norm), dim=0)
                    label_all = torch.cat((label_all, self.labels_neg_a, self.labels_neg_b), dim=0)
                    idx_all.extend([11] * (len(self.labels_neg_a) * 2))
            if 12 in samp_idx:
                c_all = torch.cat((c_all, c_a_id_norm, c_b_id_norm, c_a_recon_id_norm, c_b_recon_id_norm, c_b_recon_id_norm, c_a_recon_id_norm), dim=0)  # original
                s_all = torch.cat((s_all, s_a_recon_id_norm, s_b_recon_id_norm, s_b_id_norm, s_a_id_norm, s_a_recon_id_norm, s_b_recon_id_norm), dim=0)
                label_all = torch.cat((label_all, self.labels_a, self.labels_b, self.labels_a, self.labels_b, self.labels_b, self.labels_a), dim=0)
                idx_all.extend([12] * len(self.labels_a) * 6)
            if 13 in samp_idx:
                c_all = torch.cat((c_all, c_a_id_norm, c_b_id_norm, c_a_id_norm, c_b_id_norm, c_a_recon_id_norm, c_b_recon_id_norm, c_a_recon_id_norm, c_b_recon_id_norm), dim=0)  # original
                s_all = torch.cat((s_all, s_b_id_norm, s_a_id_norm, s_b_recon_id_norm, s_a_recon_id_norm, s_a_id_norm, s_b_id_norm, s_a_recon_id_norm, s_b_recon_id_norm), dim=0)
                label_all = torch.cat((label_all, self.labels_a, self.labels_b, self.labels_a, self.labels_b, self.labels_a, self.labels_b, self.labels_a, self.labels_b), dim=0)
                idx_all.extend([13] * len(self.labels_a) * 8)

            f_all = torch.Tensor().cuda()
            f_all = torch.cat((f_all, c_all), dim=1)
            f_all = torch.cat((f_all, s_all), dim=1)

            output_all, _, f_all_triplet = self.id_classifier(f_all)

            pivot_idx_ce = find_array(idx_all, pivot_idx_ce)
            pivot_idx_trip1 = find_array(idx_all, pivot_idx_trip1)
            target_idx_trip1 = find_array(idx_all, target_idx_trip1)
            pivot_idx_trip2 = find_array(idx_all, pivot_idx_trip2)
            target_idx_trip2 = find_array(idx_all, target_idx_trip2)

            output_all_ce = output_all[pivot_idx_ce].clone()
            label_all_ce = label_all[pivot_idx_ce].clone()
            f_all_triplet1 = f_all_triplet[pivot_idx_trip1].clone()
            label_triplet1 = label_all[pivot_idx_trip1].clone()
            f_all_triplet1_target = f_all_triplet[target_idx_trip1].clone()
            label_triplet1_target = label_all[target_idx_trip1].clone()
            f_all_triplet2 = f_all_triplet[pivot_idx_trip2].clone()
            label_triplet2 = label_all[pivot_idx_trip2].clone()
            f_all_triplet2_target = f_all_triplet[target_idx_trip2].clone()
            label_triplet2_target = label_all[target_idx_trip2].clone()

            # CE loss
            num_part = 1
            loss_all, acc_all, loss_cnt = self.apply_CE_loss_between_two_labels(opt, output_all_ce, label_all_ce,
                                                                                num_part, 0.0, 0.0, 0.0)
            self.loss_CE = loss_all
            self.acc_type['CE'] = acc_all / loss_cnt
            self.loss_type['CE'] = self.loss_CE.item()

            # Triplet loss
            loss_flag = opt.ID_TRIP_loss_flag
            w_trip_reg = opt.w_trip_reg
            w_trip = opt.w_trip
            triplet_margin = opt.triplet_margin

            loss_all, acc_all, reg_all, margin_all, pscore_all, nscore_all, loss_cnt = \
                self.apply_triplet_loss_between_features(opt, f_all_triplet1, f_all_triplet1_target, [], label_triplet1.cpu(), \
                                                         label_triplet1_target.cpu(), [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                                                         loss_flag, w_trip_reg, w_trip, triplet_margin)

            loss_all, acc_all, reg_all, margin_all, pscore_all, nscore_all, loss_cnt = \
                self.apply_triplet_loss_between_features(opt, f_all_triplet2, f_all_triplet2_target, [], label_triplet2.cpu(), \
                                                         label_triplet2_target.cpu(), [], loss_all, acc_all, reg_all, margin_all, \
                                                             pscore_all, nscore_all, loss_cnt, \
                                                             loss_flag, w_trip_reg, w_trip, triplet_margin)

            self.loss_trip = loss_all
            self.acc_type['Trip'] = acc_all / loss_cnt
            self.etc_type['Trip_reg'] = reg_all / loss_cnt
            self.etc_type['Trip_margin'] = margin_all / loss_cnt
            self.etc_type['Trip_pscore'] = pscore_all / loss_cnt
            self.etc_type['Trip_nscore'] = nscore_all / loss_cnt
            self.loss_type['TRIP'] = self.loss_trip.item() if self.loss_trip != 0 else 0

            self.loss_id_total = self.loss_CE + self.loss_trip

            if opt.HFL_ratio > 0:
                self.loss_id_total *= opt.HFL_ratio

            self.loss_type['TOT_ID'] = self.loss_id_total.item()

        else:
            try:
                self.loss_type['TOT_ID'] = self.old_loss_type['TOT_ID']
                self.loss_type['CE'] = self.loss_type['CE']
                self.acc_type['CE'] = self.acc_type['CE']
                self.loss_type['TRIP'] = self.loss_type['TRIP']
                self.acc_type['Trip'] = self.acc_type['Trip']
                self.etc_type['Trip_reg'] = self.etc_type['Trip_reg']
                self.etc_type['Trip_margin'] = self.etc_type['Trip_margin']
            except:
                self.loss_type['TOT_ID'] = 0
                self.loss_type['CE'] = 0
                self.acc_type['CE'] = 0
                self.loss_type['TRIP'] = 0
                self.acc_type['Trip'] = 0
                self.etc_type['Trip_reg'] = 0
                self.etc_type['Trip_margin'] = 0

        # Update ID-PIG and HFL

        if Do_gen_update and Do_id_update:
            self.loss_total = self.loss_gen_total + self.loss_id_total
            self.loss_total.backward()
            self.gen_optimizer.step()
            self.zero_grad_G = True
            self.id_optimizer.step()
            self.zero_grad_ID = True
            self.loss_type['TOTAL'] = self.loss_type['TOT_G'] + self.loss_type['TOT_ID'] + self.loss_type['TOT_D']
        elif Do_gen_update:

            self.loss_gen_total.backward()
            self.gen_optimizer.step()
            self.zero_grad_G = True
            try:
                self.loss_type['TOTAL'] = self.loss_type['TOT_G'] + self.old_loss_type['TOT_ID']  + self.loss_type['TOT_D']
            except:
                self.loss_type['TOTAL'] = self.loss_type['TOT_G'] + self.loss_type['TOT_ID']  + self.loss_type['TOT_D']
        elif Do_id_update:
            self.loss_id_total.backward()
            self.gen_optimizer.step()
            self.zero_grad_G = True
            self.id_optimizer.step()
            self.zero_grad_ID = True
            try:
                self.loss_type['TOTAL'] = self.old_loss_type['TOT_G'] + self.loss_type['TOT_ID']  + self.loss_type['TOT_D']
            except:
                self.loss_type['TOTAL'] = self.loss_type['TOT_G'] + self.loss_type['TOT_ID']  + self.loss_type['TOT_D']
        else:
            try:
                self.loss_type['TOTAL'] = self.old_loss_type['TOT_G'] + self.old_loss_type['TOT_ID'] + self.old_loss_type['TOT_D']
            except:
                self.loss_type['TOTAL'] = self.loss_type['TOT_G'] + self.loss_type['TOT_ID'] + self.loss_type['TOT_D']


        if self.case_a == 'RGB':
            self.dis_RGB = self.dis_a
            self.gen_RGB = self.gen_a
        elif self.case_a == 'IR':
            self.dis_IR = self.dis_a
            self.gen_IR = self.gen_a
        else:
            assert(False)


        if self.case_b == 'RGB':
            self.dis_RGB = self.dis_b
            self.gen_RGB = self.gen_b
        elif self.case_b == 'IR':
            self.dis_IR = self.dis_b
            self.gen_IR = self.gen_b
        else:
            assert(False)


    def forward(self, opt, input, modal, cam):

        self.eval()

        modal_np = np.asarray(modal)
        idx_v = np.where(modal_np == 1)
        idx_t = np.where(modal_np == 0)
        x_a = input[idx_v]
        x_b = input[idx_t]

        is_RGB = False
        is_IR = False
        if len(x_a) > 0:
            is_RGB = True
        if len(x_b) > 0:
            is_IR = True

        if opt.G_input_dim == 1:
            Gx_a = self.single(x_a.clone())
            Gx_b = self.single(x_b.clone())
        else:
            Gx_a = x_a.clone()
            Gx_b = x_b.clone()

        Gx_a_raw = Gx_a.clone()
        Gx_b_raw = Gx_b.clone()

        if is_RGB:
            c_a = self.gen_RGB.enc_pro(Gx_a)
        if is_IR:
            c_b = self.gen_IR.enc_pro(Gx_b)

        new_shape = [len(modal)]
        if is_RGB:
            new_shape.extend(c_a.shape[1:])
            c_id = torch.zeros(new_shape, dtype=c_a.dtype)
        else:
            new_shape.extend(c_b.shape[1:])
            c_id = torch.zeros(new_shape, dtype=c_b.dtype)

        if opt.use_gpu:
            c_id = c_id.cuda()
        if is_RGB:
            c_id[idx_v] = c_a
        if is_IR:
            c_id[idx_t] = c_b

        s_mat = []
        s_vec = []

        if is_RGB:
            s_a_mat, s_a = self.gen_RGB.enc_att(Gx_a, flag_raw=True)
        if is_IR:
            s_b_mat, s_b = self.gen_IR.enc_att(Gx_b, flag_raw=True)

        if is_RGB:
            new_shape = [len(modal)]
            new_shape.extend(s_a.shape[1:])
            s_id = torch.zeros(new_shape, dtype=s_a.dtype)
            new_shape = [len(modal)]
            new_shape.extend(s_a_mat.shape[1:])
            s_mat = torch.zeros(new_shape, dtype=s_a_mat.dtype)
        else:
            new_shape = [len(modal)]
            new_shape.extend(s_b.shape[1:])
            s_id = torch.zeros(new_shape, dtype=s_b.dtype)
            new_shape = [len(modal)]
            new_shape.extend(s_b_mat.shape[1:])
            s_mat = torch.zeros(new_shape, dtype=s_b_mat.dtype)

        if opt.use_gpu:
            s_id = s_id.cuda()
            s_mat = s_mat.cuda()
        if is_RGB:
            s_id[idx_v] = s_a
            s_mat[idx_v] = s_a_mat
        if is_IR:
            s_id[idx_t] = s_b
            s_mat[idx_t] = s_b_mat
        s_vec = s_id

        c_mat = []
        c_vec = []

        c_id, c_mat = self.backbone_pro(c_id, multi_output = True)
        c_vec = c_id

        s_id_all = s_id
        s_id_domain = torch_gather(s_id, self.att_illum_idx, 1)
        s_id_share = torch_gather(s_id, self.att_pose_idx, 1)
        s_id_remain = torch_gather(s_id, self.att_ex_idx, 1)
        s_id = torch_gather(s_id, self.att_style_idx, 1)

        c_id_norm = c_id.div(torch.norm(c_id, p=2, dim=1, keepdim=True).expand_as(c_id))
        s_id_norm = s_id.div(torch.norm(s_id, p=2, dim=1, keepdim=True).expand_as(s_id))

        c_id_norm *= min(self.combine_weight.multp, 1.0)
        s_id_norm *= max((1.0 - self.combine_weight.multp), 0.0)

        f0 = torch.Tensor().cuda()
        f0 = torch.cat((f0, c_id_norm), dim=1)
        f0 = torch.cat((f0, s_id_norm), dim=1)

        _, f1, f_triplet = self.id_classifier(f0)

        feature = []
        if 'content' in opt.evaluate_category:
            feature += [c_id]
        if 'style_id' in opt.evaluate_category:
            feature += [s_id]
        if 'style_share' in opt.evaluate_category:
            feature += [s_id_share]
        if 'style_domain' in opt.evaluate_category:
            feature += [s_id_domain]
        if 'style_remain' in opt.evaluate_category:
            feature += [s_id_remain]
        if 'style_all' in opt.evaluate_category:
            feature += [s_id_all]
        if 'f0' in opt.evaluate_category:
            feature += [f0]
        if 'f1' in opt.evaluate_category:
            feature += [f1]
        if 'f_triplet' in opt.evaluate_category:
            feature += [f_triplet]

        # feature = c_id, s_id, f0, f1, f_triplet
        feature_RAM = []
        if opt.test_RAM:
            feature_RAM += [c_mat]
            feature_RAM += [c_vec]
            feature_RAM += [s_mat]
            feature_RAM += [s_vec]

        self.train()
        return feature, feature_RAM


    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample_basic(self, opt, data_sample, flag):
        self.eval()

        if flag == 'train':
            x_a, x_b = data_sample['train_a'], data_sample['train_b']
            x_a = x_a[0:opt.num_sample_basic1_train]
            x_b = x_b[0:opt.num_sample_basic1_train]

        elif flag =='test':
            x_a, x_b = data_sample['test_a'], data_sample['test_b']
            x_a = x_a[0:opt.num_sample_basic1_test]
            x_b = x_b[0:opt.num_sample_basic1_test]


        gen_RGB = self.gen_RGB
        gen_IR = self.gen_IR

        x_a = x_a.cuda()
        x_b = x_b.cuda()

        x_a_recon_all, x_b_recon_all, x_ba_all, x_ab_all, x_aba_all, x_bab_all = [], [], [], [], [], []
        if opt.G_input_dim == 1:
            Gx_a_all = self.single(x_a.clone())
            Gx_b_all = self.single(x_b.clone())
        else:
            Gx_a_all = x_a.clone()
            Gx_b_all = x_b.clone()


        num_image = Gx_a_all.size(0)

        for i in range(num_image):
            Gx_a = Gx_a_all[i].unsqueeze(0)
            Gx_b = Gx_b_all[i].unsqueeze(0)

            c_a = gen_RGB.enc_pro(Gx_a)
            c_b = gen_IR.enc_pro(Gx_b)

            s_a = gen_RGB.enc_att(Gx_a)
            s_b = gen_IR.enc_att(Gx_b)

            s_a, s_b = change_two_index(s_a, s_b, self.att_style_idx, self.att_ex_idx)

            x_ba = gen_RGB.dec(c_b, s_a, gen_RGB.enc_pro.output_dim)
            x_a_recon = gen_RGB.dec(c_a, s_a, gen_RGB.enc_pro.output_dim)

            x_ab = gen_IR.dec(c_a, s_b, gen_IR.enc_pro.output_dim)
            x_b_recon = gen_IR.dec(c_b, s_b, gen_IR.enc_pro.output_dim)

            x_ab_raw = x_ab
            x_ba_raw = x_ba

            c_b_recon = gen_RGB.enc_pro(x_ba)
            c_a_recon = gen_IR.enc_pro(x_ab)

            s_a_recon = gen_RGB.enc_att(x_ba)
            s_b_recon = gen_IR.enc_att(x_ab)

            x_aba = gen_RGB.dec(c_a_recon, s_a, gen_RGB.enc_pro.output_dim)
            x_bab = gen_IR.dec(c_b_recon, s_b, gen_IR.enc_pro.output_dim)

            x_ab_all.append(x_ab_raw)
            x_ba_all.append(x_ba_raw)
            x_aba_all.append(x_aba)
            x_bab_all.append(x_bab)
            x_a_recon_all.append(x_a_recon)
            x_b_recon_all.append(x_b_recon)

        x_ab_all, x_ba_all = torch.cat(x_ab_all), torch.cat(x_ba_all)
        x_aba_all, x_bab_all = torch.cat(x_aba_all), torch.cat(x_bab_all)
        x_a_recon_all, x_b_recon_all = torch.cat(x_a_recon_all), torch.cat(x_b_recon_all)

        Gx_a_all, Gx_b_all = Gx_a_all.cpu(), Gx_b_all.cpu()
        x_ab_all, x_ba_all =  x_ab_all.cpu(), x_ba_all.cpu()
        x_aba_all, x_bab_all = x_aba_all.cpu(), x_bab_all.cpu()
        x_a_recon_all, x_b_recon_all = x_a_recon_all.cpu(), x_b_recon_all.cpu()

        exp1 = Gx_a_all, x_a_recon_all, x_aba_all, x_ba_all, Gx_b_all, x_b_recon_all, x_bab_all, x_ab_all


        if flag == 'train':
            x_a, x_b = data_sample['train_a'], data_sample['train_b']
            x_a = x_a[0:opt.num_sample_basic2_train]
            x_b = x_b[0:opt.num_sample_basic2_train]
            n_a = opt.num_sample_basic2_train
            n_b = opt.num_sample_basic2_train

        elif flag =='test':
            x_a, x_b = data_sample['test_a'], data_sample['test_b']
            x_a = x_a[0:opt.num_sample_basic2_test]
            x_b = x_b[0:opt.num_sample_basic2_test]
            n_a = opt.num_sample_basic2_test
            n_b = opt.num_sample_basic2_test

        x_a = x_a.cuda()
        x_b = x_b.cuda()

        x_trans_all = []
        x_a_new = torch.cat((x_a, x_b), dim=0)
        x_b_new = torch.cat((x_a, x_b), dim=0)
        x_a_modal = ['a']*n_a
        x_a_modal.extend(['b']*n_b)
        x_b_modal = ['a']*n_a
        x_b_modal.extend(['b']*n_b)
        x_b_modal.extend(['sa','za','oa','sb','zb','ob']) # s: style = 0, z: w/b = 0, o: w=1, b=0

        if opt.G_input_dim == 1:
            Gx_a_all = self.single(x_a_new.clone())
            Gx_b_all = self.single(x_b_new.clone())
        else:
            Gx_a_all = x_a_new.clone()
            Gx_b_all = x_b_new.clone()


        zero_image = Gx_b_all[0].unsqueeze(0).clone()
        zero_image[:] = 5000

        for i in range(len(x_a_modal)+1):
            x_trans_local = []
            i -= 1 # for pivot
            if i < 0:
                x_trans_local.append(zero_image)
                for j in range(len(x_b_modal)): # style
                    if ('a' == x_b_modal[j]) or ('b' == x_b_modal[j]):
                        x_trans_local.append(Gx_b_all[j].unsqueeze(0))
                    else:
                        x_trans_local.append(zero_image)

            else:
                Gx_a = Gx_a_all[i].unsqueeze(0)
                Gx_a_raw = Gx_a.clone()

                if x_a_modal[i] == 'a':
                    c_a = gen_RGB.enc_pro(Gx_a)
                elif x_a_modal[i] == 'b':
                    c_a = gen_IR.enc_pro(Gx_a)
                x_trans_local.append(Gx_a_raw)
                for j in range(len(x_b_modal)): # style
                    if j >= len(x_b_new):
                        Gx_b = Gx_b_all[0].unsqueeze(0)
                    else:
                        Gx_b = Gx_b_all[j].unsqueeze(0)

                    if 'a' in x_b_modal[j]:
                        s_b = gen_RGB.enc_att(Gx_b)
                    elif 'b' in x_b_modal[j]:
                        s_b = gen_IR.enc_att(Gx_b)

                    if 's' in x_b_modal[j]:
                        s_b[:] = 0.0

                    if 'a' in x_b_modal[j]:
                        s_a, s_b = change_two_index(s_a, s_b, self.att_style_idx, self.att_ex_idx)
                        if 'z' in x_b_modal[j]:
                            x_trans = gen_RGB.dec(c_a, s_b, gen_RGB.enc_pro.output_dim, 'all_zero')
                        elif 'o' in x_b_modal[j]:
                            x_trans = gen_RGB.dec(c_a, s_b, gen_RGB.enc_pro.output_dim, 'bias_zero')
                        else:
                            x_trans = gen_RGB.dec(c_a, s_b, gen_RGB.enc_pro.output_dim)

                    elif 'b' in x_b_modal[j]:
                        s_a, s_b = change_two_index(s_a, s_b, self.att_style_idx, self.att_ex_idx)
                        if 'z' in x_b_modal[j]:
                            x_trans = gen_IR.dec(c_a, s_b, gen_IR.enc_pro.output_dim, 'all_zero')
                        elif 'o' in x_b_modal[j]:
                            x_trans = gen_IR.dec(c_a, s_b, gen_IR.enc_pro.output_dim, 'bias_zero')
                        else:
                            x_trans = gen_IR.dec(c_a, s_b, gen_IR.enc_pro.output_dim)

                    if x_a_modal[i] != x_b_modal[j]:
                        x_trans_local.append(x_trans)
                    else:
                        x_trans_local.append(zero_image)


            x_trans_all.append(torch.cat(x_trans_local).cpu())

        exp2 = tuple(x_trans_all)

        self.train()
        return exp1, exp2

    def sample_latent_change(self, opt, data_sample, flag):

        self.eval()
        x_trans_all_set = []
        gen_RGB = self.gen_RGB
        gen_IR = self.gen_IR

        if flag == 'train':
            x_a, x_b = data_sample['train_a'], data_sample['train_b']

            x_a_pivot = x_a[0:opt.num_sample_latent_change_pivot_train]
            x_b_pivot = x_b[0:opt.num_sample_latent_change_pivot_train]

            x_a_col = x_a[0:opt.num_sample_latent_change_col_train]
            x_b_col = x_b[0:opt.num_sample_latent_change_col_train]

        elif flag == 'test':
            x_a, x_b = data_sample['test_a'], data_sample['test_b']

            x_a_pivot = x_a[0:opt.num_sample_latent_change_pivot_test]
            x_b_pivot = x_b[0:opt.num_sample_latent_change_pivot_test]

            x_a_col = x_a[0:opt.num_sample_latent_change_col_test]
            x_b_col = x_b[0:opt.num_sample_latent_change_col_test]


        x_a_pivot = x_a_pivot.cuda()
        x_b_pivot = x_b_pivot.cuda()
        x_a_col = x_a_col.cuda()
        x_b_col = x_b_col.cuda()

        if opt.G_input_dim == 1:
            x_a_pivot = self.single(x_a_pivot.clone())
            x_b_pivot = self.single(x_b_pivot.clone())
            x_a_col = self.single(x_a_col.clone())
            x_b_col = self.single(x_b_col.clone())

        # case_all = ['a(id)','p','p,a(id)','a(ex)','a(all)','p,a(ex)','p,a(all)']
        if opt.flag_latent_change == 1:
            case_all = ['pro','pro,a(id)','a(ex)']
        elif opt.flag_latent_change == 2:
            case_all = ['a(id)','pro','pro,a(id)','a(ex)','a(all)','pro,a(ex)','pro,a(all)']
        elif opt.flag_latent_change == 3:
            case_all = ['pro','a(id)','a(m)','a(p)',\
                        'pro,a(id)','pro,a(m)','pro,a(p)','a(id),a(m)', \
                        'a(id),a(p)', 'a(p),a(m)', 'pro,a(id),a(m)','pro,a(id),a(p)',\
                        'pro,a(m),a(p)','a(id),a(m),a(p)','pro,a(id),a(m),a(p)']

        elif opt.flag_latent_change == 4:
            case_all = ['pro', 'pro,a(id)', 'a(p),a(m)', 'a(m)']

        num_case = len(case_all)
        change_case = case_all * (len(x_a_pivot) + len(x_b_pivot))

        for i in range(len(x_a_pivot)):
            for j in range(num_case):
                if i == 0 and j == 0:
                    input1_all = x_a_pivot[0:1]
                else:
                    input1_all = torch.cat((input1_all, x_a_pivot[i:i+1]), dim = 0)

        for i in range(len(x_b_pivot)):
            for j in range(num_case):
                input1_all = torch.cat((input1_all, x_b_pivot[i:i+1]), dim = 0)

        input2_all = torch.cat((x_a_col, x_b_col), dim=0)
        modal1_all = ['a'] * len(x_a_pivot) * num_case + ['b'] * len(x_b_pivot) * num_case
        modal2_all = ['a'] * len(x_a_col) + ['b'] * len(x_b_col)

        input1_a_idx = [k for k in range(len(modal1_all)) if modal1_all[k] == 'a']
        input1_b_idx = [k for k in range(len(modal1_all)) if modal1_all[k] == 'b']
        input2_a_idx = [k for k in range(len(modal2_all)) if modal2_all[k] == 'a']
        input2_b_idx = [k for k in range(len(modal2_all)) if modal2_all[k] == 'b']

        input1_a_all = input1_all[input1_a_idx]
        input1_b_all = input1_all[input1_b_idx]
        input2_a_all = input2_all[input2_a_idx]
        input2_b_all = input2_all[input2_b_idx]

        input1_a_c = gen_RGB.enc_pro(input1_a_all)
        input2_a_c = gen_RGB.enc_pro(input2_a_all)
        input1_b_c = gen_IR.enc_pro(input1_b_all)
        input2_b_c = gen_IR.enc_pro(input2_b_all)

        input1_a_s = self.gen_RGB.enc_att(input1_a_all)
        input2_a_s = self.gen_RGB.enc_att(input2_a_all)
        input1_b_s = self.gen_IR.enc_att(input1_b_all)
        input2_b_s = self.gen_IR.enc_att(input2_b_all)

        new_shape = [len(input1_all)]
        new_shape.extend(input1_a_c.shape[1:])
        input1_c = torch.zeros(new_shape, dtype=input1_a_c.dtype)
        input1_c = input1_c.cuda()
        input1_c[input1_a_idx] = input1_a_c
        input1_c[input1_b_idx] = input1_b_c

        new_shape = [len(input1_all)]
        new_shape.extend(input1_a_s.shape[1:])
        input1_s = torch.zeros(new_shape, dtype=input1_a_s.dtype)
        input1_s = input1_s.cuda()
        input1_s[input1_a_idx] = input1_a_s
        input1_s[input1_b_idx] = input1_b_s

        new_shape = [len(input2_all)]
        new_shape.extend(input2_a_c.shape[1:])
        input2_c = torch.zeros(new_shape, dtype=input2_a_c.dtype)
        input2_c = input2_c.cuda()
        input2_c[input2_a_idx] = input2_a_c
        input2_c[input2_b_idx] = input2_b_c

        new_shape = [len(input2_all)]
        new_shape.extend(input2_a_s.shape[1:])
        input2_s = torch.zeros(new_shape, dtype=input2_a_s.dtype)
        input2_s = input2_s.cuda()
        input2_s[input2_a_idx] = input2_a_s
        input2_s[input2_b_idx] = input2_b_s

        # change_case = ['a(id)','a(ex)','a(all)']

        num1 = len(input1_all)
        num2 = len(input2_all)



        zero_image = x_a_pivot[0].unsqueeze(0).clone()
        zero_image[:] = 5000

        x_trans_all = []
        for i in range(num1 + 1):
            i -= 1
            x_trans_local = []
            if i == -1: # first row
                zero_image = x_a_pivot[0].unsqueeze(0).clone()
                zero_image[:] = 5000
                x_trans_local.append(zero_image)
                for j in range(num2):
                    x_trans_local.append(input2_all[j].unsqueeze(0))
            else:
                x_trans_local.append(input1_all[i].unsqueeze(0))
                for j in range(num2):
                    decoder_flag = modal1_all[i]

                    final_c = input1_c[i].unsqueeze(0).clone()
                    final_s = input1_s[i].unsqueeze(0).clone()

                    if 'pro' in change_case[i]:
                        final_c = input2_c[j].unsqueeze(0).clone()
                    if 'a(id)' in change_case[i]:
                        final_s[:, self.att_style_idx] = torch_gather(input2_s[j].unsqueeze(0).clone(), self.att_style_idx, 1)
                    if 'a(ex)' in change_case[i]:
                        final_s[:, self.att_ex_idx] = torch_gather(input2_s[j].unsqueeze(0).clone(), self.att_ex_idx, 1)
                        decoder_flag = modal2_all[j]
                    if 'a(p)' in change_case[i]:
                        final_s[:, self.att_pose_idx] = torch_gather(input2_s[j].unsqueeze(0).clone(), self.att_pose_idx, 1)
                    if 'a(m)' in change_case[i]:
                        final_s[:, self.att_illum_idx] = torch_gather(input2_s[j].unsqueeze(0).clone(), self.att_illum_idx, 1)
                        decoder_flag = modal2_all[j]
                    if 'a(all)' in change_case[i]:
                        final_s = input2_s[j].unsqueeze(0).clone()
                        decoder_flag = modal2_all[j]

                    if decoder_flag == 'a':
                        result = gen_RGB.dec(final_c, final_s, gen_RGB.enc_pro.output_dim)
                    else:
                        result = gen_IR.dec(final_c, final_s, gen_IR.enc_pro.output_dim)
                    x_trans_local.append(result)
            x_trans_all.append(torch.cat(x_trans_local).cpu())
        x_trans_all = tuple(x_trans_all)
        x_trans_all_set.append(x_trans_all)

        x_trans_all_set = tuple(x_trans_all_set)
        self.train()

        return x_trans_all_set


    def sample_latent_interp(self, opt, data_sample, flag):

        self.eval()

        num_im = 10

        x_trans_all_set = []

        gen_RGB = self.gen_RGB
        gen_IR = self.gen_IR

        if flag == 'train':

            x_a = data_sample['train_a'][opt.visual_pos_idx].unsqueeze(0).clone()
            x_b = data_sample['train_b'][opt.visual_pos_idx].unsqueeze(0).clone()
            x_a_pos = data_sample['train_a_pos'].unsqueeze(0).clone()
            x_b_pos = data_sample['train_b_pos'].unsqueeze(0).clone()
            x_a_neg = data_sample['train_a'][opt.visual_neg_idx].unsqueeze(0).clone()
            x_b_neg = data_sample['train_b'][opt.visual_neg_idx].unsqueeze(0).clone()

        elif flag == 'test':
            x_a = data_sample['test_a'][opt.visual_pos_idx].unsqueeze(0).clone()
            x_b = data_sample['test_b'][opt.visual_pos_idx].unsqueeze(0).clone()
            x_a_pos = data_sample['test_a_pos'].unsqueeze(0).clone()
            x_b_pos = data_sample['test_b_pos'].unsqueeze(0).clone()
            x_a_neg = data_sample['test_a'][opt.visual_neg_idx].unsqueeze(0).clone()
            x_b_neg = data_sample['test_b'][opt.visual_neg_idx].unsqueeze(0).clone()

        # input1_all = torch.cat((x_a, x_b, x_a, x_b, x_a, x_b, x_a, x_b), dim=0).cuda()
        # input2_all = torch.cat((x_a_pos, x_b_pos, x_a_neg, x_b_neg, x_b, x_a, x_b_neg, x_a_neg), dim=0).cuda()
        #
        # modal1_all = ['a','b','a','b','a','b','a','b']
        # modal2_all = ['a','b','a','b','b','a','b','a']

        input1_all = torch.cat((x_a, x_b, x_a, x_b), dim=0).cuda()
        input2_all = torch.cat((x_b, x_a, x_b_neg, x_a_neg), dim=0).cuda()

        modal1_all = ['a','b','a','b']
        modal2_all = ['b','a','b','a']


        if opt.G_input_dim == 1:
            input1_all = self.single(input1_all.clone())
            input2_all = self.single(input2_all.clone())


        input1_a_idx = [k for k in range(len(modal1_all)) if modal1_all[k] == 'a']
        input1_b_idx = [k for k in range(len(modal1_all)) if modal1_all[k] == 'b']
        input2_a_idx = [k for k in range(len(modal2_all)) if modal2_all[k] == 'a']
        input2_b_idx = [k for k in range(len(modal2_all)) if modal2_all[k] == 'b']

        input1_a_all = input1_all[input1_a_idx]
        input1_b_all = input1_all[input1_b_idx]
        input2_a_all = input2_all[input2_a_idx]
        input2_b_all = input2_all[input2_b_idx]

        input1_a_c = gen_RGB.enc_pro(input1_a_all)
        input2_a_c = gen_RGB.enc_pro(input2_a_all)
        input1_b_c = gen_IR.enc_pro(input1_b_all)
        input2_b_c = gen_IR.enc_pro(input2_b_all)

        input1_a_s = self.gen_RGB.enc_att(input1_a_all)
        input2_a_s = self.gen_RGB.enc_att(input2_a_all)
        input1_b_s = self.gen_IR.enc_att(input1_b_all)
        input2_b_s = self.gen_IR.enc_att(input2_b_all)

        new_shape = [len(input1_all)]
        new_shape.extend(input1_a_c.shape[1:])
        input1_c = torch.zeros(new_shape, dtype=input1_a_c.dtype)
        input1_c = input1_c.cuda()
        input1_c[input1_a_idx] = input1_a_c
        input1_c[input1_b_idx] = input1_b_c

        new_shape = [len(input1_all)]
        new_shape.extend(input1_a_s.shape[1:])
        input1_s = torch.zeros(new_shape, dtype=input1_a_s.dtype)
        input1_s = input1_s.cuda()
        input1_s[input1_a_idx] = input1_a_s
        input1_s[input1_b_idx] = input1_b_s

        new_shape = [len(input2_all)]
        new_shape.extend(input2_a_c.shape[1:])
        input2_c = torch.zeros(new_shape, dtype=input2_a_c.dtype)
        input2_c = input2_c.cuda()
        input2_c[input2_a_idx] = input2_a_c
        input2_c[input2_b_idx] = input2_b_c

        new_shape = [len(input2_all)]
        new_shape.extend(input2_a_s.shape[1:])
        input2_s = torch.zeros(new_shape, dtype=input2_a_s.dtype)
        input2_s = input2_s.cuda()
        input2_s[input2_a_idx] = input2_a_s
        input2_s[input2_b_idx] = input2_b_s

        # change_case = ['p','a(id)','a(p)','a(m)','a(ex)','a(all)']
        change_case = ['a(p)','a(m)','a(ex)']

        num1 = len(input1_all)
        num2 = len(input2_all)


        x_trans_all = []
        for k in range(len(input1_all)):

            init_c = input1_c[k].unsqueeze(0).clone()
            init_s = input1_s[k].unsqueeze(0).clone()

            diff_c = (input2_c[k].unsqueeze(0) - input1_c[k].unsqueeze(0)) / num_im
            diff_s = (input2_s[k].unsqueeze(0) - input1_s[k].unsqueeze(0)) / num_im

            for i in range(len(change_case)):
                x_trans_local = []
                x_trans_local.append(input1_all[k].unsqueeze(0))
                x_trans_local.append(input2_all[k].unsqueeze(0))


                for j in range(num_im):
                    final_c = init_c.clone()
                    final_s = init_s.clone()

                    # if 'p' in change_case[i]:
                    #     final_c += diff_c*(j+1)
                    if 'a(id)' in change_case[i]:
                        final_s[:, self.att_style_idx] += torch_gather(diff_s, self.att_style_idx, 1)*(j+1)
                    if 'a(ex)' in change_case[i]:
                        final_s[:, self.att_ex_idx] += torch_gather(diff_s, self.att_ex_idx, 1)*(j+1)
                    if 'a(p)' in change_case[i]:
                        final_s[:, self.att_pose_idx] += torch_gather(diff_s, self.att_pose_idx, 1)*(j+1)
                    if 'a(m)' in change_case[i]:
                        final_s[:, self.att_illum_idx] += torch_gather(diff_s, self.att_illum_idx, 1)*(j+1)
                    if 'a(all)' in change_case[i]:
                        final_s += diff_s*(j+1)

                    result = gen_RGB.dec(final_c, final_s, gen_RGB.enc_pro.output_dim)
                    x_trans_local.append(result)
                x_trans_all.append(torch.cat(x_trans_local).cpu())
        x_trans_all = tuple(x_trans_all)
        x_trans_all_set.append(x_trans_all)

        x_trans_all_set = tuple(x_trans_all_set)
        self.train()

        return x_trans_all_set

    def update_learning_rate(self, opt, phase):

        if phase == opt.phase_train:  ## Training phase
            if self.id_scheduler is not None:
                self.id_scheduler.step()
            if self.dis_scheduler is not None:
                self.dis_scheduler.step()
            if self.gen_scheduler is not None:
                self.gen_scheduler.step()

            self.etc_type['D_lr'] = self.dis_optimizer.param_groups[0]['lr']
            self.etc_type['G_lr'] = self.gen_optimizer.param_groups[0]['lr']
            self.etc_type['ID_lr'] = self.id_optimizer.param_groups[0]['lr']

    ##########################################################################################################
    # Data prosessing
    ##########################################################################################################

    def data_variable(self, opt, data):
        inputs, pos, neg, attribute, attribute_pos, attribute_neg = data
        self.b, self.c, self.h, self.w = inputs.shape
        if self.b == 1:
            self.b = opt.pos_mini_batch
        if opt.use_gpu:
            inputs = Variable(inputs.cuda())
            if len(pos):
                pos = Variable(pos.cuda())
            if len(neg):
                neg = Variable(neg.cuda())
            for i in attribute.keys():
                attribute[i] = Variable(attribute[i].cuda())
            for i in attribute_pos.keys():
                attribute_pos[i] = Variable(attribute_pos[i].cuda())
            for i in attribute_neg.keys():
                attribute_neg[i] = Variable(attribute_neg[i].cuda())
        else:
            inputs = Variable(inputs)
            if len(pos):
                pos = Variable(pos)
            if len(neg):
                neg = Variable(neg)
            for i in attribute.keys():
                attribute[i] = Variable(attribute[i])
            for i in attribute_pos.keys():
                attribute_pos[i] = Variable(attribute_pos[i])
            for i in attribute_neg.keys():
                attribute_neg[i] = Variable(attribute_neg[i])
        labels = attribute['order']

        return inputs, pos, neg, attribute, attribute_pos, attribute_neg, labels

    def apply_CE_loss_between_two_labels(self, opt, pred_labels, labels, num_part, loss_all=0.0, acc_all=0.0, loss_cnt=0.0):
        if num_part > 1:
            part = {}
            sm = nn.Softmax(dim=1)
            for i in range(num_part):
                part[i] = pred_labels[i]
            score = sm(part[0]) + sm(part[1]) + sm(part[2]) + sm(part[3]) + sm(part[4]) + sm(part[5])
            _, pred_idx = torch.max(score.data, 1)
            loss = self.CE_criterion(part[0], labels)
            for i in range(num_part - 1):
                loss += self.CE_criterion(part[i + 1], labels)
            loss /= float(num_part)
        else:
            _, pred_idx = torch.max(pred_labels.data, 1)
            loss = self.CE_criterion(pred_labels, labels)
            loss *= opt.w_CE
        acc = float(torch.sum(pred_idx == labels.data)) / len(labels)

        loss_cnt += 1.0
        acc += acc_all
        loss += loss_all

        return loss, acc, loss_cnt


    def apply_triplet_loss_between_features(self, opt, f, pf, nf, labels, p_labels, n_labels, all_loss = 0.0, \
                                            all_acc = 0.0, all_reg = 0.0, all_margin = 0.0, all_pscore = 0.0, \
                                            all_nscore = 0.0, loss_cnt = 0.0, loss_flag = 1, w_trip_reg = 0, \
                                            w_trip = 0.0, triplet_margin = 0.0 ):

        flag_normalize = True

        is_pos = True if len(p_labels) > 0 else False
        is_neg = True if len(n_labels) > 0 else False

        if is_pos:
            if len(pf.shape) == 3:
                pf = pf.view(pf.size(0), pf.size(1) * pf.size(2))
            if flag_normalize:
                pf_norm = pf.norm(p=2, dim=1, keepdim=True) + 1e-8
                pf = pf.div(pf_norm)
        else:
            if opt.use_gpu:
                pf = torch.tensor([]).cuda()
            else:
                pf = torch.tensor([])
            p_labels = torch.tensor([])

        if is_neg:
            if len(nf.shape) == 3:
                nf = nf.view(nf.size(0), nf.size(1) * nf.size(2))
            if flag_normalize:
                pf_norm = nf.norm(p=2, dim=1, keepdim=True) + 1e-8
                nf = nf.div(pf_norm)
        else:
            if opt.use_gpu:
                nf = torch.tensor([]).cuda()
            else:
                nf = torch.tensor([])
            n_labels = torch.tensor([])

        if len(f.shape) == 3:
            f = f.view(f.size(0), f.size(1) * f.size(2))

        if flag_normalize:
            f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
            f = f.div(f_norm)

        rand = np.random.permutation(pf.size(0)+nf.size(0))  # rand_idx (applying dimension reduction by pooling)

        if is_neg or is_pos:
            all_f = torch.cat((pf, nf), dim=0)  # (num_pos*b)xfeature_dim
            all_f = all_f[rand, :]  # (#dim_reduced_pos)xfeature_dim
        else:
            all_f = f[rand, :]
        if is_neg and is_pos:
            all_labels = torch.cat((p_labels, n_labels), dim=0)  # num_pos x labels
        elif is_neg:
            all_labels = n_labels
        elif is_pos:
            all_labels = p_labels
        else:
            all_labels = labels
        all_labels = all_labels[rand]  # label of dim_reduced_pos

        score = torch.mm(f.data, all_f.transpose(0, 1))  # cosine distance (original*dim_reduced_pos)
        score = score.cpu()
        _, rank = score.sort(dim=1,
                             descending=True)  # sorting for each original sample, score high == similar == hard, rank = location of score elements
        nf_hard = torch.Tensor().cuda()
        pf_hard = torch.Tensor().cuda()
        for k in range(f.size(0)):
            hard = rank[k, :]  # element location sorted in descending order (hardest order, similar order)
            anchor_label = labels[k]  # label of original
            sorted_labels = all_labels[hard]
            neg_index = np.argwhere(np.asarray(sorted_labels) != np.asarray(anchor_label))
            pos_index = np.argwhere(np.asarray(sorted_labels) == np.asarray(anchor_label))
            nf_hard = torch.cat((nf_hard, all_f[hard[neg_index[0]], :]), dim=0) if len(neg_index) > 0 else nf_hard  # the most similar feature in negative samples
            pf_hard = torch.cat((pf_hard, all_f[hard[pos_index[-1]], :]), dim=0) if len(pos_index) > 0 else nf_hard # the most different feature in positive samples
        nscore = torch.sum(f * nf_hard, dim=1) if len(nf_hard) > 0 else torch.zeros(1).cuda() # cosine similarity between neg
        pscore = torch.sum(f * pf_hard, dim=1) if len(pf_hard) > 0 else torch.zeros(1).cuda() # cosine similarity between pos
        if len(pf_hard) == 0:
            print('Warning: positive sample does not exist -> Triplet loss is not applied')
        if len(nf_hard) == 0:
            print('Warning: negative sample does not exist -> Triplet loss is not applied')

        if (len(pf_hard) == 0) or (len(nf_hard) == 0): # no training
            loss = 0.0
            acc = 0.0
            etc_reg = 0.0
            etc_margin = 0.0
        else:

            if loss_flag == 1:
                loss = self.TRIP_criterion(f, pf_hard, nf_hard)
                etc_reg = 0.0
            else:
                etc_reg = torch.sum((1 + nscore) ** 2) / float(len(nscore)) + torch.sum(
                    (-1 + pscore) ** 2) / float(len(pscore))  # **norm**
                etc_reg *= w_trip_reg
                loss = torch.mean(torch.nn.functional.relu(nscore + triplet_margin - pscore))  # **norm**
                loss = loss + etc_reg

            acc = float(torch.sum(pscore > nscore + triplet_margin)) / len(pscore)
            etc_margin = float(torch.mean(pscore - nscore))
            loss *= w_trip

        all_loss += loss
        all_acc += acc
        all_reg += etc_reg
        all_margin += etc_margin
        all_pscore += torch.sum(pscore).item()
        all_nscore += torch.sum(nscore).item()
        loss_cnt += 1

        return all_loss, all_acc, all_reg, all_margin, all_pscore, all_nscore, loss_cnt

    def save(self, opt, epoch):
        # Save generators, discriminators, and optimizers
        if not os.path.isdir(os.path.join(opt.save_dir, 'checkpoints')):
            os.mkdir(os.path.join(opt.save_dir, 'checkpoints'))
        gen_name = os.path.join(opt.save_dir, 'checkpoints', 'gen_{}.pt'.format(str(self.cnt_cumul).zfill(7)))
        dis_name = os.path.join(opt.save_dir, 'checkpoints', 'dis_{}.pt'.format(str(self.cnt_cumul).zfill(7)))
        id_name = os.path.join(opt.save_dir, 'checkpoints', 'id_{}.pt'.format(str(self.cnt_cumul).zfill(7)))
        opt_name = os.path.join(opt.save_dir, 'checkpoints', 'optimizer.pt')

        torch.save({'a': self.gen_RGB.state_dict(), 'b': self.gen_IR.state_dict()}, gen_name)
        torch.save({'a': self.dis_RGB.state_dict(), 'b': self.dis_IR.state_dict()}, dis_name)
        id_dict = {'id': self.id_classifier.state_dict()}
        id_dict['backbone_pro'] = self.backbone_pro.state_dict()
        id_dict['combine_weight'] = self.combine_weight.state_dict()

        torch.save(id_dict, id_name)
        torch.save({'gen': self.gen_optimizer.state_dict(), 'dis': self.dis_optimizer.state_dict(), \
                    'id': self.id_optimizer.state_dict()}, opt_name)



    def resume(self, opt):

        if opt.resume_name == 'last':
            model_name = get_model_list(opt.resume_dir, "id")
        else:
            model_name = os.path.join(opt.resume_dir, 'id_{}.pt'.format(opt.resume_name.zfill(7)))
        state_dict = torch.load(model_name)
        iterations = int(model_name[-10:-3])
        self.id_classifier.load_state_dict(state_dict['id'])
        self.backbone_pro.load_state_dict(state_dict['backbone_pro'])
        self.combine_weight.load_state_dict(state_dict['combine_weight'])

        if opt.resume_name == 'last':
            model_name = get_model_list(opt.resume_dir, 'gen')
        else:
            model_name = os.path.join(opt.resume_dir, 'gen_{}.pt'.format(opt.resume_name.zfill(7)))
        state_dict = torch.load(model_name)
        self.gen_RGB.load_state_dict(state_dict['a'])
        self.gen_IR.load_state_dict(state_dict['b'])

        if opt.resume_name == 'last':
            model_name = get_model_list(opt.resume_dir, 'dis')
        else:
            model_name = os.path.join(opt.resume_dir, 'dis_{}.pt'.format(opt.resume_name.zfill(7)))
        state_dict = torch.load(model_name)
        self.dis_RGB.load_state_dict(state_dict['a'])
        self.dis_IR.load_state_dict(state_dict['b'])

        # Load optimizers
        state_dict = torch.load(os.path.join(opt.resume_dir, 'optimizer.pt'))
        self.id_optimizer.load_state_dict(state_dict['id'])
        self.gen_optimizer.load_state_dict(state_dict['gen'])
        self.dis_optimizer.load_state_dict(state_dict['dis'])

        torch.backends.cuda.cufft_plan_cache.clear()

        self.id_scheduler = lr_scheduler.StepLR(self.id_optimizer, step_size=opt.step_size_bb, gamma=opt.gamma_bb, last_epoch=iterations)
        self.dis_scheduler = lr_scheduler.StepLR(self.dis_optimizer, step_size=opt.step_size_dis,
                                                 gamma=opt.gamma_dis, last_epoch=iterations)
        self.gen_scheduler = lr_scheduler.StepLR(self.gen_optimizer, step_size=opt.step_size_gen,
                                                 gamma=opt.gamma_gen, last_epoch=iterations)

        print('=*'*50)
        print('Resume from iteration {}'.format(iterations))
        print('=*'*50)

        return iterations

    def to_re(self, x):
        out = torch.FloatTensor(x.size(0), x.size(1), x.size(2), x.size(3))
        out = out.cuda()
        for i in range(x.size(0)):
            out[i,:,:,:] = self.rand_erasing(x[i,:,:,:])
        return out



def recover(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    inp = inp.astype(np.uint8)
    return inp

def to_edge(x):
    x = x.data.cpu()
    out = torch.FloatTensor(x.size(0), x.size(2), x.size(3))
    for i in range(x.size(0)):
        xx = recover(x[i,:,:,:])   # 3 channel, 256x128x3
        xx = cv2.cvtColor(xx, cv2.COLOR_RGB2GRAY) # 256x128x1
        xx = cv2.Canny(xx, 10, 200) #256x128
        xx = xx/255.0 - 0.5 # {-0.5,0.5}
        xx += np.random.randn(xx.shape[0],xx.shape[1])*0.1  #add random noise
        xx = torch.from_numpy(xx.astype(np.float32))
        out[i,:,:] = xx
    out = out.unsqueeze(1)
    return out.cuda()

def to_gray(half=False): #simple
    def forward(x):
        x = torch.mean(x, dim=1, keepdim=True)
        if half:
            x = x.half()
        return x
    return forward


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2 * m.num_features:] # remaining


def get_num_adain_params(model):
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2 * m.num_features
    return num_adain_params

def torch_gather(input, index, dim):
    index_cuda = torch.Tensor(index).cuda().long()
    if dim == 1:
        output = input[:, index_cuda].clone()
    return output

def change_two_index(var1, var2, idx1, idx2):
    new_var1 = Variable(torch.Tensor(var1.shape), requires_grad=True).cuda()
    new_var2 = Variable(torch.Tensor(var2.shape), requires_grad=True).cuda()
    new_var1 = new_var1.type(var1.dtype)
    new_var2 = new_var2.type(var2.dtype)
    new_var1[:, idx2] = var1[:, idx2]
    new_var2[:, idx2] = var2[:, idx2]
    new_var1[:, idx1] = var2[:, idx1]
    new_var2[:, idx1] = var1[:, idx1]

    return new_var1, new_var2

def find_array(idx_all, idx_target):
    new_idx = []
    idx_all_np = np.asarray(idx_all)
    for k in range(len(idx_target)):
        find_idx = np.where(idx_all_np == idx_target[k])[0]
        find_idx = find_idx.tolist()
        new_idx.extend(find_idx)
    return new_idx

