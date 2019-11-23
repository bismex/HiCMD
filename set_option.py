from __future__ import print_function, division
from util_etc import pop_after_string

def opt_model(opt):

    if opt.flag_exp == 1:
        num_step = 4
        if 'Reg' in opt.data_name:
            step_size = 25000
        else:
            step_size = 50000
    else:
        num_step = 1
        if 'Reg' in opt.data_name:
            step_size = 25000
        else:
            step_size = 50000

    if opt.flag_exp == 1:
        opt.num_cnt = step_size * num_step
        opt.cnt_draw_plot = 1000
        opt.cnt_draw_samples = 1000
        opt.cnt_print_loss = 10
        opt.cnt_test_print_loss = 10
        opt.test_cnt = 5000
        opt.cnt_save_modal = opt.test_cnt
    else:
        opt.num_cnt = 1000
        opt.cnt_draw_plot = 100
        opt.cnt_draw_samples = 2
        opt.cnt_print_loss = 1
        opt.cnt_test_print_loss = 10
        opt.test_cnt = 500
        opt.cnt_save_modal = 5000

    # ===== Data parameters =====
    opt.h = 256
    opt.w = 128
    opt.pad = 0
    opt.flip = True
    opt.cnt_warmI2I = 0
    opt.cnt_warmID = 0
    opt.apply_pos_cnt = 1000000
    opt.warm_apply_pos_cnt = opt.apply_pos_cnt
    opt.cnt_initialize_pos = 0
    opt.name_samping = 'P_PAIR,N_PAIR'
    # 'P_RAND/N_RAND (pos/neg sample randomly [n])'
    # 'P_MONO/N_MONO (pos/neg sample in cross [n/0])'
    # 'P_CROSS/N_CROSS (pos/neg sample in cross [0/n])'
    # 'P_MULTI1/N_MULTI1 (pos/neg samples in multimodal [n-1/1])'
    # 'P_MULTI2/N_MULTI2 (pos/neg samples in multimodal [1/n-1])'
    # 'P_PAIR/N_PAIR (RGB and IR image total +[n] pairs)'
    opt.color_jitter = False
    opt.skip_last_batch = False
    opt.train_all = True
    opt.type_domain_label = 0
    opt.train_batchsize = 1 # do not change
    opt.pos_mini_batch = 2 # do not change
    opt.neg_mini_batch = 2 # do not change
    opt.samp_pos = 2 # do not change
    opt.samp_neg = 1 # do not change


    # ===== optimization parameters =====
    opt.step_size_bb = step_size
    opt.step_size_dis = step_size
    opt.step_size_gen = step_size
    opt.lr_backbone = 0.001
    opt.lr_dis = 0.0001
    opt.lr_gen = 0.0001
    opt.beta1 = 0.5
    opt.beta2 = 0.999
    opt.gamma_bb = 0.5
    opt.gamma_dis = 0.5
    opt.gamma_gen = 0.5
    opt.momentum = 0.9
    opt.weight_decay_bb = 0.0001
    opt.weight_decay_dis = 0.0001
    opt.weight_decay_gen = 0.0001
    opt.flag_nesterov = True
    opt.random_seed = 777
    opt.train_workers = 4
    opt.flag_mean_plot = False
    opt.flag_synchronize = True
    opt.val_epoch = 0

    # ===== weights =====
    opt.w_recon_x = 50.0
    opt.w_cross_x = 50.0
    opt.w_cycle_x = 50.0
    opt.w_gan = 20.0
    opt.w_style_kl = 1.0
    opt.w_recon_s = 10.0
    opt.w_CE = 1.0
    opt.w_trip = 1.0

    # ===== etc =====
    opt.HFL_ratio = 0
    opt.triplet_margin = 0.5
    opt.w_trip_reg = 1.0
    opt.w_lrelu = 0.3
    opt.droprate = 0.5
    opt.CE_erasing_p = 0.5
    opt.all_erasing_p = 0.0
    opt.att_pose_ratio = 0.5
    opt.att_style_ratio = 0.5
    # opt.evaluate_category = ['content', 'style_id', 'style_share', 'style_domain', 'style_remain', 'style_all', 'f1']
    # opt.evaluate_category = ['content',  'style_id', 'f0', 'f1', 'f_triplet']
    # opt.evaluate_category = ['content',  'style_id', 'f1']
    # opt.evaluate_category = ['f0', 'f1']
    opt.evaluate_category = ['f1']


    # ===== HFL parameters =====
    opt.stride = 2
    opt.ID_TRIP_loss_flag = 0
    opt.backbone_pro_partpool = 1
    opt.backbone_pro_typepool = 'avg'
    opt.backbone_pro_pretrained = True
    opt.backbone_pro_resnet_depth = 50
    opt.backbone_pro_lr_ratio = 1.0
    opt.backbone_pro_max_num_conv = 5
    opt.backbone_pro_max_ouput_dim = 1024  # 1024, 2048
    opt.fc1_channel = 2048 # ID backbone
    opt.fc2_channel = 1024 # for TRIPLET
    opt.ID_norm = 'bn'
    opt.ID_act = 'none'
    opt.combine_weight_lr_ratio = 0.01


    # ===== ID-PIG parameters =====

    # 1) (residual encoder-decoder)
    opt.G_input_dim = 3 # 3:RGB, 1:gray
    opt.G_n_residual = 4 # number of residual blocks in content encoder/decoder
    opt.G_pad_type = 'reflect' # padding type [zero/reflect]
    opt.G_res_type = 'basic' # basic / slim / series / parallel
    opt.G_init = 'kaiming'
    opt.G_w_lrelu = opt.w_lrelu
    opt.G_tanh = False
    opt.G_enc_res_norm = 'in'
    opt.G_non_local = 0
    opt.G_dropout = 0
    opt.G_dim = 32 # number of filters in the bottommost layer
    opt.G_act = 'lrelu' # activation function [relu/lrelu/prelu/selu/tanh]
    opt.G_n_downsamp = 2 # number of down/upsampling layers in content encoder/decoder

    # 2) Encoder
    opt.G_enc_type = 2
    opt.G_ASPP = True
    opt.G_style_partpool = 1
    opt.G_style_typepool = 'avg'
    opt.G_style_resnet_depth = 50
    opt.G_style_pretrained = True
    opt.G_self_style = True

    # 3) Decoder
    opt.G_dec_res_norm = 'adain'
    opt.G_dec_type = 2
    opt.G_mlp_dim = 512 # number of filters in MLP
    opt.G_mlp_n_blk = 3

    # 4) Discriminator
    opt.D_input_dim = opt.G_input_dim
    opt.D_norm = 'none'  # normalization layer [none/bn/in/ln]
    opt.D_act = 'lrelu'  # activation function [relu/lrelu/prelu/selu/tanh]
    opt.D_w_lrelu = opt.w_lrelu
    opt.D_gan_type = 'lsgan'  # GAN loss [lsgan/nsgan]
    opt.D_n_scale = 3  # number of scales
    opt.D_pad_type = 'reflect'  # padding type [zero/reflect]
    opt.D_non_local = 0
    opt.D_init = 'gaussian'
    opt.D_dim = 32  # number of filters in the bottommost layer
    opt.D_n_layer = 2  # number of layers in D
    opt.D_LAMBDA = 0.01
    opt.D_n_res = 4
    opt.D_type = 2 # make_net2()



    # ===== Test options =====
    opt.test_multi = False
    opt.test_ms = '1.0'
    opt.test_batchsize = 64
    opt.test_workers = 16
    opt.test_norm = True
    opt.test_figure = False
    opt.test_tsne = False
    opt.test_tsne_num = 15
    opt.test_RAM = False
    opt.test_gpu = False
    opt.test_exp = False
    opt.test_hist = False
    opt.test_hist_all = True
    opt.test_hist_dist = True
    opt.test_hist_grid = False
    opt.test_hist_grid_min = 0
    opt.test_hist_grid_max = 2
    opt.eval_rerank = False
    opt.test_sample = True
    opt.test_latent = False
    opt.test_walk = False
    opt.save_all_images = False
    opt.save_row_images = False
    opt.save_col_images = False
    opt.test_show_row_num = 8
    opt.test_show_col_num = 10
    opt.visual_last_idx = 0
    opt.visual_pos_idx = 2
    opt.visual_neg_idx = 5
    opt.flag_reverse_figure = False
    opt.flag_all_figure = False
    opt.test_IR2_flag = False
    opt.num_draw_samples_idx_train_a = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20'
    opt.num_draw_samples_idx_train_b = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20'
    opt.num_draw_samples_idx_test_a = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20'
    opt.num_draw_samples_idx_test_b = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20'
    opt.num_sample_basic1_train = 10
    opt.num_sample_basic1_test = 10
    opt.num_sample_basic2_train = 12
    opt.num_sample_basic2_test = 12
    opt.num_sample_latent_change_pivot_train = 2
    opt.num_sample_latent_change_pivot_test = 2
    opt.num_sample_latent_change_col_train = 6
    opt.num_sample_latent_change_col_test = 6
    opt.flag_latent_change = 3
    opt.flag_decoder_hold = False
    opt.num_sample_latent_interp_pivot_train = 3
    opt.num_sample_latent_interp_pivot_test = 3
    opt.num_sample_latent_interp_col_train = 2
    opt.num_sample_latent_interp_col_test = 2

    #---------------------------------------# Loss
    str_samp_name = opt.name_samping.split(',')
    opt.name_samping = []
    pos_flag = False
    neg_flag = False
    for id_samp in str_samp_name:
        if id_samp in ['P_RAND', 'P_MONO', 'P_CROSS', 'P_MULTI1', 'P_MULTI2', 'P_PAIR']:
            opt.name_samping.append(id_samp)
            pos_flag = True
        elif id_samp in ['N_RAND', 'N_MONO', 'N_CROSS', 'N_MULTI1', 'N_MULTI2', 'N_PAIR']:
            opt.name_samping.append(id_samp)
            neg_flag = True
        else:
            print('[{}] Loss name error (not defined)\n'.format(id_samp))
            assert False
    if pos_flag and (opt.samp_pos == 0):
        print('please check opt.pos_name & opt.samp_pos\n')
    if neg_flag and (opt.samp_neg == 0):
        print('please check opt.pos_name & opt.samp_pos\n')

    if opt.train_batchsize == 1:
        opt.bnorm = False
        if opt.pos_mini_batch > 1:
            opt.bnorm = True
    else:
        opt.bnorm = True

    #---------------------------------------# Different setting according to datasets
    if opt.data_name[0:6] == 'Market':
        opt.cross_reid = False
        opt.data_flag = 1
        opt.num_print_rank = 100

    elif opt.data_name[0:5] == 'RegDB':
        opt.cross_reid = True
        opt.test_multi = False
        opt.data_flag = 5
        opt.num_print_rank = 100
        opt.IR_double_model_flag = False

    elif opt.data_name[0:4] == 'SYSU':
        opt.cross_reid = True
        opt.test_multi = False
        opt.data_flag = 6
        opt.num_print_rank = 100
        opt.train_all = False
        opt.val_epoch = 0


    return opt
