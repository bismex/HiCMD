# -*- coding: utf-8 -*-

from __future__ import print_function, division
import matplotlib
matplotlib.use('agg')
from set_option import opt_model
from util_etc import *
from util_test import *
from util_train import *
from data_sampler import *
from reIDmodel_others import *
from trainer import HICMD
from collections import namedtuple
version =  torch.__version__

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0, 1, 2')
parser.add_argument('--flag_exp', default=1, type=int, help='1: original(1~2days), 0: for check (~1hour)')
parser.add_argument('--data_name',default='RegDB_01',type=str, help='RegDB_01 ~ RegDB_10 / SYSU')
parser.add_argument('--data_dir',default='./data/',type=str, help='data dir: e.g. ./data/')
parser.add_argument('--name_output',default='test', type=str, help='output name')

parser.add_argument('--test_only', default=False, type=bool, help='True / False')
parser.add_argument('--test_dir', default='./model/RegDB_01/test/', type=str, help='test_dir: e.g. ./path/')
parser.add_argument('--test_name', default='last', type=str, help='name of test: e.g. last')
parser.add_argument('--resume_dir', default='./model/RegDB_01/test/checkpoints/', type=str, help='resume_dir: e.g. ./path/checkpoints/')
parser.add_argument('--resume_name', default='', type=str, help='name of resume: e.g. last')
opt = parser.parse_args()
opt = opt_model(opt)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)
opt = opt_settings(opt)
dataloaders, dataloaders_train_tsne, old_train_dataloader, data_info, data_sample, opt = data_settings(opt)
opt = opt_test_settings(opt)

trainer = HICMD(opt)
trainer.cnt_cumul = trainer.resume(opt) if len(opt.resume_name) > 0 else trainer.cnt_cumul
pp = print_and_plot(opt)

for epoch in range(opt.num_epoch):
    print('=' * 40 + ' [ Epoch {}/{} ] '.format(epoch + 1, opt.num_epoch) + '=' * 40)
    for phase in opt.phase_exp: # Each epoch has a training & validation & testing phase
        print('[{}] mode is processing..'.format(phase))

        if phase in [opt.phase_train, 'val']:
            if ('val' == phase) and ((epoch + 1) % opt.val_epoch != 0):
                continue # skip validation
            if phase == opt.phase_train:
                trainer.train()  # Set model to training mode
            else:  ## Validation and testing phase
                trainer.eval()  # Set model to evaluate mode
            for cnt, data in enumerate(dataloaders[phase]): # Iterate training or validation each batch

                # main iteration
                trainer.go_train(data, opt, phase, cnt, epoch)
                epoch_cnt = trainer.cnt_cumul

                # Print, record, and plot the training results
                if ((cnt + 1) % opt.cnt_print_loss == 0) or (cnt == len(dataloaders[phase]) - 1):
                    pp.print_info(opt, epoch, cnt, len(dataloaders[phase]) - 1, trainer.loss_type, trainer.acc_type, \
                                  trainer.etc_type, trainer.cnt_cumul)
                if (trainer.cnt_cumul % opt.cnt_draw_plot == 0) or pp.plot_init:
                    pp.record_info(opt, phase, epoch, trainer.loss_type, trainer.acc_type, trainer.etc_type, trainer.cnt_cumul)
                pp.plot_initialization(opt)

                # Draw samples
                if (trainer.cnt_cumul % opt.cnt_draw_samples == 0):
                    with torch.no_grad():
                        train_image_outputs = trainer.sample_basic(opt, data_sample, 'train')
                        test_image_outputs = trainer.sample_basic(opt, data_sample, 'test')
                    if len(train_image_outputs):
                        write_2images(opt, test_image_outputs, os.path.join(opt.save_dir, 'sample_basic'), \
                                      'test_{}'.format(str(trainer.cnt_cumul).zfill(7)))
                        write_2images(opt, train_image_outputs, os.path.join(opt.save_dir, 'sample_basic'), \
                                      'train_{}'.format(str(trainer.cnt_cumul).zfill(7)))

                    if opt.test_latent:
                        with torch.no_grad():
                            train_image_outputs = trainer.sample_latent_change(opt, data_sample, 'train')
                            test_image_outputs = trainer.sample_latent_change(opt, data_sample, 'test')
                        if len(train_image_outputs):
                            write_2images(opt, test_image_outputs,
                                          os.path.join(opt.save_dir, 'sample_latent_change'), \
                                          'test_{}'.format(str(trainer.cnt_cumul).zfill(7)))
                            write_2images(opt, train_image_outputs,
                                          os.path.join(opt.save_dir, 'sample_latent_change'), \
                                          'train_{}'.format(str(trainer.cnt_cumul).zfill(7)))

                    if opt.test_walk:
                        with torch.no_grad():
                            train_image_outputs = trainer.sample_latent_interp(opt, data_sample, 'train')
                            test_image_outputs = trainer.sample_latent_interp(opt, data_sample, 'test')
                        if len(train_image_outputs):
                            write_2images(opt, test_image_outputs,
                                          os.path.join(opt.save_dir, 'sample_interp_change'), \
                                          'test_{}'.format(str(trainer.cnt_cumul).zfill(7)))
                            write_2images(opt, train_image_outputs,
                                          os.path.join(opt.save_dir, 'sample_interp_change'), \
                                          'train_{}'.format(str(trainer.cnt_cumul).zfill(7)))

                # Save the network
                if phase in opt.phase_train:
                    if epoch_cnt % opt.cnt_save_modal == 0:
                        trainer.save(opt, epoch)

                # Drawing plot and saving phase
                if (not opt.test_only) and (phase in opt.phase_train):
                    if epoch_cnt % opt.cnt_draw_plot == 0:
                        pp.draw_and_save_info(opt, epoch)

                trainer.update_learning_rate(opt, phase) # Update learning rate

                if opt.test_exp:
                    break

                # test
                if epoch_cnt % opt.test_cnt == 0:
                    trainer.eval()  # Set model to evaluate mode
                    result, result_RAM, result_multi = extract_test_features(opt, trainer, dataloaders, data_info)
                    if opt.test_tsne:
                        with torch.no_grad():
                            feat_tsne, _ = extract_feature(opt, trainer, dataloaders_train_tsne, 'train_tsne',
                                                           data_info['train_tsne_modal'], data_info['train_tsne_cam'])
                    for k in range(len(result)):
                        result_k = result[k]
                        save_path = save_test_features(opt, epoch_cnt, result_k, result_RAM, result_multi, k)
                        CMC_single, ap_single = evaluate_result(opt, epoch_cnt, result_k, result_RAM, result_multi, save_path, k)
                        pp.record_test_result(CMC_single, ap_single, epoch_cnt, k)
                        pp.draw_and_save_info_test(opt, epoch_cnt, k)
                        if opt.test_tsne:
                            try:
                                draw_tsne_visualization(opt, epoch_cnt, result_k, feat_tsne[k], data_info, k)
                            except:
                                print('error in draw_tsne_visualization')
                    trainer.train()  # Set model to training mode

        else: # for test only
            if opt.test_only:
                epoch_cnt = trainer.cnt_cumul

                # Draw samples
                if opt.test_latent:
                    if not len(opt.resume_name) > 0:
                        print('please check resume name (test_sample)')
                        assert(False)
                    with torch.no_grad():
                        train_image_outputs = trainer.sample_latent_change(opt, data_sample, 'train')
                        test_image_outputs = trainer.sample_latent_change(opt, data_sample, 'test')
                    if len(train_image_outputs):
                        write_2images(opt, test_image_outputs, os.path.join(opt.save_dir, 'sample_latent_change'), \
                                      'test_{}'.format(str(trainer.cnt_cumul).zfill(7)))
                        write_2images(opt, train_image_outputs, os.path.join(opt.save_dir, 'sample_latent_change'), \
                                      'train_{}'.format(str(trainer.cnt_cumul).zfill(7)))

                if opt.test_walk:
                    with torch.no_grad():
                        train_image_outputs = trainer.sample_latent_interp(opt, data_sample, 'train')
                        test_image_outputs = trainer.sample_latent_interp(opt, data_sample, 'test')
                    if len(train_image_outputs):
                        write_2images(opt, test_image_outputs, os.path.join(opt.save_dir, 'sample_interp_change'), \
                                      'test_{}'.format(str(trainer.cnt_cumul).zfill(7)))
                        write_2images(opt, train_image_outputs, os.path.join(opt.save_dir, 'sample_interp_change'), \
                                      'train_{}'.format(str(trainer.cnt_cumul).zfill(7)))

                if opt.test_sample:
                    if not len(opt.resume_name) > 0:
                        print('please check resume name (test_sample)')
                        assert(False)

                    with torch.no_grad():
                        train_image_outputs = trainer.sample_basic(opt, data_sample, 'train')
                        test_image_outputs = trainer.sample_basic(opt, data_sample, 'test')
                    if len(train_image_outputs):
                        write_2images(opt, test_image_outputs, os.path.join(opt.save_dir, 'sample_basic'), \
                                      'test_{}'.format(str(trainer.cnt_cumul).zfill(7)))
                        write_2images(opt, train_image_outputs, os.path.join(opt.save_dir, 'sample_basic'), \
                                      'train_{}'.format(str(trainer.cnt_cumul).zfill(7)))

                # test
                result, result_RAM, result_multi = extract_test_features(opt, trainer, dataloaders, data_info)
                if opt.test_tsne:
                    with torch.no_grad():
                        feat_tsne, _ = extract_feature(opt, trainer, dataloaders_train_tsne, 'train_tsne',
                                                       data_info['train_tsne_modal'], data_info['train_tsne_cam'])
                for k in range(len(result)):
                    result_k = result[k]
                    save_path = save_test_features(opt, epoch_cnt, result_k, result_RAM, result_multi, k)
                    CMC_single, ap_single = evaluate_result(opt, epoch_cnt, result_k, result_RAM, result_multi, save_path, k)
                    pp.record_test_result(CMC_single, ap_single, epoch_cnt, k)
                    if not opt.test_only:
                        pp.draw_and_save_info_test(opt, epoch_cnt, k)
                    if opt.test_tsne:
                        try:
                            draw_tsne_visualization(opt, epoch_cnt, result_k, feat_tsne[k], data_info, k)
                        except:
                            print('error in draw_tsne_visualization')

    time_elapsed = time.time() - pp.since_init
    print('Experiment completes in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

time_elapsed = time.time() - pp.since_init
print('Experiment completes in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
for k in range(len(opt.evaluate_category)):
    print('[{}] Best result: [Epoch {}], Rank 1: {:.2f}, Rank 5: {:.2f}, Rank 10: {:.2f}, Rank 20: {:.2f}, mAP: {:.2f}'
        .format(opt.evaluate_category[k], pp.best_epoch[k], pp.old_rank1[k], pp.old_rank5[k], pp.old_rank10[k], pp.old_rank20[k], pp.old_map[k]))

