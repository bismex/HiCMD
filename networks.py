"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch import nn
from torch.autograd import Variable
from util_train import weights_init
from reIDmodel import *
import torch
import torch.nn.functional as F
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

##################################################################################
# Discriminator
##################################################################################

class Discriminator(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.input_dim = opt.D_input_dim
        self.dim = opt.D_dim
        self.norm = opt.D_norm
        self.activ = opt.D_act
        self.n_layer = opt.D_n_layer
        self.gan_type = opt.D_gan_type
        self.num_scales = opt.D_n_scale
        self.pad_type = opt.D_pad_type
        self.w_lrelu = opt.D_w_lrelu

        self.non_local = opt.D_non_local
        self.n_res = opt.D_n_res
        self.LAMBDA = opt.D_LAMBDA
        init = opt.D_init

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        if self.gan_type == 'wgan':
             self.cnn = self.one_cnn()
        else:
            self.cnns = nn.ModuleList()
            for _ in range(self.num_scales):
                if opt.D_type == 1:
                    Dis = self._make_net()
                    Dis.apply(weights_init(init))
                    self.cnns.append(Dis)
                elif opt.D_type == 2:
                    Dis = self._make_net2()
                    Dis.apply(weights_init(init))
                    self.cnns.append(Dis)



    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type, w_lrelu = self.w_lrelu)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type, w_lrelu = self.w_lrelu)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def _make_net2(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 1, 1, 0, norm=self.norm, activation=self.activ, pad_type=self.pad_type, w_lrelu = self.w_lrelu)]
        cnn_x += [Conv2dBlock(dim, dim, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type, w_lrelu = self.w_lrelu)]
        cnn_x += [Conv2dBlock(dim, dim, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type, w_lrelu = self.w_lrelu)]
        for i in range(self.n_layer - 1):
            dim2 = min(dim*2, 512)
            cnn_x += [Conv2dBlock(dim, dim, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type, w_lrelu = self.w_lrelu)]
            cnn_x += [Conv2dBlock(dim, dim2, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type, w_lrelu = self.w_lrelu)]
            dim = dim2
        if self.non_local>1:
            cnn_x += [NonlocalBlock(dim)]
        for i in range(self.n_res):
            cnn_x += [ResBlock(dim, norm=self.norm, activation=self.activ, pad_type=self.pad_type, res_type='basic', w_lrelu = self.w_lrelu)]
        if self.non_local>0:
            cnn_x += [NonlocalBlock(dim)]
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def one_cnn(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type, w_lrelu = self.w_lrelu)]
        for i in range(5):
            dim2 = min(dim*2, 512)
            cnn_x += [Conv2dBlock(dim, dim2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type, w_lrelu = self.w_lrelu)]
            dim = dim2
        cnn_x += [nn.Conv2d(dim, 1, (4,2), 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        if not self.gan_type == 'wgan':
            outputs = []
            for model in self.cnns:
                outputs.append(model(x))
                x = self.downsample(x)
        else:
             outputs = self.cnn(x)
             outputs = torch.squeeze(outputs)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        if self.LAMBDA > 0:
            input_real.requires_grad_()
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0
        reg = 0
        Drift = 0.001
        LAMBDA = self.LAMBDA


        if self.gan_type == 'wgan':
            loss += torch.mean(outs0) - torch.mean(outs1)
            # progressive gan
            loss += Drift*( torch.sum(outs0**2) + torch.sum(outs1**2))
            alpha = torch.FloatTensor(input_fake.shape).uniform_(0., 1.)
            alpha = alpha.cuda()
            differences = input_fake - input_real
            interpolates =  Variable(input_real + (alpha*differences), requires_grad=True)
            dis_interpolates = self.forward(interpolates)
            gradient_penalty = self.compute_grad2(dis_interpolates, interpolates).mean()
            loss += LAMBDA*gradient_penalty
            return loss

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
                if self.LAMBDA > 0:
                    reg += LAMBDA* self.compute_grad2(out1, input_real).mean()
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        loss = loss + reg

        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        Drift = 0.001
        if self.gan_type == 'wgan':
            loss += -torch.mean(outs0)
            # progressive gan
            loss += Drift*torch.sum(outs0**2)
            return loss

        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg

##################################################################################
# Encoder and Decoders
##################################################################################

class ResidualEncoder(nn.Module):
    def __init__(self, n_downsample = 0, n_res = 0, input_dim = 0, bottleneck_dim = 0, norm='in', \
                 activ='relu', pad_type='reflect', tanh = False, res_type = 'basic', enc_type = 1, \
                 flag_ASPP = False, init='kaiming', w_lrelu = 0.2):
        super(ResidualEncoder, self).__init__()
        self.model1 = []
        self.model2 = []
        self.model3 = []
        if enc_type == 1:
            self.model += [Conv2dBlock(input_dim, bottleneck_dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type, w_lrelu = w_lrelu)]
            # downsampling blocks
            for i in range(n_downsample):
                self.model += [Conv2dBlock(bottleneck_dim, 2 * bottleneck_dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type, w_lrelu = w_lrelu)]
                bottleneck_dim *= 2
        elif enc_type == 2:
            self.model1 += [Conv2dBlock(input_dim, bottleneck_dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type, w_lrelu = w_lrelu)]
            self.model1 += [Conv2dBlock(bottleneck_dim, 2*bottleneck_dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type, w_lrelu = w_lrelu)]
            bottleneck_dim *=2 # 32dim
            # downsampling blocks
            for i in range(n_downsample-1):
                self.model2 += [Conv2dBlock(bottleneck_dim, bottleneck_dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type, w_lrelu = w_lrelu)]
                self.model2 += [Conv2dBlock(bottleneck_dim, 2 * bottleneck_dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type, w_lrelu = w_lrelu)]
                bottleneck_dim *= 2

        self.model1 = nn.Sequential(*self.model1)
        self.model2 = nn.Sequential(*self.model2)

        # residual blocks
        self.model3 += [ResBlocks(n_res, bottleneck_dim, norm=norm, activation=activ, pad_type=pad_type, res_type=res_type, w_lrelu = w_lrelu)]
        if flag_ASPP:
            self.model3 += [ASPP(bottleneck_dim, norm=norm, activation=activ, pad_type=pad_type, w_lrelu = w_lrelu)]
            bottleneck_dim *= 2
        if tanh:
            self.model3 +=[nn.Tanh()]
        self.model3 = nn.Sequential(*self.model3)
        self.output_dim = bottleneck_dim

        self.apply(weights_init(init))


    def forward(self, x):

        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)

        return x

class ResidualDecoder(nn.Module):
    def __init__(self, n_upsample = 0, n_res = 0, input_dim = 0, output_dim = 0, dropout=0, res_norm='adain', activ='relu', pad_type='zero', res_type='basic', non_local=False, dec_type = 1, init='kaiming', w_lrelu = 0.2, mlp_input = 0, mlp_output = 0, mlp_dim = 0, mlp_n_blk = 0, mlp_norm = 'none', mlp_activ = ''):
        super(ResidualDecoder, self).__init__()

        self.model = []
        if dropout > 0:
            self.model += [nn.Dropout(p = dropout)]
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, input_dim, res_norm, activ, pad_type=pad_type, res_type=res_type, w_lrelu = w_lrelu)]
        if non_local>0:
            self.model += [NonlocalBlock(input_dim)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(input_dim, input_dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type, w_lrelu = w_lrelu)]
            input_dim //= 2
        # use reflection padding in the last conv layer
        if dec_type == 1:
            self.model += [Conv2dBlock(input_dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type, w_lrelu = w_lrelu)]
        elif dec_type == 2:
            self.model += [Conv2dBlock(input_dim, input_dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type, w_lrelu = w_lrelu)]
            self.model += [Conv2dBlock(input_dim, input_dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type, w_lrelu = w_lrelu)]
            self.model += [Conv2dBlock(input_dim, output_dim, 1, 1, 0, norm='none', activation='none', pad_type=pad_type, w_lrelu = w_lrelu)]
        self.model = nn.Sequential(*self.model)

        self.mlp_w1 = MLP(input_dim=mlp_input, output_dim=mlp_output, bottleneck_dim=mlp_dim, n_blk=mlp_n_blk, norm=mlp_norm, activ=mlp_activ,  w_lrelu=w_lrelu)
        self.mlp_w2 = MLP(input_dim=mlp_input, output_dim=mlp_output, bottleneck_dim=mlp_dim, n_blk=mlp_n_blk, norm=mlp_norm, activ=mlp_activ,  w_lrelu=w_lrelu)
        self.mlp_w3 = MLP(input_dim=mlp_input, output_dim=mlp_output, bottleneck_dim=mlp_dim, n_blk=mlp_n_blk, norm=mlp_norm, activ=mlp_activ,  w_lrelu=w_lrelu)
        self.mlp_w4 = MLP(input_dim=mlp_input, output_dim=mlp_output, bottleneck_dim=mlp_dim, n_blk=mlp_n_blk, norm=mlp_norm, activ=mlp_activ,  w_lrelu=w_lrelu)
        self.mlp_b1 = MLP(input_dim=mlp_input, output_dim=mlp_output, bottleneck_dim=mlp_dim, n_blk=mlp_n_blk, norm=mlp_norm, activ=mlp_activ,  w_lrelu=w_lrelu)
        self.mlp_b2 = MLP(input_dim=mlp_input, output_dim=mlp_output, bottleneck_dim=mlp_dim, n_blk=mlp_n_blk, norm=mlp_norm, activ=mlp_activ,  w_lrelu=w_lrelu)
        self.mlp_b3 = MLP(input_dim=mlp_input, output_dim=mlp_output, bottleneck_dim=mlp_dim, n_blk=mlp_n_blk, norm=mlp_norm, activ=mlp_activ,  w_lrelu=w_lrelu)
        self.mlp_b4 = MLP(input_dim=mlp_input, output_dim=mlp_output, bottleneck_dim=mlp_dim, n_blk=mlp_n_blk, norm=mlp_norm, activ=mlp_activ,  w_lrelu=w_lrelu)


        self.apply(weights_init(init))

    def forward(self, x, code, output_dim, flag = ''):

        id_dim = round(code.size(1) / 4)
        ID1 = code[:, :id_dim]
        ID2 = code[:, id_dim:2 * id_dim]
        ID3 = code[:, 2 * id_dim:3 * id_dim]
        ID4 = code[:, 3 * id_dim:]

        adain_params_w = torch.cat( (self.mlp_w1(ID1), self.mlp_w2(ID2), self.mlp_w3(ID3), self.mlp_w4(ID4)), 1)
        adain_params_b = torch.cat( (self.mlp_b1(ID1), self.mlp_b2(ID2), self.mlp_b3(ID3), self.mlp_b4(ID4)), 1)

        if flag == 'all_zero':
            adain_params_w[:] = 0
            adain_params_b[:] = 0
        if flag == 'bias_zero':
            adain_params_w[:] = 1
            adain_params_b[:] = 0

        self.assign_adain_params(adain_params_w, adain_params_b, self.model, output_dim)

        output = self.model(x)
        return output


    def assign_adain_params(self, adain_params_w, adain_params_b, model, dim):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                # print('.')
                # print(adain_params_b.size(1))
                mean = adain_params_b[:,:dim].contiguous()
                std = adain_params_w[:,:dim].contiguous()
                m.bias = mean.view(-1)
                m.weight = std.view(-1)
                if adain_params_w.size(1)>dim :  #Pop the parameters
                    adain_params_b = adain_params_b[:,dim:]
                    adain_params_w = adain_params_w[:,dim:]

##################################################################################
# Basic Blocks
##################################################################################

class MLP(nn.Module):
    def __init__(self, input_dim=0, output_dim=0, bottleneck_dim=0, n_blk=0, norm='none', activ='relu', w_lrelu = 0.2):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, bottleneck_dim, norm=norm, activation=activ, w_lrelu = w_lrelu)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(bottleneck_dim, bottleneck_dim, norm=norm, activation=activ, w_lrelu = w_lrelu)]
        self.model += [LinearBlock(bottleneck_dim, output_dim, norm='none', activation='none', w_lrelu = w_lrelu)]  # no output activations
        self.model = nn.Sequential(*self.model)

        # self.apply(weights_init(init))

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))




class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', res_type='basic', w_lrelu = 0.2):
        super(ResBlocks, self).__init__()
        self.model = []
        self.res_type = res_type
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, res_type=res_type, w_lrelu = w_lrelu)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm, activation='relu', pad_type='zero', res_type='basic', w_lrelu = 0.2):
        super(ResBlock, self).__init__()

        model = []
        if res_type=='basic' or res_type=='nonlocal':
            model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, w_lrelu = w_lrelu)]
            model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type, w_lrelu = w_lrelu)]
        elif res_type=='slim':
            dim_half = dim//2
            model += [Conv2dBlock(dim ,dim_half, 1, 1, 0, norm='in', activation=activation, pad_type=pad_type, w_lrelu = w_lrelu)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, w_lrelu = w_lrelu)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, w_lrelu = w_lrelu)]
            model += [Conv2dBlock(dim_half, dim, 1, 1, 0, norm='in', activation='none', pad_type=pad_type, w_lrelu = w_lrelu)]
        elif res_type=='series':
            model += [Series2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, w_lrelu = w_lrelu)]
            model += [Series2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type, w_lrelu = w_lrelu)]
        elif res_type=='parallel':
            model += [Parallel2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, w_lrelu = w_lrelu)]
            model += [Parallel2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type, w_lrelu = w_lrelu)]
        else:
            ('unkown block type')
        self.res_type = res_type
        self.model = nn.Sequential(*model)
        if res_type=='nonlocal':
            self.nonloc = NonlocalBlock(dim)

    def forward(self, x):
        if self.res_type == 'nonlocal':
            x = self.nonloc(x)
        residual = x
        out = self.model(x)
        out += residual
        return out


class ASPP(nn.Module):
    # ASPP (a)
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', w_lrelu = 0.2):
        super(ASPP, self).__init__()
        dim_part = dim//2
        self.conv1 = Conv2dBlock(dim,dim_part, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type, w_lrelu = w_lrelu)

        self.conv6 = []
        self.conv6 += [Conv2dBlock(dim,dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type, w_lrelu = w_lrelu)]
        self.conv6 += [Conv2dBlock(dim_part,dim_part, 3, 1, 3, norm=norm, activation='none', pad_type=pad_type, dilation=3, w_lrelu = w_lrelu)]
        self.conv6 = nn.Sequential(*self.conv6)

        self.conv12 = []
        self.conv12 += [Conv2dBlock(dim,dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type, w_lrelu = w_lrelu)]
        self.conv12 += [Conv2dBlock(dim_part,dim_part, 3, 1, 6, norm=norm, activation='none', pad_type=pad_type, dilation=6, w_lrelu = w_lrelu)]
        self.conv12 = nn.Sequential(*self.conv12)

        self.conv18 = []
        self.conv18 += [Conv2dBlock(dim,dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type, w_lrelu = w_lrelu)]
        self.conv18 += [Conv2dBlock(dim_part,dim_part, 3, 1, 9, norm=norm, activation='none', pad_type=pad_type, dilation=9, w_lrelu = w_lrelu)]
        self.conv18 = nn.Sequential(*self.conv18)

        self.fuse = Conv2dBlock(4*dim_part,2*dim, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type, w_lrelu = w_lrelu)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv6 = self.conv6(x)
        conv12 = self.conv12(x)
        conv18 = self.conv18(x)
        out = torch.cat((conv1,conv6,conv12, conv18), dim=1)
        out = self.fuse(out)
        return out

class NonlocalBlock(nn.Module):
    def __init__(self, in_dim, norm='in'):
        super(NonlocalBlock, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

class Series2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', w_lrelu = 0.2):
        super(Series2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(w_lrelu, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

        self.instance_norm = nn.InstanceNorm2d(norm_dim)

    def forward(self, x):
        x = self.conv(self.pad(x))
        x = self.norm(x) + x
        x = self.instance_norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class Parallel2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', w_lrelu = 0.2):
        super(Parallel2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(w_lrelu, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

        self.instance_norm = nn.InstanceNorm2d(norm_dim)

    def forward(self, x):
        x = self.conv(self.pad(x)) + self.norm(x)
        x = self.instance_norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1, w_lrelu = 0.2):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(w_lrelu, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu', w_lrelu = 0.2):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(w_lrelu, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        # self.weight = self.weight[:b*c]

        # out = F.batch_norm(
        #     x_reshaped, running_mean, running_var, self.weight[:b*c], self.bias[:b*c],
        #     True, self.momentum, self.eps)
        # self.weight = self.weight[b*c:]
        # self.bias = self.bias[b*c:]

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)