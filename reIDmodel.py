import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable, Function
import pretrainedmodels
import pdb
import math

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num = 10, droprate = 0.5, relu=False, bnorm=True, \
                 num_bottleneck=512, add_feature = 0, linear=True, return_f = False,\
                 ID_norm = 'bn', ID_act = 'lrelu', w_lrelu = 0.2):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            if ID_norm == 'bn':
                add_block += [nn.BatchNorm1d(num_bottleneck)]
            elif ID_norm == 'in':
                add_block += [nn.InstanceNorm1d(num_bottleneck)]
            elif ID_norm == 'ln':
                add_block += [LayerNorm(num_bottleneck)]
            elif ID_norm == 'adain':
                add_block += [AdaptiveInstanceNorm2d(num_bottleneck)]
            elif ID_norm == 'none':
                add_block = None
            else:
                assert 0, "Unsupported normalization: {}".format(ID_norm)
        # initialize activation
        if ID_act == 'relu':
            add_block += [nn.ReLU(inplace=True)]
        elif ID_act == 'lrelu':
            add_block += [nn.LeakyReLU(w_lrelu, inplace=True)]
        elif ID_act == 'prelu':
            add_block += [nn.PReLU()]
        elif ID_act == 'selu':
            add_block += [nn.SELU(inplace=True)]
        elif ID_act == 'tanh':
            add_block += [nn.Tanh()]
        elif ID_act == 'none':
            print('.')
        else:
            assert 0, "Unsupported activation: {}".format(ID_act)
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        self.add_feature = add_feature
        if add_feature > 0:
            add_block2 = [nn.Linear(num_bottleneck, add_feature)]
            # add_block2 += [nn.BatchNorm1d(num_bottleneck)]
            # add_block2 += [nn.LeakyReLU(0.1)]
            # add_block2 += [nn.Dropout(p=droprate)]
            add_block2 = nn.Sequential(*add_block2)
            add_block2.apply(weights_init_kaiming)
            self.add_block2 = add_block2

        # linear, BN, dropout, classifier

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x) # [b, 512]

        if self.return_f:
            f = x
            if self.add_feature > 0:
                f2 = self.add_block2(x)
            else:
                f2 = []
            x = self.classifier(x) # [b, num_classes]
            return x, f, f2
        else:
            x = self.classifier(x)
            return x


class style_disentangler(nn.Module):

    def __init__(self, input_dim, bottleneck_dim, bnorm, n_layer, act, w_lrelu, droprate):
        super(style_disentangler, self).__init__()

        add_block = []
        for i in range(n_layer):
            add_block += [nn.Linear(input_dim, bottleneck_dim)]
            if bnorm:
                add_block += [nn.BatchNorm1d(bottleneck_dim)]
            if act == 'relu':
                add_block += [nn.ReLU(inplace=True)]
            elif act == 'lrelu':
                add_block += [nn.LeakyReLU(w_lrelu, inplace=True)]
            elif act == 'prelu':
                add_block += [nn.PReLU()]
            elif act == 'selu':
                add_block += [nn.SELU(inplace=True)]
            elif act == 'tanh':
                add_block += [nn.Tanh()]
            elif act == 'none':
                print('.')
            if droprate:
                add_block += [nn.Dropout(p=droprate)]
            input_dim = bottleneck_dim
            bottleneck_dim = bottleneck_dim // 2

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.fc_layers = add_block

        self.output_dim = input_dim
    def forward(self, x):
        x = self.fc_layers(x)
        return x


class ft_resnet2(nn.Module):
    __factory = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    def __init__(self, input_dim, depth, stride, max_num_conv, max_ouput_dim, \
                 pretrained, partpool, pooltype):
        super(ft_resnet2, self).__init__()

        if depth >= 50:
            self.resnet_small = False
        else:
            self.resnet_small = True

        # ft_net = nn.Module()
        model_ft = ft_resnet2.__factory[depth](pretrained=pretrained)

        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        self.model = model_ft

        self.dim = input_dim
        self.max_num_conv = max_num_conv
        self.max_ouput_dim = max_ouput_dim
        if pow(2, self.max_num_conv) * self.dim > self.max_ouput_dim:
            real_num_conv = round(math.log2(round(self.max_ouput_dim / input_dim)))
            print('+=' * 50)
            print('opt.DC_content_max_num_conv is changed {} to {}'.format(self.max_num_conv, real_num_conv))
            print(
                'Since the output dimension of content encoder is {} and the input dimension of fc layer is {}'.format(
                    self.dim, self.max_ouput_dim))
            print('+=' * 50)
        if self.dim > self.max_ouput_dim:
            print(
                'Error: the output dimension of content encoder [{}] is bigger than the input dimension of fc layer [{}]'.format(
                    self.dim, self.max_ouput_dim))
            assert False
        self.output_dim = min(self.max_ouput_dim, pow(2, self.max_num_conv) * input_dim) * partpool

        self.part = partpool
        if pooltype == 'max':
            self.partpool = nn.AdaptiveMaxPool2d((self.part, 1))
            self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.partpool = nn.AdaptiveAvgPool2d((self.part, 1))
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, multi_output = False):
        if self.resnet_small:
            if self.dim == 64:
                if self.max_num_conv >= 1 and self.max_ouput_dim >= 128:
                    x = self.model.layer2(x) # 64 -> 128
                if self.max_num_conv >= 2 and self.max_ouput_dim >= 256:
                    x = self.model.layer3(x) # 128 -> 256
                if self.max_num_conv >= 3 and self.max_ouput_dim >= 512:
                    x = self.model.layer4(x) # 256 -> 512
            if self.dim == 128:
                if self.max_num_conv >= 1 and self.max_ouput_dim >= 256:
                    x = self.model.layer3(x) # 128 -> 256
                if self.max_num_conv >= 2 and self.max_ouput_dim >= 512:
                    x = self.model.layer4(x) # 256 -> 512

            if self.dim == 256:
                if self.max_num_conv >= 1 and self.max_ouput_dim >= 512:
                    x = self.model.layer4(x) # 256 -> 512
        else:
            if self.dim == 128:
                print('It is not supported! if self.dim == 128')
                assert(False)

            if self.dim == 64:
                if self.max_num_conv >= 1 and self.max_ouput_dim >= 256:
                    x = self.model.layer1(x) # 64 -> 256
                if self.max_num_conv >= 2 and self.max_ouput_dim >= 512:
                    x = self.model.layer2(x) # 256 -> 512
                if self.max_num_conv >= 3 and self.max_ouput_dim >= 1024:
                    x = self.model.layer3(x) # 512 -> 1024
                if self.max_num_conv >= 4 and self.max_ouput_dim >= 2048:
                    x = self.model.layer4(x) # 1024 -> 2048
            if self.dim == 256:
                if self.max_num_conv >= 1 and self.max_ouput_dim >= 512:
                    x = self.model.layer2(x) # 256 -> 512
                if self.max_num_conv >= 2 and self.max_ouput_dim >= 1024:
                    x = self.model.layer3(x) # 512 -> 1024
                if self.max_num_conv >= 3 and self.max_ouput_dim >= 2048:
                    x = self.model.layer4(x) # 1024 -> 1048
            if self.dim == 512:
                if self.max_num_conv >= 1 and self.max_ouput_dim >= 1024:
                    x = self.model.layer3(x) # 512 -> 1024
                if self.max_num_conv >= 2 and self.max_ouput_dim >= 2048:
                    x = self.model.layer4(x) # 1024 -> 2048
            if self.dim == 1024:
                if self.max_num_conv >= 1 and self.max_ouput_dim >= 2048:
                    x = self.model.layer4(x) # 1024 -> 2048

        f_raw = x

        if self.part > 1:
            f = self.partpool(f_raw)  # [b, c, part, 1]
            f = f.view(f.size(0), f.size(1) * self.part)  # [b, part*2048]
        else:
            f = self.avgpool(f_raw)  # [b, 256 1, 1] # 256dim
            f = f.view(f.size(0), f.size(1))  # [b, 256]
        if multi_output:
            return f, f_raw
        else:
            return f

class ft_classifier(nn.Module):

    def __init__(self, input_dim, class_num, droprate, fc1, fc2, bnorm, ID_norm, ID_act, w_lrelu, return_f = True):
        super(ft_classifier, self).__init__()

        self.classifier = ClassBlock(input_dim, class_num, droprate, num_bottleneck=fc1, add_feature=fc2, bnorm=bnorm,
                                     return_f=return_f, ID_norm=ID_norm, ID_act=ID_act, w_lrelu = w_lrelu)


    def forward(self, x, alpha = -1):
        if alpha == -1: # basic
            output, f_fc1, f_fc2 = self.classifier(x)
            return output, f_fc1, f_fc2
        else:
            x = ReverseLayerF.apply(x, alpha) # [128, 800]
            output = self.classifier(x)
            return output

class ft_weight(nn.Module):
    def __init__(self):
        super(ft_weight, self).__init__()
        self.multp = nn.Parameter(torch.tensor(0.5))

class ft_fc(nn.Module):

    def __init__(self, input_dim, output_dim, act, bnorm, droprate, w_lrelu):
        super(ft_fc, self).__init__()

        add_block = []
        add_block += [nn.Linear(input_dim, output_dim)]
        if bnorm:
            add_block += [nn.BatchNorm1d(output_dim)]
        if act == 'relu':
            add_block += [nn.ReLU(inplace=True)]
        elif act == 'lrelu':
            add_block += [nn.LeakyReLU(w_lrelu, inplace=True)]
        elif act == 'prelu':
            add_block += [nn.PReLU()]
        elif act == 'selu':
            add_block += [nn.SELU(inplace=True)]
        elif act == 'tanh':
            add_block += [nn.Tanh()]
        elif act == 'none':
            print('.')
        if droprate:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.fc_layer = add_block

    def forward(self, x):
        x = self.fc_layer(x)
        return x

class ft_resnet(nn.Module):

    __factory = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    def __init__(self, depth, pretrained, stride, partpool, pooltype, input_channel):
        super(ft_resnet, self).__init__()
        model_ft = ft_resnet.__factory[depth](pretrained=pretrained)
        if model_ft.conv1.weight.size(1) != input_channel:
            new_features = nn.Conv2d(input_channel, model_ft.conv1.weight.size(0), \
                                        kernel_size = model_ft.conv1.kernel_size[0], \
                                        stride = model_ft.conv1.stride[0],\
                                        padding = model_ft.conv1.padding[0],\
                                        bias = model_ft.conv1.bias)
            new_features.apply(weights_init_kaiming)
            if input_channel > model_ft.conv1.weight.size(1):
                new_features.weight.data[:, :model_ft.conv1.weight.size(1), :, :] = model_ft.conv1.weight
                if model_ft.conv1.weight.size(1) * 2 == input_channel:
                    new_features.weight.data[:, model_ft.conv1.weight.size(1):, :, :] = model_ft.conv1.weight
            else:
                new_features.weight.data = model_ft.conv1.weight[:, :input_channel, :, :]
            model_ft.conv1 = new_features

        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        self.part = partpool
        if pooltype == 'max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part, 1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part, 1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.model = model_ft
        self.output_dim = 2048 * self.part

    def forward(self, x, flag_raw = False):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)  # [b, 2048, 8, 4]
        x_raw = x
        if self.part > 1:
            x = self.model.partpool(x)  # [b, 2048, 4, 1]
            x = x.view(x.size(0), x.size(1) * self.part)  # [b, part*2048]
        else:
            x = self.model.avgpool(x)  # [b, 2048, 1, 1]
            x = x.view(x.size(0), x.size(1))  # [b, 2048]


        if flag_raw:
            return x_raw, x
        else:
            return x


class ft_domain_classifier(nn.Module):

    def __init__(self, input_dim, bottleneck_dim, output_dim, n_layer, bnorm, droprate, act, w_lrelu):
        super(ft_domain_classifier, self).__init__()

        add_block = []
        for i in range(n_layer):
            add_block += [nn.Linear(input_dim, bottleneck_dim)]
            if bnorm:
                add_block += [nn.BatchNorm1d(bottleneck_dim)]
            if act == 'relu':
                add_block += [nn.ReLU(inplace=True)]
            elif act == 'lrelu':
                add_block += [nn.LeakyReLU(w_lrelu, inplace=True)]
            elif act == 'prelu':
                add_block += [nn.PReLU()]
            elif act == 'selu':
                add_block += [nn.SELU(inplace=True)]
            elif act == 'tanh':
                add_block += [nn.Tanh()]
            elif act == 'none':
                print('.')
            if droprate:
                add_block += [nn.Dropout(p=droprate)]
            input_dim = bottleneck_dim
            bottleneck_dim = bottleneck_dim // 2
        add_block += [nn.Linear(input_dim, output_dim)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.domain_classifier = add_block


    def forward(self, f):
        domain_output = self.domain_classifier(f)
        return domain_output

# Define the ResNet50-based Model
class ft_GRL_classifier(nn.Module):

    def __init__(self, input_dim, bottleneck_dim, output_dim, n_layer, bnorm, droprate, act, w_lrelu):
        super(ft_GRL_classifier, self).__init__()

        add_block = []
        for i in range(n_layer):
            add_block += [nn.Linear(input_dim, bottleneck_dim)]
            if bnorm:
                add_block += [nn.BatchNorm1d(bottleneck_dim)]
            if act == 'relu':
                add_block += [nn.ReLU(inplace=True)]
            elif act == 'lrelu':
                add_block += [nn.LeakyReLU(w_lrelu, inplace=True)]
            elif act == 'prelu':
                add_block += [nn.PReLU()]
            elif act == 'selu':
                add_block += [nn.SELU(inplace=True)]
            elif act == 'tanh':
                add_block += [nn.Tanh()]
            elif act == 'none':
                print('.')
            if droprate:
                add_block += [nn.Dropout(p=droprate)]
            input_dim = bottleneck_dim
            bottleneck_dim = bottleneck_dim // 2
        add_block += [nn.Linear(input_dim, output_dim)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.domain_classifier = add_block


    def forward(self, f, alpha):
        reverse_f = ReverseLayerF.apply(f, alpha) # [128, 800]
        domain_output = self.domain_classifier(reverse_f)

        return domain_output

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None





# Define the ResNet50-based Model
class ft_net(nn.Module):

    __factory = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    def __init__(self, class_num=10, droprate=0.5, stride=2, depth = 50, \
                 pretrained = True, norm = True, input_channel = 3, \
                 fc1 = 1024, fc2 = 0, bnorm = True, ID_norm = 'bn', ID_act = 'lrelu', partpool = 1, pooltype = 'avg', w_lrelu = 0.2):
        super(ft_net, self).__init__()
        model_ft = ft_net.__factory[depth](pretrained=pretrained)
        if model_ft.conv1.weight.size(1) != input_channel:
            new_features = nn.Conv2d(input_channel, model_ft.conv1.weight.size(0), \
                                        kernel_size = model_ft.conv1.kernel_size[0], \
                                        stride = model_ft.conv1.stride[0],\
                                        padding = model_ft.conv1.padding[0],\
                                        bias = model_ft.conv1.bias)
            new_features.apply(weights_init_kaiming)
            if input_channel > model_ft.conv1.weight.size(1):
                new_features.weight.data[:, :model_ft.conv1.weight.size(1), :, :] = model_ft.conv1.weight
                if model_ft.conv1.weight.size(1) * 2 == input_channel:
                    new_features.weight.data[:, model_ft.conv1.weight.size(1):, :, :] = model_ft.conv1.weight
            else:
                new_features.weight.data = model_ft.conv1.weight[:, :input_channel, :, :]
            model_ft.conv1 = new_features

        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)


        self.part = partpool
        if pooltype=='max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate, num_bottleneck=fc1, add_feature=fc2, bnorm = bnorm, return_f = True, ID_norm = ID_norm, ID_act = ID_act, w_lrelu = w_lrelu)
        self.flag_norm = norm
        self.l2norm = Normalize(2)
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x) # [b, 64, 57, 57]
        x = self.model.layer1(x) # [b, 256, 57, 57]
        x = self.model.layer2(x) # [b, 256, 29, 29]
        x = self.model.layer3(x) # [b, 1024, 15, 15]
        f_raw = self.model.layer4(x) # [b, c, 8, 8]
        x = self.model.avgpool(f_raw) # [b, 2048, 1, 1]
        x = x.view(x.size(0), x.size(1)) # [b, 2048]

        if self.part > 1:
            f = self.model.partpool(f_raw)  # [b, c, part, 1]
            f = f.view(f.size(0), f.size(1) * self.part)  # [b, part*2048]
        else:
            f = x
        output, f_fc1, f_fc2 = self.classifier(x)
        # if self.flag_norm:
        #     feature1 = self.l2norm(feature1)
        #     feature2 = self.l2norm(feature2)
        return output, f, f_fc1, f_fc2, f_raw



# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num=10, droprate=0.5, pretrained = True, \
                 norm = True, input_channel = 3, \
                 fc1 = 1024, fc2 = 0, bnorm = True, ID_norm = 'bn', ID_act = 'lrelu', partpool = 1, pooltype = 'avg', w_lrelu = 0.2):
        super().__init__()
        model_ft = models.densenet121(pretrained=pretrained)
        if model_ft.conv1.weight.size(1) != input_channel:
            new_features = nn.Conv2d(input_channel, model_ft.conv1.weight.size(0), \
                                        kernel_size = model_ft.conv1.kernel_size[0], \
                                        stride = model_ft.conv1.stride[0],\
                                        padding = model_ft.conv1.padding[0],\
                                        bias = model_ft.conv1.bias)
            new_features.apply(weights_init_kaiming)
            if input_channel > model_ft.conv1.weight.size(1):
                new_features.weight.data[:, :model_ft.conv1.weight.size(1), :, :] = model_ft.conv1.weight
                if model_ft.conv1.weight.size(1) * 2 == input_channel:
                    new_features.weight.data[:, model_ft.conv1.weight.size(1):, :, :] = model_ft.conv1.weight
            else:
                new_features.weight.data = model_ft.conv1.weight[:, :input_channel, :, :]
            model_ft.conv1 = new_features

        if pooltype=='max':
            model_ft.features.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))


        model_ft.fc = nn.Sequential()

        self.model = model_ft
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock(1024, class_num, droprate, num_bottleneck=fc1, add_feature=fc2, bnorm = bnorm, return_f = True, ID_norm = ID_norm, ID_act = ID_act)
        self.flag_norm = norm
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        output, f_fc1, f_fc2 = self.classifier(x)
        f = x
        f_raw = []
        # if self.flag_norm:
        #     feature1 = self.l2norm(feature1)
        #     feature2 = self.l2norm(feature2)
        return output, f, f_fc1, f_fc2, f_raw

# Define the NAS-based Model
class ft_net_NAS(nn.Module):

    def __init__(self, class_num=10, droprate=0.5, pretrained = True, \
                 norm = True, input_channel = 3, \
                 fc1 = 1024, fc2 = 0, bnorm = True, ID_norm = 'bn', ID_act = 'lrelu', partpool = 1, pooltype = 'avg', w_lrelu = 0.2):
        super().__init__()  
        model_name = 'nasnetalarge' 
        # pip install pretrainedmodels
        if pretrained:
            model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        else:
            model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000)

        if model_ft.conv1.weight.size(1) != input_channel:
            new_features = nn.Conv2d(input_channel, model_ft.conv1.weight.size(0), \
                                        kernel_size = model_ft.conv1.kernel_size[0], \
                                        stride = model_ft.conv1.stride[0],\
                                        padding = model_ft.conv1.padding[0],\
                                        bias = model_ft.conv1.bias)
            new_features.apply(weights_init_kaiming)
            if input_channel > model_ft.conv1.weight.size(1):
                new_features.weight.data[:, :model_ft.conv1.weight.size(1), :, :] = model_ft.conv1.weight
                if model_ft.conv1.weight.size(1) * 2 == input_channel:
                    new_features.weight.data[:, model_ft.conv1.weight.size(1):, :, :] = model_ft.conv1.weight
            else:
                new_features.weight.data = model_ft.conv1.weight[:, :input_channel, :, :]
            model_ft.conv1 = new_features


        self.part = partpool
        if pooltype=='max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 4032
        self.classifier = ClassBlock(4032, class_num, droprate, num_bottleneck=fc1, add_feature=fc2, bnorm = bnorm, return_f = True, ID_norm = ID_norm, ID_act = ID_act, w_lrelu = w_lrelu)
        self.flag_norm = norm
        self.l2norm = Normalize(2)

    def forward(self, x):
        f_raw = self.model.features(x)
        x = self.model.avgpool(f_raw) # [b, 2048, 1, 1]
        x = x.view(x.size(0), x.size(1)) # [b, 2048]

        if self.part > 1:
            f = self.model.partpool(f_raw)  # [b, c, part, 1]
            f = f.view(f.size(0), f.size(1) * self.part)  # [b, part*2048]
        else:
            f = x
        output, f_fc1, f_fc2 = self.classifier(x)

        # if self.flag_norm:
        #     feature1 = self.l2norm(feature1)
        #     feature2 = self.l2norm(feature2)
        return output, f, f_fc1, f_fc2, f_raw
    
# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num = 10, droprate=0.5, pretrained = True, \
                 norm = True, input_channel = 3, \
                 fc1 = 1024, fc2 = 0, bnorm = True, ID_norm = 'bn', ID_act = 'lrelu', partpool = 1, pooltype = 'avg', w_lrelu = 0.2):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=pretrained)

        if model_ft.conv1.weight.size(1) != input_channel:
            new_features = nn.Conv2d(input_channel, model_ft.conv1.weight.size(0), \
                                        kernel_size = model_ft.conv1.kernel_size[0], \
                                        stride = model_ft.conv1.stride[0],\
                                        padding = model_ft.conv1.padding[0],\
                                        bias = model_ft.conv1.bias)
            new_features.apply(weights_init_kaiming)
            if input_channel > model_ft.conv1.weight.size(1):
                new_features.weight.data[:, :model_ft.conv1.weight.size(1), :, :] = model_ft.conv1.weight
                if model_ft.conv1.weight.size(1) * 2 == input_channel:
                    new_features.weight.data[:, model_ft.conv1.weight.size(1):, :, :] = model_ft.conv1.weight
            else:
                new_features.weight.data = model_ft.conv1.weight[:, :input_channel, :, :]
            model_ft.conv1 = new_features

        # avg pooling to global pooling

        self.part = partpool
        if pooltype=='max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num, droprate, num_bottleneck=fc1, add_feature=fc2, bnorm = bnorm, return_f = True, ID_norm = ID_norm, ID_act = ID_act, w_lrelu = w_lrelu)
        self.flag_norm = norm
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        f_raw = x
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = x.view(x.size(0), x.size(1))
        output, f_fc1, f_fc2 = self.classifier(x)

        if self.part > 1:
            f = self.model.partpool(f_raw)  # [b, c, part, 1]
            f = f.view(f.size(0), f.size(1) * self.part)  # [b, part*2048]
        else:
            f = x

        # if self.flag_norm:
        #     feature1 = self.l2norm(feature1)
        #     feature2 = self.l2norm(feature2)
        return output, f, f_fc1, f_fc2, f_raw

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num=10, droprate=0.5, pretrained = True, \
                 norm = True, input_channel = 3, \
                 fc1 = 256, fc2 = 0, bnorm = True, \
                 ID_norm = 'bn', ID_act = 'lrelu', partpool = 1, pooltype = 'avg', w_lrelu = 0.2):
        super(PCB, self).__init__()

        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=pretrained)

        if model_ft.conv1.weight.size(1) != input_channel:
            new_features = nn.Conv2d(input_channel, model_ft.conv1.weight.size(0), \
                                        kernel_size = model_ft.conv1.kernel_size[0], \
                                        stride = model_ft.conv1.stride[0],\
                                        padding = model_ft.conv1.padding[0],\
                                        bias = model_ft.conv1.bias)
            new_features.apply(weights_init_kaiming)
            if input_channel > model_ft.conv1.weight.size(1):
                new_features.weight.data[:, :model_ft.conv1.weight.size(1), :, :] = model_ft.conv1.weight
                if model_ft.conv1.weight.size(1) * 2 == input_channel:
                    new_features.weight.data[:, model_ft.conv1.weight.size(1):, :, :] = model_ft.conv1.weight
            else:
                new_features.weight.data = model_ft.conv1.weight[:, :input_channel, :, :]
            model_ft.conv1 = new_features

        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=droprate)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=droprate, \
                                           relu=False, num_bottleneck=fc1, add_feature=fc2, bnorm = bnorm, return_f = True, ID_norm = ID_norm, ID_act = ID_act, w_lrelu = w_lrelu))

        self.flag_norm = norm
        self.l2norm = Normalize(2)
        # if not self.pretrained:
        #     self.reset_params()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        f_raw = self.model.layer4(x)
        x = self.avgpool(f_raw)
        f = x.view(x.size(0), x.size(1) * self.part)  # [b, part*2048]
        x = x.view(x.size(0),x.size(1),x.size(2))
        x = self.dropout(x)
        part = {}
        predict = {}
        new_feat1 = {}
        new_feat2 = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i], new_feat1[i], new_feat2[i] = c(part[i])
            # new_feature[i] = new_feature[i].view(new_feature[i].size(0),new_feature[i].size(1),1)

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        output = []
        f_fc1 = torch.cat((new_feat1[0], new_feat1[1], new_feat1[2], new_feat1[3], new_feat1[4], new_feat1[5]), 1)
        f_fc2 = torch.cat((new_feat2[0], new_feat2[1], new_feat2[2], new_feat2[3], new_feat2[4], new_feat2[5]), 1)
        for i in range(self.part):
            output.append(predict[i])
        return output, f, f_fc1, f_fc2, f_raw


# Define the AB Model
class ft_netAB(nn.Module):

    def __init__(self, class_num=10, stride=2, droprate=0.5, pool='avg', \
                 norm = True, input_channel = 3, \
                 fc1 = 1024, fc2 = 0, bnorm = True, ID_norm = 'bn', ID_act = 'lrelu', partpool = 1, pooltype = 'avg', w_lrelu = 0.2):
        super(ft_netAB, self).__init__()
        model_ft = models.resnet50(pretrained=True)

        if model_ft.conv1.weight.size(1) != input_channel:
            new_features = nn.Conv2d(input_channel, model_ft.conv1.weight.size(0), \
                                        kernel_size = model_ft.conv1.kernel_size[0], \
                                        stride = model_ft.conv1.stride[0],\
                                        padding = model_ft.conv1.padding[0],\
                                        bias = model_ft.conv1.bias)
            new_features.apply(weights_init_kaiming)
            if input_channel > model_ft.conv1.weight.size(1):
                new_features.weight.data[:, :model_ft.conv1.weight.size(1), :, :] = model_ft.conv1.weight
                if model_ft.conv1.weight.size(1) * 2 == input_channel:
                    new_features.weight.data[:, model_ft.conv1.weight.size(1):, :, :] = model_ft.conv1.weight
            else:
                new_features.weight.data = model_ft.conv1.weight[:, :input_channel, :, :]
            model_ft.conv1 = new_features

        self.part = partpool
        if pooltype=='max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.model = model_ft

        if stride == 1:
            self.model.layer4[0].downsample[0].stride = (1,1)
            self.model.layer4[0].conv2.stride = (1,1)

        self.classifier1 = ClassBlock(2048, class_num, num_bottleneck=fc1, add_feature=fc2, bnorm = bnorm, droprate=0.5, ID_norm = ID_norm, ID_act = ID_act, w_lrelu = w_lrelu)
        self.classifier2 = ClassBlock(2048, class_num, num_bottleneck=fc1, add_feature=fc2, bnorm = bnorm, droprate=0.75, ID_norm = ID_norm, ID_act = ID_act, w_lrelu = w_lrelu)
        self.flag_norm = norm
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f = self.model.partpool(x)
        f = f.view(f.size(0),f.size(1)*self.part)
        f = f.detach() # no gradient
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x=[]
        x.append(x1)
        x.append(x2)
        return f, x


class verif_net(nn.Module):
    def __init__(self, num_bottleneck):
        super(verif_net, self).__init__()
        self.classifier = ClassBlock(512, class_num = 2, droprate=0.75, \
                                     num_bottleneck = num_bottleneck, add_feature=0, relu=False, return_f = False)
    def forward(self, x):
        x = self.classifier.classifier(x)
        return x

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
        running_mean = self.running_mean.repeat(b).type_as(x)
        running_var = self.running_var.repeat(b).type_as(x)
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, fp16=False):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.fp16 = fp16
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.type() == 'torch.cuda.HalfTensor':  # For Safety
            mean = x.view(-1).float().mean().view(*shape)
            std = x.view(-1).float().std().view(*shape)
            mean = mean.half()
            std = std.half()
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it.
    train_flag = True
    input = Variable(torch.FloatTensor(8, 6, 256, 128))
    if train_flag:
        model = ft_net(751, input_channel = 6)
        # model.classifier = nn.Sequential()
        # model.classifier.classifier = nn.Sequential()
        print(model)
        output, f, _, _ = model(input)
        print('net output size:')
        print(output.shape) # [8, 2048]
    else:
        model = ft_net(751)
        model.classifier.classifier = nn.Sequential()
        print(model)
        model.eval()
        feature, output = model(input)
        print('net output size:')
        print(output.shape) # [8, 512]

