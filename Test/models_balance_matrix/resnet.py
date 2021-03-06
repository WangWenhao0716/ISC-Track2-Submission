from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import random

from .gem import GeneralizedMeanPoolingP

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

class Waveblock(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(0.3 * h)
            sx = random.randint(0, h-rh)
            mask = (x.new_ones(x.size()))*1.5
            mask[:, :, sx:sx+rh, :] = 1
            x = x * mask 
        return x

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=False, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 dev = None):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        gap = GeneralizedMeanPoolingP() #nn.AdaptiveAvgPool2d(1)
        print("The init norm is ",gap)
        waveblock = Waveblock()
        
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, resnet.relu,
            resnet.layer1,
            resnet.layer2, waveblock,
            resnet.layer3, waveblock,
            resnet.layer4, gap
        ).cuda()
        
        if not self.cut_at_pooling:
            self.num_features = 2048*4
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                projecter = nn.Sequential(
                                nn.Linear(2048, 4096),
                                nn.BatchNorm1d(4096),
                                nn.LeakyReLU(0.2, inplace = True),
                                nn.Linear(4096, 2048*4)
                )
                
                matrix = nn.Sequential(
                                nn.BatchNorm1d(8192),
                                nn.LeakyReLU(0.2, inplace = True),
                                nn.Linear(8192, 256)
                )
                assert num_classes % 4 == 0
                    
                self.classifier_0 = nn.Linear(self.num_features, self.num_classes//4, bias=False).cuda()
                init.normal_(self.classifier_0.weight, std=0.001)
                self.classifier_1 = nn.Linear(self.num_features, self.num_classes//4, bias=False).cuda()
                init.normal_(self.classifier_1.weight, std=0.001)
                self.classifier_2 = nn.Linear(self.num_features, self.num_classes//4, bias=False).cuda()
                init.normal_(self.classifier_2.weight, std=0.001)
                self.classifier_3 = nn.Linear(self.num_features, self.num_classes//4, bias=False).cuda()
                init.normal_(self.classifier_3.weight, std=0.001)
                
                self.classifier_0_1 = nn.Linear(256, self.num_classes//4, bias=False).cuda()
                init.normal_(self.classifier_0_1.weight, std=0.001)
                self.classifier_1_1 = nn.Linear(256, self.num_classes//4, bias=False).cuda()
                init.normal_(self.classifier_1_1.weight, std=0.001)
                self.classifier_2_1 = nn.Linear(256, self.num_classes//4, bias=False).cuda()
                init.normal_(self.classifier_2_1.weight, std=0.001)
                self.classifier_3_1 = nn.Linear(256, self.num_classes//4, bias=False).cuda()
                init.normal_(self.classifier_3_1.weight, std=0.001)
                
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = 8192
                feat_bn = nn.BatchNorm1d(self.num_features)
                feat_bn_1 = nn.BatchNorm1d(256)
            feat_bn.bias.requires_grad_(False)
            feat_bn_1.bias.requires_grad_(False)
            
        init.constant_(feat_bn.weight, 1)
        init.constant_(feat_bn.bias, 0)
        init.constant_(feat_bn_1.weight, 1)
        init.constant_(feat_bn_1.bias, 0)
        
        self.projector_feat_bn = nn.Sequential(
            projecter,
            feat_bn
        ).cuda()
        
        self.matrix_feat_bn = nn.Sequential(
            matrix,
            feat_bn_1
        ).cuda()
    
    
    def forward(self, x, feature_withbn=False):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        
        bn_x = self.projector_feat_bn(x)
        bn_x_1 = self.matrix_feat_bn(bn_x)

        # Split FC->
        prob = [None for _ in range(4)]
        prob[0] = self.classifier_0(bn_x.cuda())
        prob[1] = self.classifier_1(bn_x.cuda())
        prob[2] = self.classifier_2(bn_x.cuda())
        prob[3] = self.classifier_3(bn_x.cuda())
        
        prob = torch.cat(prob, dim = 1)
        # <-Split FC
        return x, prob

def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


