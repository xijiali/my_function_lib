from __future__ import absolute_import
import torchvision
from torch import nn
from torch.nn import functional as F
import torch

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        self.num_classes=num_classes
        self.feat_dim = 2048 # The feature dim affects the performance

        # pretrained resnet50 conv4-6
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        # fc_conv5-3
        self.fc_high = nn.Linear(self.feat_dim, num_classes,bias=False)
        self.fc_high.apply(weights_init_classifier)


    def forward(self, x):
        # send into resnet50
        f0 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        # GAP
        high_f=F.avg_pool2d(f4,[f4.size(2),f4.size(3)]).view(f4.size(0),f4.size(1))

        # fc_high
        fc_high=self.fc_high(high_f)

        if not self.training:
            return high_f

        return fc_high,(f0,f1,f2,f3,f4)

class PrivateResNet50(nn.Module):
    def __init__(self,num_classes=None):
        super(PrivateResNet50, self).__init__()

        self.feat_dim = 2048 # The feature dim affects the performance

        # pretrained resnet50 conv4-6
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.conv1=resnet50.conv1
        self.bn1=resnet50.bn1
        self.relu=resnet50.relu
        self.maxpool=resnet50.maxpool
        self.layer1=resnet50.layer1
        self.layer2=resnet50.layer2
        self.layer3=resnet50.layer3
        self.layer4=resnet50.layer4

    def forward(self, x):
        # send into resnet50
        f0=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f1=self.layer1(f0)
        f2=self.layer2(f1)
        f3=self.layer3(f2)
        f4=self.layer4(f3)
        if not self.training:
            f4=F.avg_pool2d(f4,[f4.size(2),f4.size(3)]).view(f4.size(0),f4.size(1))
            return f4

        return f0,f1,f2,f3,f4


__factory = {
    'baseline': ResNet50,
    'private':PrivateResNet50,
}

def get_names():
    return __factory.keys()

def init_network(name, num_classes=None):
    if name not in __factory.keys():
        raise KeyError("Unknown network: {}".format(name))
    else:
        return __factory[name](num_classes)






