import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m,nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m,nn.Sigmoid):
            pass
        elif isinstance(m,nn.BatchNorm2d):
            pass
        elif isinstance(m,nn.MaxPool2d):
            pass
        elif isinstance(m,nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m,nn.AdaptiveMaxPool2d):
            pass
        elif isinstance(m,nn.ReLU6):
            pass
        elif isinstance(m,nn.Softmax):
            pass
        elif isinstance(m,nn.ModuleList):
            pass
        elif isinstance(m,nn.Flatten):
            pass        
        elif isinstance(m,nn.Tanh):
            pass
      
        else:
            m.initialize()
            
class Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        # downsample = nn.Sequential(
        #     nn.Conv2d(self.inplace,planes*4,kernel_size=1,)
        # )
        layers = [Block(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5


    def initialize(self):
        # print("load ResNet50")
        self.load_state_dict(torch.load('res/resnet50-19c8e357.pth'), strict=False)


def main():
    model = ResNet()
    summary(model,input_size=[(3,352,352)],batch_size=1,device="cpu")
    img = torch.ones(12,3,352,352)
    print(img.shape)
    out2, out3, out4, out5 = model(img)
    print(out2.shape,out3.shape,out4.shape,out5.shape)

if __name__ == "__main__":
    main()
