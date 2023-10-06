#!/usr/bin/python3
#coding=utf-8
import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class Test(object):
    def __init__(self, Dataset, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path,mode="test")
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=1)
        ## network
        self.net    = torch.load('model.pth') 
        self.net.train(False)
        self.net.cuda()     
        
    def save(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                out1, out2, out3, pred = self.net(image, shape)
                pred  = (torch.sigmoid(pred[0,0])*255).cpu().numpy()
                head = os.path.join("predict","FBENet",self.cfg.datapath.split('/')[-1])
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

if __name__=='__main__':
    for path in [os.path.join("data","ECSSD"), os.path.join("data","DUTS"), os.path.join("data","HKU-IS"), os.path.join("data","DUT-OMRON"), os.path.join("data","PASCAL-S")]:
        t = Test(dataset,path)
        t.save()
        print("Saved:",path)


