import torch
import torch.nn as nn
import torch.nn.functional as F
def iou(pred,mask):
    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)).sum(dim=(2,3))
    union = ((pred+mask)).sum(dim=(2,3))
    iou = 1-(inter+1)/(union-inter+1)
    return iou.mean()

def dice(pred,mask):
    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)).sum(dim=(2,3))
    union = ((pred+mask)).sum(dim=(2,3))
    dice = 1-(2.*inter+1)/(union+1)
    return dice.mean()

def foreLoss(pred,mask):
    weight = torch.zeros_like(mask)
    weight = torch.fill_(weight,0.25)
    weight[mask>0]=1
    wbce = F.binary_cross_entropy_with_logits(pred,mask,weight=weight)
    return wbce+iou(pred,mask)

def backLoss(pred,mask):
    weight = torch.zeros_like(mask)
    weight = torch.fill_(weight,1)
    weight[mask>0]=0.25
    wbce = F.binary_cross_entropy_with_logits(pred,mask,weight=weight)
    return wbce+iou(pred,mask)

def edgeLoss(pred,edge):
    weight = torch.zeros_like(edge)
    weight = torch.fill_(weight,0.25)
    weight[edge>0]=1
    wbce = F.binary_cross_entropy_with_logits(pred,edge,weight=weight)
    return wbce + iou(pred,edge)


