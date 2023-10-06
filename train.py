# config=utf-8
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset
import os
from apex import amp
from net.B3Net_3branch import B3Net
from loss import foreLoss,backLoss,edgeLoss
from torchsummaryX import summary

def structure_loss(pred,mask):
    bce = F.binary_cross_entropy_with_logits(pred,mask,reduce='none')
    pred = torch.sigmoid(pred)
    inter = ((pred*mask)).sum(dim=(2,3))
    union = ((pred+mask)).sum(dim=(2,3))
    iou = 1-(inter+1)/(union-inter+1)
    return (bce+iou).mean()

def train(Dataset=dataset, Network=B3Net):
    ## train config
        cfg  = Dataset.Config(datapath=os.path.join("data","DUTS"), savepath=os.path.join("out"), mode="train", batch=24, lr=0.01, momen=0.9, decay=5e-4, epoch=64)
        data   = Dataset.Data(cfg)
        loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=2)
    ## network
        net    = Network(cfg)
        img = torch.rand((1,3,352,352))
        summary(net,img)
    # test set
        test_cfg    = Dataset.Config(datapath=os.path.join("data","DUT-OMRON"),mode="test")
        test_data   = Dataset.Data(test_cfg)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)    
    ## parameter
        base, head = [], []
        for name, param in net.named_parameters():
            if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
                print(name)
            elif 'bkbone' in name:
                base.append(param)
            else:
                head.append(param)
        
        net.cuda()
        optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
        net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
        global_step    = 0

        for epoch in range(cfg.epoch):
            net.train(True)
            optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
            optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

            for step, (image, mask, edge) in enumerate(loader):
                image, mask,edge = image.cuda().float(), mask.cuda().float(), edge.cuda().float()
                out1,out2,out3,out = net(image)

                loss1 = foreLoss(out1,mask)
                loss2 = backLoss(out2,mask)
                loss3 = edgeLoss(out3,edge)
                loss4 = structure_loss(out,mask)
                
                loss = loss4 + loss1 + loss2 + loss3

                optimizer.zero_grad()
                with amp.scale_loss(loss, optimizer) as scale_loss:
                    scale_loss.backward()
                optimizer.step()
                global_step += 1
                if step%50 == 0:
                    print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))
            if (epoch+1)>(cfg.epoch-10):
                net.train(False)
                with torch.no_grad():
                    mae=0
                    eval_step=0
                    for image, mask, shape, name in test_loader:
                        image = image.cuda().float()
                        out1, out2, out3, pred = net(image,shape)
                        out   = pred
                        pred  = np.round((torch.sigmoid(out[0,0])*255.0).cpu().numpy())
                        mask  = mask[0].numpy()*255.0
                        assert pred.shape == mask.shape
                        mae += np.mean(np.abs(pred - mask)/255.0)
                        eval_step+=1
                    mae/=eval_step
                    if mae<=0.0485:
                        savepath = cfg.savepath+"/b3net/model-"+str(datetime.datetime.now())+'_'+str(mae)+'_'+str(epoch+1)+".pth"
                        torch.save(net.state_dict(),savepath)
                        print("saved in"+savepath)

if __name__=='__main__':
    train()
