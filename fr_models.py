import torch
import os
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from fr_metrics import  CosMargin, ArcMarginProduct, AddMarginProduct
from mi_neural_estimators import CLUB, CLUBSample
from iresnet_plus_spu import iresnet50_spu

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)





class fr_model_1(nn.Module):
    def __init__(self, n_cls, args):
        super().__init__()
        face_id_loss = args.face_id_loss
        self.backbone_id = iresnet50_spu(dropout=0, num_features=args.feature_dim, spu_scale=args.spu_scale, fp16=args.fp16)
        if face_id_loss in 'cosface':
            self.margin_fc = AddMarginProduct(n_cls, args.feature_dim, args.scale, args.margin)  
        elif face_id_loss in 'arcface':
            self.margin_fc = ArcMarginProduct(n_cls, args.feature_dim, args.scale, args.margin)

        # initialize
        if not args.pretrained:
            self.backbone_id.apply(init_weights)
        self.margin_fc.apply(init_weights)


    def forward(self, xs, ys=None, emb=False):
        # 512-D embedding
        embs_id = self.backbone_id(xs)       
        if emb:
            return F.normalize(embs_id)
        id_logits = self.margin_fc(embs_id, ys)
        return id_logits