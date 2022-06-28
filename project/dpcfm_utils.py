
import random
import numpy as np
import torch
import torch.nn as nn

from dpfm.utils import DPFMLoss, FrobeniusLoss, WeightedBCELoss, NCESoftmaxLoss


class DPCFMLoss(DPFMLoss):
    def __init__(self, w_fmap=1, w_acc=1, w_nce=0.1, w_coup = 0.1, nce_t=0.07, nce_num_pairs=4096):
        super().__init__()

        self.w_fmap = w_fmap
        self.w_acc = w_acc
        self.w_nce = w_nce
        self.w_coup = w_coup

        self.frob_loss = FrobeniusLoss()
        self.binary_loss = WeightedBCELoss()
        self.nce_softmax_loss = NCESoftmaxLoss(nce_t, nce_num_pairs)

    def forward(self, C12, C21, C_gt, C_gt2, map21, feat1, feat2, overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21):
        loss = 0

        # fmap loss
        fmap_loss = self.frob_loss(C12, C_gt) * self.w_fmap
        fmap_loss2 = self.frob_loss(C21, C_gt2) *self.w_fmap
        loss += fmap_loss
        loss += fmap_loss2

        # overlap loss
        acc_loss = self.binary_loss(overlap_score12, gt_partiality_mask12.float()) * self.w_acc
        acc_loss += self.binary_loss(overlap_score21, gt_partiality_mask21.float()) * self.w_acc
        loss += acc_loss

        # nce loss
        nce_loss = self.nce_softmax_loss(feat1, feat2, map21) * self.w_nce
        loss += nce_loss

        #coupling loss
        #TODO cut down the rank of identity matrix
        I = C12 @ C21
        coup_loss = self.frob_loss(I, torch.eye(I.shape[0])) * self.w_coup
        loss += coup_loss
        

        return loss, fmap_loss, acc_loss, nce_loss
