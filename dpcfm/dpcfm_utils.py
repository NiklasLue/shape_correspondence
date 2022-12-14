
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

    def forward(self, C12, C21, C_gt, C_gt2, map21, feat1, feat2, evals1, evals2, overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21):
        I = torch.empty(C12.size()).to(C12.device)
        for i in range(I.size(dim=0)):
            J = torch.eye(I.size(dim=1))
            r = get_rank(evals1[i,:], evals2[i,:])
            for k in range(r, I.size(dim=1)):
                J[k,k] = 0
            I[i] = J
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
        coup_loss = self.frob_loss(torch.bmm(C12, C21), I) * self.w_coup
        loss += coup_loss
        

        return loss, fmap_loss, acc_loss, nce_loss, coup_loss



class DPCFMLossV2(DPFMLoss):
    def __init__(self, w_fmap=1, w_acc=1, w_nce=0.1, w_coup = 0.1, nce_t=0.07, nce_num_pairs=4096):
        super().__init__()

        self.w_fmap = w_fmap
        self.w_acc = w_acc
        self.w_nce = w_nce
        self.w_coup = w_coup

        self.frob_loss = FrobeniusLoss()
        self.binary_loss = WeightedBCELoss()
        self.nce_softmax_loss = NCESoftmaxLoss(nce_t, nce_num_pairs)

    def forward(self, C12, C21, C_gt, C_gt2, map21, feat1, feat2, evals1, evals2, overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21):

        I = torch.empty(C12.size()).to(C12.device)
        for i in range(I.size(dim=0)):
            J = torch.eye(I.size(dim=1))
            r = get_rank(evals1[i,:], evals2[i,:])
            for k in range(r, I.size(dim=1)):
                J[k,k] = 0
            I[i] = J


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
        #can either comment this part out due to the coupling constarit for computing C2
        coup_loss = self.frob_loss(torch.bmm(C12, C21), I) * self.w_coup
        loss += coup_loss
        

        return loss, fmap_loss, acc_loss, nce_loss, coup_loss






def get_rank(evals1, evals2):
    est_rank = 0
    m = max(evals1)
    for i in range(len(evals2)):
        if evals2[i] < m:
            est_rank += 1
    return est_rank

