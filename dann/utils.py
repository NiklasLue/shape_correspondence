import numpy as np
from torch.autograd import Function
import torch
import itertools
import os
import torch.nn as nn



class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

    
class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum((a - b) ** 2, axis=(1, 2))
        return torch.mean(loss)


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.binary_loss = nn.BCELoss(reduction="none")

    def forward(self, prediction, gt):
        class_loss = self.binary_loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative

        return torch.mean(weights * class_loss)


class NCESoftmaxLoss(nn.Module):
    def __init__(self, nce_t, nce_num_pairs):
        super().__init__()
        self.nce_t = nce_t
        self.nce_num_pairs = nce_num_pairs
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, features_1, features_2, map21):
        features_1, features_2 = features_1.squeeze(0), features_2.squeeze(0)

        if map21.shape[0] > self.nce_num_pairs:
            selected = np.random.choice(map21.shape[0], self.nce_num_pairs, replace=False)
        else:
            selected = torch.arange(map21.shape[0])

        query = features_1[map21[selected]]
        keys = features_2[selected]

        logits = - torch.cdist(query, keys)
        logits = torch.div(logits, self.nce_t)
        labels = torch.arange(selected.shape[0]).long().to(features_1.device)
        loss = self.cross_entropy(logits, labels)
        return loss


class ExtLoss(nn.Module):
    def __init__(self, w_fmap=1, w_acc=1, w_nce=0.1, nce_t=0.07, nce_num_pairs=4096):
        super().__init__()

        self.w_acc = w_acc
        self.w_nce = w_nce

        self.binary_loss = WeightedBCELoss()
        self.nce_softmax_loss = NCESoftmaxLoss(nce_t, nce_num_pairs)

    def forward(self, map21, feat1, feat2, overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21):
        loss = 0

        # overlap loss
        acc_loss = self.binary_loss(overlap_score12, gt_partiality_mask12.float()) * self.w_acc
        acc_loss += self.binary_loss(overlap_score21, gt_partiality_mask21.float()) * self.w_acc
        loss += acc_loss

        # nce loss
        nce_loss = self.nce_softmax_loss(feat1, feat2, map21) * self.w_nce
        loss += nce_loss

        return loss, fmap_loss, acc_loss, nce_loss


class DPFMLoss_da(nn.Module):
    def __init__(self, w_fmap=1, w_acc=1, w_nce=0.1, nce_t=0.07, nce_num_pairs=4096):
        super().__init__()

        self.w_fmap = w_fmap
        self.w_acc = w_acc
        self.w_nce = w_nce
        
        self.frob_loss = FrobeniusLoss()
        self.binary_loss = WeightedBCELoss()
        self.nce_softmax_loss = NCESoftmaxLoss(nce_t, nce_num_pairs)
        self.discriminator_loss = nn.CrossEntropyLoss()

    def forward(self, C12, C_gt, map21, feat1, feat2, overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21, domain_output=0, label=0):
        loss = 0

        # fmap loss
        fmap_loss = self.frob_loss(C12, C_gt) * self.w_fmap
        loss += fmap_loss

        # overlap loss
        acc_loss = self.binary_loss(overlap_score12, gt_partiality_mask12.float()) * self.w_acc
        acc_loss += self.binary_loss(overlap_score21, gt_partiality_mask21.float()) * self.w_acc
        loss += acc_loss

        # nce loss
        nce_loss = self.nce_softmax_loss(feat1, feat2, map21) * self.w_nce
        loss += nce_loss
        
        # discriminator loss
        discriminator_loss = self.discriminator_loss(domain_output, label)
        #loss += discriminator_loss
        
        return loss, fmap_loss, acc_loss, nce_loss, discriminator_loss

    

def save_model(model, save_name):
    print('Save models ...')

    save_folder = 'trained_models'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    torch.save(model.state_dict(), 'models/da_models/source' + '_' + str(save_name) + '.pt')

    print('Model is saved !!!')
    
def set_model_mode(mode='train', models=None):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()