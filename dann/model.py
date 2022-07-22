import os
import argparse
import yaml

import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from diffusion_net.layers import DiffusionNet
from dpfm.utils import get_mask, nn_interpolate

from diffusion_net.layers import DiffusionNet
from dpfm.model import DPFMNet, CrossAttentionRefinementNet, RegularizedFMNet
from dpfm.utils import DPFMLoss, augment_batch
from project.datasets import ShrecPartialDataset, Tosca, shape_to_device
from .utils import ReverseLayerF

class FeatExtractorNet(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, cfg):
        super().__init__()
        self.robust = cfg["fmap"]["robust"]
        self.n_fmap = cfg["fmap"]["n_fmap"]

        # feature extractor
        self.feature_extractor = DiffusionNet(
            C_in=cfg["fmap"]["C_in"],
            C_out=cfg["fmap"]["n_feat"],
            C_width=128,
            N_block=4,
            dropout=True,
            with_gradient_features=True,
            with_gradient_rotations=True,
        )

        # cross attention refinement
        self.feat_refiner = CrossAttentionRefinementNet(n_in=cfg["fmap"]["n_feat"], num_head=cfg["attention"]["num_head"], gnn_dim=cfg["attention"]["gnn_dim"],
                                                        overlap_feat_dim=cfg["overlap"]["overlap_feat_dim"],
                                                        n_layers=cfg["attention"]["ref_n_layers"],
                                                        cross_sampling_ratio=cfg["attention"]["cross_sampling_ratio"],
                                                        attention_type=cfg["attention"]["attention_type"])

        
    def forward(self, batch):
        verts1, faces1, mass1, L1, evals1, evecs1, gradX1, gradY1 = (batch["shape1"]["xyz"], batch["shape1"]["faces"], batch["shape1"]["mass"],
                                                                     batch["shape1"]["L"], batch["shape1"]["evals"], batch["shape1"]["evecs"],
                                                                     batch["shape1"]["gradX"], batch["shape1"]["gradY"])
        verts2, faces2, mass2, L2, evals2, evecs2, gradX2, gradY2 = (batch["shape2"]["xyz"], batch["shape2"]["faces"], batch["shape2"]["mass"],
                                                                     batch["shape2"]["L"], batch["shape2"]["evals"], batch["shape2"]["evecs"],
                                                                     batch["shape2"]["gradX"], batch["shape2"]["gradY"])

        # set features to vertices
        features1, features2 = verts1, verts2

        feat1 = self.feature_extractor(features1, mass1, L=L1, evals=evals1, evecs=evecs1,
                                       gradX=gradX1, gradY=gradY1, faces=faces1).unsqueeze(0)
        feat2 = self.feature_extractor(features2, mass2, L=L2, evals=evals2, evecs=evecs2,
                                       gradX=gradX2, gradY=gradY2, faces=faces2).unsqueeze(0)
        
        # refine features
        ref_feat1, ref_feat2, overlap_score12, overlap_score21 = self.feat_refiner(verts1, verts2, feat1, feat2, batch)
        use_feat1, use_feat2 = (ref_feat1, ref_feat2) if self.robust else (feat1, feat2)
        #prepare output
        evecs_trans1, evecs_trans2 = evecs1.t()[:self.n_fmap] @ torch.diag(mass1), evecs2.t()[:self.n_fmap] @ torch.diag(mass2)
        evals1, evals2 = evals1[:self.n_fmap], evals2[:self.n_fmap]

        return overlap_score12, overlap_score21, use_feat1, use_feat2, evals1, evals2, evecs_trans1, evecs_trans2

    
    
class MapExtractorNet(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, lambda_=1e-3, resolvant_gamma=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        # compute linear operator matrix representation C1 and C2
        evecs_trans_x, evecs_trans_y = evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0)
        evals_x, evals_y = evals_x.unsqueeze(0), evals_y.unsqueeze(0)

        F_hat = torch.bmm(evecs_trans_x, feat_x)
        G_hat = torch.bmm(evecs_trans_y, feat_y)
        A, B = F_hat, G_hat

        D = get_mask(evals_x.flatten(), evals_y.flatten(), self.resolvant_gamma, feat_x.device).unsqueeze(0)

        A_t = A.transpose(1, 2)
        A_A_t = torch.bmm(A, A_t)
        B_A_t = torch.bmm(B, A_t)

        C_i = []
        for i in range(evals_x.size(1)):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + self.lambda_ * D_i), B_A_t[:, i, :].unsqueeze(1).transpose(1, 2))
            C_i.append(C.transpose(1, 2))
        C = torch.cat(C_i, dim=1)

        return C
    
    
class Discriminator(nn.Module):
    
    def __init__(self, n_in):
        super(Discriminator, self).__init__()

        self.n_in = n_in

        self.discriminator = nn.Sequential(
            nn.Linear(n_in, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2)
        )

    def forward(self, use_feat, alpha):
        
        reversed_input = ReverseLayerF.apply(use_feat, alpha)
        x = self.discriminator(reversed_input)
        
        return F.softmax(x,dim=1)

class DPFMNet_DA(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, cfg):
        super().__init__()

        # feature extractor
        self.feature_extractor = DiffusionNet(
            C_in=cfg["fmap"]["C_in"],
            C_out=cfg["fmap"]["n_feat"],
            C_width=128,
            N_block=4,
            dropout=True,
            with_gradient_features=True,
            with_gradient_rotations=True,
        )

        # cross attention refinement
        self.feat_refiner = CrossAttentionRefinementNet(n_in=cfg["fmap"]["n_feat"], num_head=cfg["attention"]["num_head"], gnn_dim=cfg["attention"]["gnn_dim"],
                                                        overlap_feat_dim=cfg["overlap"]["overlap_feat_dim"],
                                                        n_layers=cfg["attention"]["ref_n_layers"],
                                                        cross_sampling_ratio=cfg["attention"]["cross_sampling_ratio"],
                                                        attention_type=cfg["attention"]["attention_type"])
        
        self.discriminator = Discriminator(n_in=cfg["fmap"]["n_feat"])

        # regularized fmap
        self.fmreg_net = RegularizedFMNet(lambda_=cfg["fmap"]["lambda_"], resolvant_gamma=cfg["fmap"]["resolvant_gamma"])
        self.n_fmap = cfg["fmap"]["n_fmap"]
        self.robust = cfg["fmap"]["robust"]

    def forward(self, batch, alpha):
        verts1, faces1, mass1, L1, evals1, evecs1, gradX1, gradY1 = (batch["shape1"]["xyz"], batch["shape1"]["faces"], batch["shape1"]["mass"],
                                                                     batch["shape1"]["L"], batch["shape1"]["evals"], batch["shape1"]["evecs"],
                                                                     batch["shape1"]["gradX"], batch["shape1"]["gradY"])
        verts2, faces2, mass2, L2, evals2, evecs2, gradX2, gradY2 = (batch["shape2"]["xyz"], batch["shape2"]["faces"], batch["shape2"]["mass"],
                                                                     batch["shape2"]["L"], batch["shape2"]["evals"], batch["shape2"]["evecs"],
                                                                     batch["shape2"]["gradX"], batch["shape2"]["gradY"])

        # set features to vertices
        features1, features2 = verts1, verts2

        feat1 = self.feature_extractor(features1, mass1, L=L1, evals=evals1, evecs=evecs1,
                                       gradX=gradX1, gradY=gradY1, faces=faces1).unsqueeze(0)
        feat2 = self.feature_extractor(features2, mass2, L=L2, evals=evals2, evecs=evecs2,
                                       gradX=gradX2, gradY=gradY2, faces=faces2).unsqueeze(0)
        
        del gradX1, gradX2, gradY1, gradY2
        torch.cuda.empty_cache()
        # refine features
        ref_feat1, ref_feat2, overlap_score12, overlap_score21 = self.feat_refiner(verts1, verts2, feat1, feat2, batch)
        use_feat1, use_feat2 = (ref_feat1, ref_feat2) if self.robust else (feat1, feat2)
        torch.cuda.empty_cache()
        del feat1, feat2
        # predict fmap
        evecs_trans1, evecs_trans2 = evecs1.t()[:self.n_fmap] @ torch.diag(mass1), evecs2.t()[:self.n_fmap] @ torch.diag(mass2)
        evals1, evals2 = evals1[:self.n_fmap], evals2[:self.n_fmap]
        C_pred = self.fmreg_net(use_feat1, use_feat2, evals1, evals2, evecs_trans1, evecs_trans2)
        torch.cuda.empty_cache()
        use_feat = torch.cat((use_feat1,use_feat2), dim=1)
        domain_output = self.discriminator(use_feat, alpha)

        return C_pred, domain_output, overlap_score12, overlap_score21, use_feat1, use_feat2
