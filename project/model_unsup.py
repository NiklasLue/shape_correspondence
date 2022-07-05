#from copy import deepcopy
import torch
import torch.nn as nn

from diffusion_net.layers import DiffusionNet

from dpfm.model import RegularizedFMNet

class DPFMNet_unsup(nn.Module):
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

        # regularized fmap
        self.fmreg_net = RegularizedFMNet(lambda_=cfg["fmap"]["lambda_"], resolvant_gamma=cfg["fmap"]["resolvant_gamma"])
        self.n_fmap = cfg["fmap"]["n_fmap"]
        # self.robust = cfg["fmap"]["robust"]

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

        # predict fmap
        evecs_trans1, evecs_trans2 = evecs1.t()[:self.n_fmap] @ torch.diag(mass1), evecs2.t()[:self.n_fmap] @ torch.diag(mass2)
        evals1, evals2 = evals1[:self.n_fmap], evals2[:self.n_fmap]
        C1_pred = self.fmreg_net(feat1, feat2, evals1, evals2, evecs_trans1, evecs_trans2)
        C2_pred = self.fmreg_net(feat2, feat1, evals2, evals1, evecs_trans2, evecs_trans1)

        return C1_pred, C2_pred, feat1, feat2, evals1.unsqueeze(0), evals2.unsqueeze(0)
