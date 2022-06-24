import yaml

import torch
import numpy as np

import pyFM.signatures as sg
from pyFM.functional import FunctionalMapping
from dpfm.model import DPFMNet

class FunctionalMappingAdv(FunctionalMapping):
    """
    A class to compute functional maps between two meshes, added usage of DPFM descriptors

    Attributes
    ----------------------
    mesh1  : TriMesh - first mesh
    mesh2  : TriMesh - second mesh

    descr1 : (n1,p) descriptors on the first mesh
    descr2 : (n2,p) descriptors on the second mesh
    D_a    : (k1,k1) area-based shape differnence operator
    D_c    : (k1,k1) conformal-based shape differnence operator

    Properties
    ----------------------
    FM_type : 'classic' | 'icp' | 'zoomout' which FM is currently used
    k1      : dimension of the first eigenspace (varies depending on the type of FM)
    k2      : dimension of the seconde eigenspace (varies depending on the type of FM)
    FM      : (k2,k1) current FM
    p2p     : (n2,) point to point map associated to the current functional map
    """
    def __init__(self, mesh1, mesh2, shape1=None, shape2=None):

        super(FunctionalMappingAdv, self).__init__(mesh1, mesh2)
        
        # SHAPES FROM DATALOADERS (only needed for descr_type='DPFM')
        self.shape1 = shape1
        self.shape2 = shape2


    def preprocess(self, n_ev=(50,50), n_descr=100, descr_type='WKS', landmarks=None, subsample_step=1, k_process=None, verbose=False, **kwargs):
        """
        Saves the information about the Laplacian mesh for opt

        Parameters
        -----------------------------
        n_ev           : (k1, k2) tuple - with the number of Laplacian eigenvalues to consider.
        n_descr        : int - number of descriptors to consider
        descr_type     : str - "HKS" | "WKS"
        landmarks      : (p,1|2) array of indices of landmarks to match.
                         If (p,1) uses the same indices for both.
        subsample_step : int - step with which to subsample the descriptors.
        """
        self.k1, self.k2 = n_ev

        if k_process is None:
            k_process = 200

        use_lm = landmarks is not None and len(landmarks) > 0

        # Compute the Laplacian spectrum
        if verbose:
            print('\nComputing Laplacian spectrum')
        self.mesh1.process(max(self.k1, k_process), verbose=verbose)
        self.mesh2.process(max(self.k2, k_process), verbose=verbose)

        if verbose:
            print('\nComputing descriptors')

        # Extract landmarks indices
        if use_lm:
            lmks1, lmks2 = self._get_lmks(landmarks, verbose=False)

        # Compute descriptors
        if descr_type == 'HKS':
            self.descr1 = sg.mesh_HKS(self.mesh1, n_descr, k=self.k1)  # (N1, n_descr)
            self.descr2 = sg.mesh_HKS(self.mesh2, n_descr, k=self.k2)  # (N2, n_descr)

            if use_lm:
                lm_descr1 = sg.mesh_HKS(self.mesh1, n_descr,landmarks=lmks1, k=self.k1)  # (N1, p*n_descr)
                lm_descr2 = sg.mesh_HKS(self.mesh2, n_descr, landmarks=lmks2, k=self.k2)  # (N2, p*n_descr)

                self.descr1 = np.hstack([self.descr1, lm_descr1])  # (N1, (p+1)*n_descr)
                self.descr2 = np.hstack([self.descr2, lm_descr2])  # (N2, (p+1)*n_descr)

        elif descr_type == 'WKS':
            self.descr1 = sg.mesh_WKS(self.mesh1, n_descr, k=self.k1)  # (N1, n_descr)
            self.descr2 = sg.mesh_WKS(self.mesh2, n_descr, k=self.k2)  # (N2, n_descr)

            if use_lm:
                lm_descr1 = sg.mesh_WKS(self.mesh1, n_descr, landmarks=lmks1, k=self.k1)  # (N1, p*n_descr)
                lm_descr2 = sg.mesh_WKS(self.mesh2, n_descr, landmarks=lmks2, k=self.k2)  # (N2, p*n_descr)

                self.descr1 = np.hstack([self.descr1, lm_descr1])  # (N1, (p+1)*n_descr)
                self.descr2 = np.hstack([self.descr2, lm_descr2])  # (N2, (p+1)*n_descr)

        elif descr_type == "DPFM":
            if torch.cuda.is_available() and cfg["misc"]["cuda"]:
                device = torch.device(f'cuda:{cfg["misc"]["device"]}')
            else:
                device = torch.device("cpu")
            cfg = yaml.safe_load(open(f"dpfm/config/{kwargs['config']}.yaml", "r"))
            dpfm_net = DPFMNet(cfg).to(device)
            dpfm_net.load_state_dict(torch.load(kwargs["model_path"], map_location=device))
            dpfm_net.eval()

            batch = {'shape1': self.shape1, 'shape2': self.shape2}
            C, _, _, self.descr1, self.descr2 = dpfm_net(batch)

            self.descr1 = self.descr1.detach().cpu().numpy().squeeze(0)
            self.descr2 = self.descr2.detach().cpu().numpy().squeeze(0)

            self.k1, self.k2 = C.squeeze(0).shape

        else:
            raise ValueError(f'Descriptor type "{descr_type}" not implemented')

        # Subsample descriptors
        self.descr1 = self.descr1[:, np.arange(0, self.descr1.shape[1], subsample_step)]
        self.descr2 = self.descr2[:, np.arange(0, self.descr2.shape[1], subsample_step)]

        # Normalize descriptors
        if verbose:
            print('\tNormalizing descriptors')

        no1 = np.sqrt(self.mesh1.l2_sqnorm(self.descr1))  # (p,)
        no2 = np.sqrt(self.mesh2.l2_sqnorm(self.descr2))  # (p,)

        self.descr1 /= no1[None, :]
        self.descr2 /= no2[None, :]

        if verbose:
            n_lmks = np.asarray(landmarks).shape[0] if use_lm else 0
            print(f'\n\t{self.descr1.shape[1]} out of {n_descr*(1+n_lmks)} possible descriptors kept')

        return self
