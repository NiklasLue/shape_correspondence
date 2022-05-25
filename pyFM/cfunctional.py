import copy
import time

from tqdm import tqdm

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import fmin_cg

import pyFM.signatures as sg
import pyFM.optimize as opt_func
import pyFM.refine
import pyFM.spectral as spectral


class CoupledFunctionalMapping:
    """
    A class to compute coupled functional maps between two meshes

    Attributes
    ----------------------
    mesh1  : TriMesh - first mesh
    mesh2  : TriMesh - second mesh

    descr1 : (n1,p) descriptors on the first mesh
    descr2 : (n2,p) descriptors on the second mesh

    Properties
    ----------------------
    C_type : 'classic' | 'icp' | 'zoomout' which C is currently used
    k1      : dimension of the first eigenspace (varies depending on the type of C)
    k2      : dimension of the seconde eigenspace (varies depending on the type of C)
    C1, C2  : (k2,k1) (k1,k2) current coupled functional maps
    p2p     : (n2,) point to point map associated to the current functional map
    """
    def __init__(self, mesh1, mesh2):

        self.mesh1 = copy.deepcopy(mesh1)
        self.mesh2 = copy.deepcopy(mesh2)

        # DESCRIPTORS
        self.descr1 = None
        self.descr2 = None

        # FUNCTIONAL MAP
        #self._C_type = 'classic'
        #self._C_base = None
        #self._C_icp = None
        #self._C_zo = None
        
        # COUPLED FUNCTIONAL MAP
        self._C1 = None
        self._C2 = None

        # AREA AND CONFORMAL SHAPE DIFFERENCE OPERATORS
        #self.SD_a = None
        #self.SD_c = None

        self._k1, self._k2,  = None, None

    # DIMENSION PROPERTIES
    @property
    def k1(self):
        """"
        Return the input dimension of the functional map
        """
        if self._k1 is None and not self.preprocessed and not self.fitted:
            raise ValueError('No information known about dimensions')
        if self.fitted:
            return self.C1.shape[1]
        else:
            return self._k1
    def k1(self):
        """"
        Return the input dimension of the functional map
        """
        if self._k1 is None and not self.preprocessed and not self.fitted:
            raise ValueError('No information known about dimensions')
        if self.fitted:
            return self.C2.shape[0]
        else:
            return self._k1    
    
    def k1(self, k1):
        self._k1 = k1
        
    @property
    def k2(self):
        if self._k2 is None and not self.preprocessed and not self.fitted:
            raise ValueError('No information known about dimensions')
        if self.fitted:
            return self.C1.shape[0]
        else:
            return self._k2
    
    def k2(self):
        if self._k2 is None and not self.preprocessed and not self.fitted:
            raise ValueError('No information known about dimensions')
        if self.fitted:
            return self.C2.shape[1]
        else:
            return self._k2

    def k2(self, k2):
        self._k2 = k2

    # FUNCTIONAL MAP SWITCHER (REFINED OR NOT)
    
    @property
    def C1(self):
        if True:
            return self._C1
    def C2(self):
        if True:
            return self._C2
    
    def C1(self, C1):
        self._C1 = C1
    def C2(self,C2):
        self._C2 = C2

    # BOOLEAN PROPERTIES
    @property
    def preprocessed(self):
        """
        check if enough information is provided to fit the model
        """
        test_descr = (self.descr1 is not None) and (self.descr2 is not None)
        test_evals = (self.mesh1.eigenvalues is not None) and (self.mesh2.eigenvalues is not None)
        test_evects = (self.mesh1.eigenvectors is not None) and (self.mesh2.eigenvectors is not None)
        return test_descr and test_evals and test_evects

    @property
    def fitted(self):
        return self.C1 is not None

    def get_p2p(self, use_adj=False, n_jobs=1):
        
        coupled_p2p = spectral.mesh_FM_to_p2p(self.C1, self.mesh1, self.mesh2,
                                              use_adj=use_adj, n_jobs=n_jobs)
        return coupled_p2p
        

    def get_precise_map(self, precompute_dmin=True, use_adj=True, batch_size=None, n_jobs=1, verbose=False):
        """
        Returns a precise map from mesh2 to mesh1

        Paramaters
        -------------------
        precompute_dmin : Whether to precompute all the values of delta_min.
                          Faster but heavier in memory
        use_adj         : use the adjoint method
        batch_size      : If precompute_dmin is False, projects batches of points on the surface
        n_jobs          : number of parallel process for nearest neighbor precomputation

        Output
        -------------------
        CoP21 : (n2,n1) sparse - precise map from mesh2 to mesh1
        """
    
        if not self.fitted:
            raise ValueError("Model should be fit and fit to obtain p2p map")
        
        CoP21= spectral.mesh_FM_to_p2p_precise(self.C1, self.mesh, self.mesh2,
                                               precompute_dmin=precompute_dmin, use_adj=use_adj, batch_size=batch_size,
                                               n_jobs=n_jobs, verbose=verbose)
        return CoP21
    
    def _get_lmks(self, landmarks, verbose=False):
        if np.asarray(landmarks).squeeze().ndim == 1:
            if verbose:
                print('\tUsing same landmarks indices for both meshes')
            lmks1 = np.asarray(landmarks).squeeze()
            lmks2 = lmks1.copy()
        else:
            lmks1, lmks2 = landmarks[:, 0], landmarks[:, 1]

        return lmks1, lmks2

    def preprocess(self, n_ev=(50,50), n_descr=100, descr_type='WKS', landmarks=None, subsample_step=1, k_process=None,
                           verbose=False):
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
        self.k2, self.k1 = n_ev

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
    
    def fit(self, mu_cons, mu_LB, optinit='zeros', verbose=False):
        """
        Solves the functional map optimization problem :

        min_C1,C2 ||C1@A - B|| + ||A - C2 @ B|| +  mu_cons * ||C1 @ C2 - I||^2)
              + mu_reg * (sum_i ||Ci * W||^2)

        with A and B descriptors, I identity matrix (k2,k2), W weight matrix see D. Eynard, E. Rodola, K. Glashoff, and M. M. Bronstein, 
        Coupled functional maps, in 2016 Fourth International Conference on 3D Vision (3DV), IEEE, pp. 399–407.

        Parameters
        -------------------------------
        w_reg            : weight matrix
      
        optinit          : 'zeros' (| 'random' | 'identity' |) initialization.
                           In any case, the first column of the functional map is computed by hand
                           and not modified during optimization
        """
        if optinit not in ['random', 'identity', 'zeros']:
            raise ValueError(f"optinit arg should be 'random', 'identity' or 'zeros', not {optinit}")

        if not self.preprocessed:
            self.coupled_preprocess()

        # Project the descriptors on the LB basis
        descr1_red = self.project(self.descr1, mesh_ind=1)  # (n_ev1, n_descr)
        descr2_red = self.project(self.descr2, mesh_ind=2)  # (n_ev2, n_descr)
        
        # Squared difference of eigenvalues
        ev_sqdiff = np.square(self.mesh1.eigenvalues[None, :self.k1] - self.mesh2.eigenvalues[:self.k2, None])
        ev_sqdiff /= ev_sqdiff.sum()
        
        # Compute weight matrix 
        W = self.weight_matrix()

        # Arguments for the optimization problem
        args = (descr1_red, descr2_red, ev_sqdiff, mu_cons, mu_LB) # to add weight matrix W

        # Initialization
        C1, C2 = np.zeros((self.k2, self.k1)), np.zeros((self.k1, self.k2))
                
        # Optimization
        # To be define l_bfgs_b
        res = fmin_l_bfgs_b(opt_func.loss, np.concatenate((C1.ravel(), C2.ravel())), fprime=opt_func.loss_grad, args=args)
        res = fmin_cg(opt_func.loss, np.concatenate((C1.ravel(), C2.ravel())), fprime=opt_func.loss_grad, args=args)
        
        #A = res[0][:self.k1*self.k2]
        sol = res
        C1sol, C2sol = sol[0:len(sol)//2], sol[len(sol)//2 : len(sol)]
        self.C1, self.C2 = np.reshape(C1sol, (self.k2,self.k1)), np.reshape(C2sol, (self.k1,self.k2))

    def weight_matrix(self):
        est_rank = 0
        for i in range(len(self.mesh2.eigenvalues)):
                if self.mesh2.eigenvalues[i] - max(self.mesh1.eigenvalues) < 0:
                    est_rank += 1
        W = np.zeros((self.k1,self.k1));
        
        for i in range(self.k1):
            for j in range(self.k1):
                slope = est_rank/self.k1
                
                direction = np.array([1, 1])
                
                direction = direction / np.linalg.norm(direction)
                direction = np.append(direction,[0])
                
                W[i,j] = np.exp(-0.03*np.sqrt(i**2 + j**2))*np.linalg.norm(np.cross(direction, np.array([i, j, 0])-np.array([1, 1, 0])))
        return W
        
    def project(self, func, k=None, mesh_ind=1):
        """
        Projects a function on the LB basis

        Parameters
        -----------------------
        func    : array - (n1|n2,p) evaluation of the function
        mesh_in : int  1 | 2 index of the mesh on which to encode

        Output
        -----------------------
        encoded_func : (n1|n2,p) array of decoded f
        """
        if k is None:
            k = self.k1 if mesh_ind == 1 else self.k2

        if mesh_ind == 1:
            return self.mesh1.project(func, k=k)
        elif mesh_ind == 2:
            return self.mesh2.project(func, k=k)
        else:
            raise ValueError(f'Only indices 1 or 2 are accepted, not {mesh_ind}')
            
"""
       
    def get_x0(self, optinit="zeros"):
        
        Returns the initial functional map for optimization. not used but could be

        Parameters
        ------------------------
        optinit : 'random' | 'identity' | 'zeros' initialization.
                  In any case, the first column of the functional map is computed by hand
                  and not modified during optimization

        Output
        ------------------------
        x0 : corresponding initial vector
        
        if optinit == 'random':
            x0 = np.random.random((self.k2, self.k1))
        elif optinit == 'identity':
            x0 = np.eye(self.k2, self.k1)
        else:
            x0 = np.zeros((self.k2, self.k1))

        # Sets the equivalence between the constant functions
        ev_sign = np.sign(self.mesh1.eigenvectors[0,0]*self.mesh2.eigenvectors[0,0])
        area_ratio = np.sqrt(self.mesh2.area/self.mesh1.area)

        x0[:,0] = np.zeros(self.k2)
        x0[0,0] = ev_sign * area_ratio

        return x0

    

    def decode(self, encoded_func, mesh_ind=2):
        
        Decode a function from the LB basis

        Parameters
        -----------------------
        encoded_func : array - (k1|k2,p) encoding of the functions
        mesh_ind     : int  1 | 2 index of the mesh on which to decode

        Output
        -----------------------
        func : (n1|n2,p) array of decoded f
        

        if mesh_ind == 1:
            return self.mesh1.decode(encoded_func)
        elif mesh_ind == 2:
            return self.mesh2.decode(encoded_func)
        else:
            raise ValueError(f'Only indices 1 or 2 are accepted, not {mesh_ind}')

    def transport(self, encoded_func, reverse=False):
        
        transport a function from LB basis 1 to LB basis 2. 
        If reverse is True, then the functions are transposed the other way
        using the transpose of the functional map matrix

        Parameters
        -----------------------
        encoded_func : array - (k1|k2,p) encoding of the functions
        reverse      : bool If true, transpose from 2 to 1 using the transpose of the FM

        Output
        -----------------------
        transp_func : (n2|n1,p) array of new encoding of the functions
        
        if not self.preprocessed:
            raise ValueError("The Functional map must be fit before transporting a function")

        if not reverse:
            return self.FM @ encoded_func
        else:
            return self.FM.T @ encoded_func

    def transfer(self, func, reverse=False):
        
        Transfer a function from mesh1 to mesh2.
        If 'reverse' is set to true, then the transfer goes
        the other way using the transpose of the functional
        map as approximate inverser transfer.

        Parameters
        ----------------------
        func : (n1|n2,p) evaluation of the functons

        Output
        -----------------------
        transp_func : (n2|n1,p) transfered function

        
        if not reverse:
            return self.decode(self.transport(self.project(func)))

        else:
            encoding = self.project(func, mesh_ind=2)
            return self.decode(self.transport(encoding, reverse=True),
                               mesh_ind=1)
"""