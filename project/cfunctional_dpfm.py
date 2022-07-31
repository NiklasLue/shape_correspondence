import numpy as np
from scipy.optimize import minimize
import torch

from .base_functions import loss, loss_grad
import pyFM.spectral as spectral


class CoupledFunctionalMappingDPFM:
    """
    A class to compute coupled functional maps between two meshes

    Attributes
    ----------------------
    dpfm_net : initialized 

    Properties
    ----------------------
    k1      : dimension of the first eigenspace (varies depending on the type of C)
    k2      : dimension of the seconde eigenspace (varies depending on the type of C)
    C1, C2  : (k2,k1) (k1,k2) current coupled functional maps
    p2p     : (n2,) point to point map associated to the current functional map
    """
    def __init__(self, dpfm_net):
        # DPFM NET
        #TODO: decide if we want an initialized dpfm net class instance or just the config file
        self.dpfm_net = dpfm_net

        # SHAPES
        self.shape1 = None
        self.shape2 = None
    
        # DESCRIPTORS
        self.descr1 = None
        self.descr2 = None
        
        # COUPLED FUNCTIONAL MAP
        self._C1 = None
        self._C2 = None

        self._k1, self._k2,  = dpfm_net.n_fmap, dpfm_net.n_fmap

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
    
    @k1.setter
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

    @k2.setter
    def k2(self, k2):
        self._k2 = k2

    # FUNCTIONAL MAP SWITCHER (REFINED OR NOT)
    @property
    def C1(self):
        return self._C1
    
    @C1.setter
    def C1(self, C1):
        self._C1 = C1

    @property
    def C2(self):
        return self._C2

    @C2.setter
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
        return self.C1 is not None and self.C2 is not None

    def get_p2p_map(self, use_adj=False, n_jobs=1):
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
        
        p2p= spectral.FM_to_p2p(self.C1, self.shape1["evecs"].cpu(), self.shape2["evecs"].cpu(), use_adj=use_adj, n_jobs=n_jobs)

        return p2p
    
    def fit(self, shape1 , shape2, mu_pres = 1, mu_coup = 1e-1, mu_mask = 0, mu_des = 0, mu_orient = 0, orient_reversing=False, optinit='zeros', optmask = 'lbo', rank = True, subsample_step=1, verbose=False):
        """
        Solves the functional map optimization problem :

        min_C1,C2 ||C1@A - B|| + ||A - C2 @ B|| +  mu_coup * ||C1 @ C2 - I||^2)
              + mu_reg * (sum_i ||Ci * W||^2)

        with A and B descriptors, I identity matrix (k2,k2), W weight matrix see D. Eynard, E. Rodola, K. Glashoff, and M. M. Bronstein, 
        Coupled functional maps, in 2016 Fourth International Conference on 3D Vision (3DV), IEEE, pp. 399–407.

        Parameters
        -------------------------------
        shape1           : first shape, from dataloader
        shape2           : second shape, from dataloader
        mu_pres          : weight for descriptor preservation
        mu_coup          : weight for coupling
        mu_mask          : weight for mask
        mu_des           : weight for commutativity operators
        mu_orient        : weight for orientation agnostic term
        orient_reversing : 'False' (| 'True' |) option for orientation agnostic term
        subsample_step   : how to subsample descriptors (for quicker inference)
        w_reg            : weight matrix
        optinit          : 'zeros' (| 'nocoup' | 'identity' |) initialization.
        optmask          : 'lbo' (| 'resolvent' | 'slanted' |) mask.

        Output
        -------------------
        C1 : (k2,k1) functional map from shape1 to shape2
        C2 : (k1,k2) functional map from shape2 to shape1
        """
        if optinit not in ['nocoup', 'identity', 'zeros']:
            raise ValueError(f"optinit arg should be 'nocoup', 'identity' or 'zeros', not {optinit}")
            
        if optmask not in ['lbo', 'resolvent', 'slanted']:
            raise ValueError(f"optinit arg should be 'lbo', 'resolvent', 'slanted', not {optinit}")    

        # if not self.preprocessed:
        #     self.coupled_preprocess()

        self.shape1 = shape1
        self.shape2 = shape2

        # calculate descriptors
        self.dpfm_net.eval()
        batch = {'shape1': self.shape1, 'shape2': self.shape2}
        _, _, _, descr1, descr2 = self.dpfm_net(batch)

        self.A1, self.A2 = torch.diag(shape1["mass"]), torch.diag(shape2["mass"])

        self.descr1 = descr1.squeeze(0)
        self.descr2 = descr2.squeeze(0)

        # Subsample descriptors
        self.descr1 = self.descr1[:, np.arange(0, self.descr1.shape[1], subsample_step)]
        self.descr2 = self.descr2[:, np.arange(0, self.descr2.shape[1], subsample_step)]

        # Project the descriptors on the LB basis
        descr1_red = self.project(self.descr1, shape_ind=1).detach().cpu().numpy()  # (n_ev1, n_descr)
        descr2_red = self.project(self.descr2, shape_ind=2).detach().cpu().numpy()  # (n_ev2, n_descr)
        
        # Compute multiplicative operators associated to each descriptor
        list_descr = []
        if mu_des > 0:
            if verbose:
                print('Computing commutativity operators')
            list_descr = self.compute_descr_op()  # (n_descr, ((k1,k1), (k2,k2)) )

        # Compute orientation operators associated to each descriptor
        #TODO: implement based on shapes!
        orient_op = []
        if mu_orient > 0:
            raise ValueError("orientation regularization not implemented!")
            if verbose:
                print('Computing orientation operators')
            orient_op = self.compute_orientation_op(reversing=orient_reversing)  # (n_descr,)
          
        # Compute mask for regularization
        mask = self.get_mask(optmask=optmask)

        self.descr1 = self.descr1.detach().cpu().numpy()
        self.descr2 = self.descr2.detach().cpu().numpy()
        
        # Identity matrix for coupling loss (including rank computation)
        I = np.identity(self.k2)
        if rank:
            e1, e2 = self.shape1["evals"][:self.k1],  self.shape2["evals"][:self.k2]
            max_e1 = max(e1)
            est_rank = sum([ev - max_e1 < 0 for ev in e2])
            for k in range(est_rank, self.k2):
                I[k,k] = 0

        # Arguments for the optimization problem
        args = (descr1_red, descr2_red, I, list_descr, orient_op, mask, mu_pres, mu_coup, mu_mask, mu_des, mu_orient) # to add weight matrix W
        
        # Initialization of C1 and C2
        C1, C2 = self.get_x0(optinit, descr1_red = descr1_red, descr2_red = descr2_red, mu_mask = mu_mask, mask = mask)         
        
        # Optimization
        res = minimize(loss, torch.concat((C1.ravel(), C2.ravel())), method = 'L-BFGS-B', jac=loss_grad, args=args)
        
        sol = res.x
        C1sol, C2sol = sol[0:len(sol)//2], sol[len(sol)//2 : len(sol)]
        self.C1, self.C2 = np.reshape(C1sol, (self.k2,self.k1)), np.reshape(C2sol, (self.k1,self.k2))
        nit = res.nit
        print('nit:' + str(nit))

        return self.C1, self.C2
        
    def get_mask(self, optmask = 'lbo'):
        """
        Computes a maask for regularization

        Parameters
        -----------------
        optmask : 'lbo' (| 'resolvent' | 'slanted' |) option for mask calculation

        Output
        -----------------
        mask : calculated mask
        """

        e1, e2 = self.shape1["evals"][:self.k1],  self.shape2["evals"][:self.k2]
        if optmask == 'resolvent': # Resolvent mask (Commutativity with resolvent of LBO)
            #TODO: find a good value for gammaa
            mask = self.resolvent_matrix(e1, e2, gamma=0.8)
        elif optmask == 'slanted': # Slanted mask
            mask = self.slanted_matrix(e1, e2)
        elif optmask == 'lbo': # Squared difference of eigenvalues (LBO commutativity)
            mask = np.square(e1[None, :self.k1] - e2[:self.k2, None])
            mask /= mask.sum()
        else:
            raise ValueError(f"Option '{optmask}' for computing the mask is not available!")
        
        return mask.detach().cpu().numpy()
        
    def resolvent_matrix(self, evals1, evals2, device="cpu", gamma=0.5):
        scaling_factor = max(torch.max(evals1), torch.max(evals2)).to(device)
        evals1, evals2 = evals1.to(device) / scaling_factor, evals2.to(device) / scaling_factor
        evals_gamma1, evals_gamma2 = (evals1 ** gamma)[None, :], (evals2 ** gamma)[:, None]

        M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
        M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
        return M_re.square() + M_im.square()
        
    def slanted_matrix(self, e1, e2):
        max_e1 = max(e1)
        est_rank = sum([ev - max_e1 < 0 for ev in e2])

        W = torch.zeros((self.k2,self.k1));
        
        for i in range(self.k2):
            for j in range(self.k1):
                origin = torch.tensor([1,1,0])

                slope = est_rank/self.k1
                direction = torch.tensor([1, slope])
                direction = direction / torch.linalg.norm(direction)
                direction = torch.concat((direction, torch.tensor([0])))
                W[i,j] = torch.exp(-0.03*torch.sqrt(torch.tensor(i**2 + j**2)))*torch.linalg.norm(torch.cross(direction, torch.tensor([i, j, 0], dtype=torch.float32)-origin))
        return W
        
    def get_x0(self, optinit, descr1_red = 0, descr2_red = 0, mu_mask = 0, mask = 0):
        """
        Returns the initial functional map for optimization. not used but could be

        Parameters
        ------------------------
        optinit : 'random' | 'identity' | 'zeros' initialization.

        Output
        ------------------------
        C1 : initialization of the C1 matrix
        C2 : initialization of the C2 matrix
        """
        ev1 = self.shape1["evals"]
        ev2 = self.shape2["evals"]

        if optinit == 'nocoup': 
            # Linear-system initialization (solution of C1 and C2 if coupling is removed)
            A_A_t = descr1_red @ descr1_red.T
            B_A_t = descr2_red @ descr1_red.T
            B_B_t = descr2_red @ descr2_red.T
            A_B_t = descr1_red @ descr2_red.T
            mask_t = mask.T
            C1, C2 = np.zeros((self.k2, self.k1)), np.zeros((self.k1, self.k2))
            for i in range(self.k2):
                C1[i] = np.linalg.solve(A_A_t + mu_mask * np.diag(mask[i]), B_A_t[i])
            for i in range(self.k1):
                C2[i] = np.linalg.solve(B_B_t + mu_mask * np.diag(mask_t[i]), A_B_t[i]) 
            C1, C2 = torch.tensor(C1), torch.tensor(C2)

        elif optinit == 'identity': # Close-to-identity initialization
            C1, C2 = torch.eye(self.k2, self.k1), torch.eye(self.k1, self.k2)

        else: # Zero initialization
            C1, C2 = torch.zeros((self.k2, self.k1)), torch.zeros((self.k1, self.k2))

        return C1, C2
        
        
    def compute_descr_op(self):
        """
        Compute the multiplication operators associated with the descriptors

        Output
        ---------------------------
        operators : n_descr long list of ((k1,k1),(k2,k2)) operators.
        """

        e1 = self.shape1["evecs"]
        e2 = self.shape2["evecs"]

        pinv1 = e1[:, :self.k1].T @ self.A1  # (k1,n)
        pinv2 = e2[:, :self.k2].T @ self.A2  # (k2,n)

        list_descr = [
                      ((pinv1@(self.descr1[:, i, None] * e1[:, :self.k1])).detach().cpu().numpy(),
                       (pinv2@(self.descr2[:, i, None] * e2[:, :self.k2])).detach().cpu().numpy()
                       )
                      for i in range(self.descr1.shape[1])
                      ]

        return list_descr

    def compute_orientation_op(self, reversing=False, normalize=False):
        """
        Compute orientation preserving or reversing operators associated to each descriptor.

        Parameters
        ---------------------------------
        reversing : whether to return operators associated to orientation inversion instead
                    of orientation preservation (return the opposite of the second operator)
        normalize : whether to normalize the gradient on each face. Might improve results
                    according to the authors

        Output
        ---------------------------------
        list_op : (n_descr,) where term i contains (D1,D2) respectively of size (k1,k1) and
                  (k2,k2) which represent operators supposed to commute.
        """
        raise ValueError("orientation regularizer not available yet!")
        # n_descr = self.descr1.shape[1]

        # # Precompute the inverse of the eigenvectors matrix
        # pinv1 = self.shape1["evecs"][:, :self.k1].T @ torch.diag(self.shape1["mass"])  # (k1,n)
        # pinv2 = self.shape2["evecs"][:, :self.k2].T @ torch.diag(self.shape2["mass"])  # (k2,n)

        # # Compute the gradient of each descriptor
        # grads1 = [self.mesh1.gradient(self.descr1[:, i], normalize=normalize) for i in range(n_descr)]
        # grads2 = [self.mesh2.gradient(self.descr2[:, i], normalize=normalize) for i in range(n_descr)]

        # # Compute the operators in reduced basis
        # can_op1 = [pinv1 @ self.mesh1.orientation_op(gradf) @ self.mesh1.eigenvectors[:, :self.k1]
        #            for gradf in grads1]

        # if reversing:
        #     can_op2 = [- pinv2 @ self.mesh2.orientation_op(gradf) @ self.mesh2.eigenvectors[:, :self.k2]
        #                for gradf in grads2]
        # else:
        #     can_op2 = [pinv2 @ self.mesh2.orientation_op(gradf) @ self.mesh2.eigenvectors[:, :self.k2]
        #                for gradf in grads2]

        # list_op = list(zip(can_op1, can_op2))

        # return list_op
    

    def project(self, func, k=None, shape_ind=1):
        """
        Projects a function on the LB basis

        Parameters
        -----------------------
        func    : array - (n1|n2,p) evaluation of the function
        shape_ind : int  1 | 2 index of the shaape on which to encode

        Output
        -----------------------
        encoded_func : (n1|n2,p) array of decoded f
        """
        if k is None:
            k = self.k1 if shape_ind == 1 else self.k2

        if shape_ind == 1:
            evecs = self.shape1["evecs"]
            A = self.A1
        elif shape_ind == 2:
            evecs = self.shape2["evecs"]
            A = self.A2
        else:
            raise ValueError(f'Only indices 1 or 2 are accepted, not {shape_ind}')

        return evecs[:,:k].t() @ A @ func

    def l2_inner(self, func1, func2, shape_ind):
        """
        Return the L2 inner product of two functions, or pairwise inner products if lists
        of function is given.

        For two functions f1 and f2, this returns f1.T @ A @ f2 with A the area matrix.

        Parameters
        -----------------
        func1 : (n,p) or (n,) functions on one shape
        func2 : (n,p) or (n,) functions on one shape
        shape_ind : 1 or 2, indicating on which shape the inner product should be calculated

        Returns
        -----------------
        sqnorm : (p,) array of L2 inner product or a float only one function per argument
                  was provided.
        """
        assert func1.shape == func2.shape, "Shapes must be equal"

        if shape_ind == 1:
            A = self.A1
        elif shape_ind == 2:
            A = self.A2
        else:
            raise ValueError("shape_ind has to be one of 1 and 2")

        if func1.ndim == 1:
            return func1 @ A @ func2

        return torch.einsum('np,np->p', func1, A@func2)
                  