import numpy as np

from pyFM.optimize.base_functions import descr_preservation, LB_commutation, descr_preservation_grad, LB_commutation_grad


def L21(X):
    return np.sqrt(np.square(X).sum(axis = 0)).sum() 


def L21_grad(X):
    return 0


def loss(C, A, B, ev_sqdiff, mu_coup, mu_LB): 
    """
    Evaluation of the loss for coupled functional maps computation
    Parameters:
    ----------------------
    C               : (2*K2*K1) couple maps C1, C2 map
    A               : (K1,p) descriptors on first basis
    B               : (K2,p) descriptros on second basis
    ev_sqdiff       : (K2,K1) [normalized] matrix of squared eigenvalue differences
    mu_coup         : scaling of the coupling constrain term
    mu_LB           : scaling laplacian commutativity term
    Output
    ------------------------
    loss : float - value of the loss
    """
    k1, k2 = A.shape[0], B.shape[0] 
    
    C1, C2 = C[0:len(C)//2], C[len(C)//2 : len(C)]
    
    C1, C2 = np.reshape(C1, (k2,k1)), np.reshape(C2, (k1,k2))
    
    I = np.identity(k2)
    
    loss = 0
    
    # Descriptors preservation
    loss += descr_preservation(C1, A, B) + descr_preservation(C2, B, A)
    
    # Coupling condition
    loss += mu_coup * descr_preservation(C1, C2, I)
    
    # LBO commutativity
    loss += mu_LB * (LB_commutation(C1, ev_sqdiff) + LB_commutation(C2.T, ev_sqdiff))
                     
    return loss 
                     
# (set up gradient of loss function -> for optimizer)


def loss_grad(C, A, B, ev_sqdiff, mu_coup, mu_LB):
    """
    Evaluation of the gradient of the loss for coupled functional maps computation

    Parameters:
    ----------------------
    C               : (2*K2*K1) couple maps C1, C2 map
    A               : (K1,p) descriptors on first basis
    B               : (K2,p) descriptros on second basis
    ev_sqdiff       : (K2,K1) [normalized] matrix of squared eigenvalue differences
    mu_coup         : scaling of the coupling constrain term
    mu_LB           : scaling laplacian commutativity term
    
    Output
    ------------------------
    loss_grad :  - array value of the gradient of the loss
    """
    k1, k2 = A.shape[0], B.shape[0] 
    
    C1, C2 = C[0:len(C)//2], C[len(C)//2 : len(C)]
    
    C1, C2 = np.reshape(C1, (k2,k1)), np.reshape(C2, (k1,k2))
    
    I = np.identity(k2)
    
    
    grad_C1 = 0
    grad_C2 = 0
    
    
    grad_C1 += descr_preservation_grad(C1, A, B)
    grad_C2 += descr_preservation_grad(C2, B, A)
    
    grad_C1 += mu_coup * descr_preservation_grad(C1, C2, I)
    grad_C2 += mu_coup * ( C2.T @ (C1 @ C2 - I) )
    
    grad_C1 += mu_LB * LB_commutation_grad(C1, ev_sqdiff)
    grad_C2 += mu_LB * LB_commutation_grad(C2, ev_sqdiff.T)
    
   
    return np.concatenate((grad_C1.ravel(),grad_C2.ravel()))

    