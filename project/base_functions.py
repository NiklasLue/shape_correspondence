import numpy as np

from pyFM.optimize.base_functions import descr_preservation, LB_commutation, descr_preservation_grad, LB_commutation_grad, op_commutation, op_commutation_grad, oplist_commutation, oplist_commutation_grad


def L21(X):
    return np.sqrt(np.square(X).sum(axis = 0)).sum() 


def L21_grad(X):
    return 0

def oplist_commutation_C2(C, op_list):
    """
    Compute the operator commutativity constraint for a list of pairs of operators
    Can be used with a list of descriptor multiplication operator

    Parameters
    ---------------------
    C   : (K2,K1) Functional map
    op_list : list of tuple( (K1,K1), (K2,K2) ) operators on first and second basis

    Output
    ---------------------
    energy : (float) sum of operators commutativity squared norm
    """
    energy = 0
    for (op1, op2) in op_list:
        energy += op_commutation(C, op2, op1)

    return energy

def oplist_commutation_grad_C2(C, op_list):
    """
    Compute the gradient of the operator commutativity constraint for a list of pairs of operators
    Can be used with a list of descriptor multiplication operator

    Parameters
    ---------------------
    C   : (K2,K1) Functional map
    op_list : list of tuple( (K1,K1), (K2,K2) ) operators on first and second basis

    Output
    ---------------------
    gradient : (K2,K1) gradient of the sum of operators commutativity squared norm
    """
    gradient = 0
    for (op1, op2) in op_list:
        gradient += op_commutation_grad(C, op2, op1)
    return gradient


def oplist_commutation_t(C, op_list):
    """
    Compute the operator commutativity constraint for a list of pairs of operators
    Can be used with a list of descriptor multiplication operator

    Parameters
    ---------------------
    C   : (K2,K1) Functional map
    op_list : list of tuple( (K1,K1), (K2,K2) ) operators on first and second basis

    Output
    ---------------------
    energy : (float) sum of operators commutativity squared norm
    """
    energy = 0
    for (op1, op2) in op_list:
        energy += op_commutation(C, op1, op2)

    return energy





def loss(C, A, B, list_descr, orient_op, ev_sqdiff, mu_pres, mu_coup, mu_LB, mu_des, mu_orient): 
    """
    Evaluation of the loss for coupled functional maps computation
    Parameters:
    ----------------------
    C               : (2*K2*K1) couple maps C1, C2 map
    A               : (K1,p) descriptors on first basis
    B               : (K2,p) descriptros on second basis
    ev_sqdiff       : (K2,K1) [normalized] matrix of squared eigenvalue differences
    mu_pres         : scaling descriptor preservation term
    mu_coup         : scaling of the coupling constrain term
    mu_LB           : scaling laplacian commutativity term
    mu_des          : scaling descriptor commutativity term
    mu_orient       : scaling orientation commutativity term
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
    if mu_pres >0:
        loss += mu_pres * (descr_preservation(C1, A, B) + descr_preservation(C2, B, A))
    
    # Coupling condition
    if mu_coup >0:
        loss += mu_coup * descr_preservation(C1, C2, I)
    
    # LBO commutativity
    if mu_LB >0:
        loss += mu_LB * (LB_commutation(C1, ev_sqdiff) + LB_commutation(C2.T, ev_sqdiff))
    
    # Descriptor Commutativity
    if mu_des >0:
        loss += mu_des * (oplist_commutation(C1, list_descr) + oplist_commutation_C2(C2, list_descr)) #oplist_commutation(C2.T, orient_op)
    
    #Orientation
    if mu_orient >0:
        loss += mu_orient * (oplist_commutation(C1, orient_op) + oplist_commutation_C2(C2, orient_op)) #oplist_commutation(C2.T, orient_op)
                     
    return loss 
                     
# (set up gradient of loss function -> for optimizer)


def loss_grad(C, A, B, list_descr, orient_op, ev_sqdiff, mu_pres, mu_coup, mu_LB, mu_des, mu_orient):
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
    
    if mu_pres >0:
        grad_C1 += descr_preservation_grad(C1, A, B)
        grad_C2 += descr_preservation_grad(C2, B, A)
    
    if mu_coup >0:
        grad_C1 += mu_coup * descr_preservation_grad(C1, C2, I)
        grad_C2 += mu_coup * ( C1.T @ (C1 @ C2 - I) )
    
    if mu_LB >0:
        grad_C1 += mu_LB * LB_commutation_grad(C1, ev_sqdiff)
        grad_C2 += mu_LB * LB_commutation_grad(C2, ev_sqdiff.T)
     
    if mu_des >0:
        grad_C1 += mu_des * oplist_commutation_grad(C1, list_descr)
        grad_C2 += mu_des * oplist_commutation_grad_C2(C2, list_descr) #oplist_commutation_grad(C2.T, orient_op).T
    
    if mu_orient >0:
        grad_C1 += mu_orient * oplist_commutation_grad(C1, orient_op)
        grad_C2 += mu_orient * oplist_commutation_grad_C2(C2, orient_op) #oplist_commutation_grad(C2.T, orient_op).T
    
   
    return np.concatenate((grad_C1.ravel(),grad_C2.ravel()))


