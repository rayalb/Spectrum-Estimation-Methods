#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:44:32 2022

@author: ray
"""

import numpy as np
from scipy import linalg, sparse

import SpectrumEstimationMethods as sem


def tensor_unfold(tensor, mode = 1):
    '''
     Tensor Unfold. 
    Kolda, T. and Bader, B. Tensor Decomposition and Applications. 
        SIAM Review, vol. 51, No. 3, pp 455-500.
    
    Parameters
    ----------
    tensor: data tensor, ndarray.
    mode : Tensor mode. Default is 1

    Returns
    -------
    Unfold Tensor.
    '''
    mode = mode - 1 # 
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F') 

def CPDecomposition_3dTensor(tensor, rank, maxIter = 100, showIter = False):
    '''
    Tensor CP decomposition of a 3d tensor using Alternate Least Squares (ALS).

     Kolda, T. and Bader, B. Tensor Decomposition and Applications. 
        SIAM Review, vol. 51, No. 3, pp 455-500.

    Parameters
    ----------
    tensor : tensor data, ndarray
        
    rank : integer
        tensor rank
    maxIter : integer, optional
        Number of iteration. The default is 100.
    
    Returns
    -------
    A, B, C : Matrices of the CP decomposition
    '''
    
    X_1 = tensor_unfold(tensor, mode = 1)
    X_2 = tensor_unfold(tensor, mode = 2)
    X_3 = tensor_unfold(tensor, mode = 3)
    
    B = linalg.svd(X_2)[0][:, :rank]
    C = linalg.svd(X_3)[0][:, :rank]
   
    for kk in range(maxIter):
        # Find A
        A_ini = linalg.pinv((C.conj().T @ C)*(B.conj().T @ B)) @ linalg.khatri_rao(C, B).conj().T
        A = X_1 @ A_ini.T
        
        # Find B
        B_ini = linalg.pinv((C.conj().T @ C)*(A.conj().T @ A)) @ linalg.khatri_rao(C, A).conj().T
        B  = X_2 @ B_ini.T
        
        # Find C
        C_ini = linalg.pinv((B.conj().T @ B)*(A.conj().T @ A)) @ linalg.khatri_rao(B, A).conj().T
        C = X_3 @ C_ini.T
        
        if kk % int(maxIter*.2) == 0 and showIter == True:
            res_A = np.square((A @ linalg.khatri_rao(C, B).T) - X_1)
            res_B = np.square((B @ linalg.khatri_rao(C, A).T) - X_2)
            res_C = np.square((C @ linalg.khatri_rao(B, A).T) - X_3)
            print("Iteration: ", kk, "| Loss (A): ", abs(res_A.mean()), 
                  "| Loss (B): ", abs(res_B.mean()), "| Loss (C): ", abs(res_C.mean()))
            
    return A, B, C
        

'''
TODO: funciones para descomposicion CP teniendo en cuenta la
        estructura Hankel de las slice del tensor. 

def someFunction(tensor, rank, lam = 1, niter = 200, beta = 0.1, rho = 1, norm_nuc = False):
    
    # Tensor Unfold
    X_1 = tensor_unfold(tensor, mode = 0);   # X_(1)
    X_2 = tensor_unfold(tensor, mode = 1);   # X_(2)
    X_3 = tensor_unfold(tensor, mode = 2);   # X_(3)
    
    # Initialite Matrices A^{(1)}, A^{(2)}, y A^{(3)}
    A1 = linalg.svd(X_1)[0][:, :rank]; n = int((A1.shape[0]-1)/2)+1
    A2 = linalg.svd(X_2)[0][:, :rank]; m = int((A2.shape[0]-1)/2)+1
    A3 = linalg.svd(X_3)[0][:, :rank]; L = A3.shape[0]
    
    U1 = np.zeros((n, rank*n), dtype=complex)
    U2 = np.zeros((m, rank*m), dtype=complex)
    U3 = np.zeros((L, rank), dtype = complex)
    
    M1 = np.zeros((n, rank*n), dtype = complex)
    M2 = np.zeros((m, rank*m), dtype = complex)
    M3 = np.zeros((L, rank), dtype = complex)
    
    w1 = np.append(range(1, n), range(n, 0, -1))
    w2 = np.append(range(1, m), range(m, 0, -1))
    err = 1000; kk = 0
    for kk in range(niter):
        # Update A^(1) and A^(2)
        G1_k = linalg.khatri_rao(A3, A2).T
        #G2_k = linalg.khatri_rao(A3, A1).T
        #G3_k = linalg.khatri_rao(A2, A1).T
        
        CH1 = np.zeros((2*n-1, rank), dtype = complex)
        CH2 = np.zeros((2*m-1, rank), dtype = complex)
        for ii in range(rank):
            CH1[:,ii] = sem.sum_Hank(U1[:,ii*n:(ii+1)*n]-M1[:, ii*n:(ii+1)*n]/beta)
            CH2[:,ii] = sem.sum_Hank(U2[:,ii*m:(ii+1)*m]-M2[:, ii*m:(ii+1)*m]/beta)

        P1 =  lam*(X_1 @ G1_k.conj().T) + beta*CH1
        #P2 =  lam*(X_2 @ G2_k.conj().T) + beta*CH2
        
        for ii in range(A1.shape[0]):
            A1[ii,:] = P1[ii,:] @ linalg.inv(lam*G1_k @ G1_k.conj().T + beta*w1[ii]*np.eye(rank))
        
        G2_k = linalg.khatri_rao(A3, A1).T
        P2 =  lam*(X_2 @ G2_k.conj().T) + beta*CH2
        for jj in range(A2.shape[0]):
            A2[jj,:] = P2[jj,:] @ linalg.inv(lam*G2_k @ G2_k.conj().T + beta*w2[jj]*np.eye(rank))
        
        # Update A^(3)
        G3_k = linalg.khatri_rao(A2, A1).T
        A3 = (lam*(X_3 @ G3_k.conj().T) + beta*U3 - M3) @ linalg.inv(lam*G3_k @ G3_k.conj().T + beta*np.eye(rank))
        
      
        # Update U_1^(1), U_2^(1), ..., U_r^(1)
        # and U_1^(2), U_2^(2), ..., U_r^(2)
        for ii in range(rank):
           ai_1 = A1[:,ii]
           A1_aux = linalg.hankel(ai_1[:n], ai_1[n-1:,]) + M1[:,ii*n:(ii+1)*n]/beta
           U, S, Vh = linalg.svd(A1_aux); 
           if norm_nuc:
               S = S-1/beta; S[S<0] = 0
               U1[:, ii*n:(ii+1)*n] = U @ np.diag(S) @ Vh
           else:
               U1[:, ii*n:(ii+1)*n] = U[:,:1] @ np.diag(S[:1]) @ Vh[:1,:]
           
           ai_2 = A2[:,ii]
           A2_aux = linalg.hankel(ai_2[:m], ai_2[m-1:]) + M2[:,ii*m:(ii+1)*m]/beta
           U, S, Vh = linalg.svd(A2_aux); 
           if norm_nuc:
               S = S-1/beta; S[S<0] = 0
               U2[:, ii*m:(ii+1)*m] = U @ np.diag(S) @ Vh
           else:
               U2[:, ii*m:(ii+1)*m] = U[:,:1] @ np.diag(S[:1]) @ Vh[:1,:]
            
        # Update U^(3)
        U, S, Vh = linalg.svd(A3 + M3/beta) 
        if norm_nuc:
            S = S-1/beta; S[S<0] = 0; l = len(S)
            U3 = U[:,:l] @ np.diag(S) @ Vh[:l,:]
        else:
            U3 = U[:,:rank] @ np.diag(S[:rank]) @ Vh[:rank,:]
         
       
        # Update M_1^(1), M_2^(1), ..., M_r^(1)
        # and M_1^(2), M_2^(2), ..., M_r^(2)
        for ii in range(rank):
           ai_1 = A1[:,ii]
           M1[:, ii*n:(ii+1)*n] = M1[:, ii*n:(ii+1)*n] + beta*(linalg.hankel(ai_1[:n], ai_1[n-1:,]) - U1[:,ii*n:(ii+1)*n])
           ai_2 = A2[:,ii]
           M2[:, ii*m:(ii+1)*m] = M2[:, ii*m:(ii+1)*m] + beta*(linalg.hankel(ai_2[:n], ai_2[n-1:,]) - U2[:,ii*m:(ii+1)*m])
            
        # Update M^(3)
        M3 = M3 + beta*(A3 - U3)
        beta = beta*rho
        
        err = linalg.norm(A1 @ A2.T - sem.projHankel(A1 @ A2.T ), ord = 'fro')
        kk+=1
        
    return A1, A2, A3
        
##############################################################################
      
def HankOp(n, m):
    
     # Hankel operator.

    S = np.zeros((n*m, m+n-1))
    for ii in range(m):
        S_aux = np.concatenate((np.zeros((n,ii)), np.eye(n), np.zeros((n,m-1-ii))), axis = 1)
        S[ii*n:(ii+1)*n,:] = S_aux
        
    return S
        
#############################################################################

def TensorHankelopt(tensor, rank, beta = 0.025, niter = 200):
    
    X_1 = tensor_unfold(tensor, mode = 0);   # X_(1)
    X_2 = tensor_unfold(tensor, mode = 1);   # X_(2)
    X_3 = tensor_unfold(tensor, mode = 2);   # X_(3)
    
    # Initialite Matrices A^{(1)}, A^{(2)}, y A^{(3)}
    A1 = linalg.svd(X_1)[0][:, :rank]; n = A1.shape[0]
    A2 = linalg.svd(X_2)[0][:, :rank]; m = A2.shape[0]
    A3 = linalg.svd(X_3)[0][:, :rank]; L = A3.shape[0]
    
    S = HankOp(n,m)
    Pi_S = sparse.csr_matrix(S @ linalg.pinv(S))
    del S
    U1 = np.zeros((n, m), dtype=complex)
    M = np.zeros((n, m), dtype = complex)
    
    for kk in range(niter):
        
        
        # Update A^(1) and A^(2)
        G1_k = linalg.khatri_rao(A3, A2).T
        
        T1 = np.concatenate((np.sqrt(beta/2)*Pi_S.dot(linalg.kron(A2, np.eye(n))), 
                             -linalg.kron(G1_k.T, np.eye(n))), axis = 0)
        b1 = np.concatenate((np.sqrt(beta/2)*(M.flatten(order = 'F') - U1.flatten(order = 'F')),
                             X_1.flatten(order = 'F')))
        
        #vec_A1 = sparse.linalg.lsqr(T1, b1)[0]
        vec_A1 = np.linalg.lstsq(T1, b1, rcond=-1)[0]
        #vec_A1 = linalg.pinv(T1) @ b1
        A1 = vec_A1.reshape((n, rank), order = 'F')
        del T1, b1, vec_A1
        
        G2_k = linalg.khatri_rao(A3, A1).T
        
        T2 = np.concatenate((np.sqrt(beta/2)*Pi_S.dot(linalg.kron(np.eye(m), A1)), 
                             -linalg.kron(np.eye(m), G2_k.T)), axis = 0)
        b2 = np.concatenate((np.sqrt(beta/2)*(M.flatten(order = 'F') - U1.flatten(order = 'F')), 
                             X_2.T.flatten(order = 'F')))
        
        #vec_A2 = sparse.linalg.lsqr(T2, b2)[0]
        vec_A2 = np.linalg.lstsq(T2, b2, rcond=-1)[0]
        #vec_A2 = linalg.pinv(T2) @ b2
        A2 = vec_A2.reshape((m, rank))
        del T2, b2, vec_A2
        
        C = linalg.pinv((A2.conj().T @ A2)*(A1.conj().T @ A1)) @ linalg.khatri_rao(A2, A1).conj().T
        A3 = X_3 @ C.T
        
        H_hat = sem.projHankel(A1 @ A2.T)
        U, S, Vh = linalg.svd(H_hat + M/beta)
        U1 = U[:,:rank] @ np.diag(S[:rank]) @ Vh[:rank,:]
        
        M = M + beta*(H_hat - U1)
        #beta = beta*1.5
        #err = linalg.norm(A1 @ A2.T - H_hat , ord = 'fro')**2/linalg.norm(A1 @ A2.T, ord = 'fro')**2
        
        
    return A1, A2, A3

def lasso(vector, n, r, beta):
    u = np.zeros(n*r, dtype=complex)
    for jj in range(r):
        u_aux = vector[jj*n:(jj+1)*n]
        
        if linalg.norm(u_aux) < beta:
            u[jj*n:(jj+1)*n] = np.zeros(r, dtype=complex)
        else:
            u[jj*n:(jj+1)*n] = (linalg.norm(u_aux)-beta)/linalg.norm(u_aux)*u_aux
    return u

def BTD(tensor, rank, beta = 1e-3, maxIter = 100, structured = False):
    
    X_1 = tensor_unfold(tensor, mode = 0);   # X_(1)
    X_2 = tensor_unfold(tensor, mode = 1);   # X_(2)
    X_3 = tensor_unfold(tensor, mode = 2);   # X_(3)
    
    # Initialite Matrices A^{(1)}, A^{(2)}, y A^{(3)}
    A1 = linalg.svd(X_1)[0][:, :rank]; n = A1.shape[0]
    A2 = linalg.svd(X_2)[0][:, :rank]; m = A2.shape[0]
    A3 = linalg.svd(X_3)[0][:, :rank]; L = A3.shape[0]

    for ii in range(maxIter):
        # Update A^(1)
        T1 = np.concatenate((linalg.kron(linalg.khatri_rao(A3, A2), np.eye(n)), 
                            np.sqrt(beta)*np.eye(n*rank)), axis = 0)
        b1 = np.concatenate((X_1.flatten(order = 'F'), np.sqrt(beta)*A1.flatten(order = 'F')))
        
        u1_r = np.linalg.lstsq(T1, b1, rcond=-1)[0]
        A1 = np.reshape(lasso(u1_r, n, rank, beta), n, rank)
         # Update A^(2)
        T2 = np.concatenate((linalg.kron(linalg.khatri_rao(A3, A1), np.eye(m)), 
                             np.sqrt(beta)*np.eye(m*rank)), axis = 0)
        b2 = np.concatenate((X_2.flatten(order = 'F'), np.sqrt(beta)*A2.flatten(order = 'F')))
        
        u2_r = np.linalg.lstsq(T2, b2, rcond=-1)[0]
        A2 = np.reshape(lasso(u2_r, m, rank, beta), m, rank)
        
        # Update A^(3)
        T3 = np.concatenate((linalg.kron(linalg.khatri_rao(A2, A1)@ np.eye(rank),np.eye(L)),
                             np.sqrt(beta)*np.eye(L*rank)), axis = 0)
        b3 = np.concatenate((X_3.flatten(order = 'F'), np.sqrt(beta)*A3.flatten(order = 'F')))
        
        u3_r = np.linalg.lstsq(T3, b3, rcond=-1)[0]
        A3 = np.reshape(lasso(u3_r, L, rank, beta), L, rank)
        
    return A1, A2, A3
        
'''        
        
        
        
    
    
            
        
    