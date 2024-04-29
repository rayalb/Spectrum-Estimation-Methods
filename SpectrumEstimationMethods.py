#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 10:38:20 2022

Spectrum Estimation Methods.

@author: ray
"""

import numpy as np
from scipy import linalg, signal
import myFunctions as myFunc
#import ModelOrderSelection as mos

def projHankel(Z):
    '''
    Orthogonal projection onto the subspace of Hankel matrices

    Parameters
    ----------
    Z : Matrix
        Input Matrix.

    Returns
    -------
    H : Matrix
        Output matrix with Hankel structure.

    '''

    n, m = Z.shape
    if n < m:
        Z = Z.T
        transp_Z = 1
        n, m = Z.sahpe
    else:
        transp_Z = 0
        
    B = np.zeros((m+n,m), dtype = complex)
    B[:n,:] = Z
    N = np.sum(np.reshape(B.flatten('F')[:(n+m-1)*m], (n+m-1,m), 
                          order = 'F'), axis = 1)         
    D = np.append(np.append(np.arange(1,m), m*np.ones((1, n-m+1))), np.arange(1,m)[::-1])
    k = np.divide(N,D)
    H = linalg.hankel(k[:n],k[n-1:])
    
    if transp_Z == 1:
        H = H.T
        
    return H

##############################################################################

def blkHankel(Y):
    '''
    Generate an array of Hankel Matrices from n x K matrix 
    Y = [y_1, y_2, ..., y_k], with y_i \in C^n.
    
    H = [H_1 H_2 ... H_K]
    where H_i = Hankel(y_i), with i = 1,2,...,K

    Parameters
    ----------
    Y : Matrix with signals in its columns

    Returns
    -------
    H : Array of Hankel matrices

    '''
    n = int((Y.shape[0] + 1)/2)
    K = Y.shape[1]
    H = np.zeros((n, int(n*K)), dtype = complex)
    for nn in np.arange(K):
        y = Y[:,nn]
        H[:, np.arange(n*nn, n*(nn+1))] = linalg.hankel(y[:n], y[n-1:])
        
    return H

#############################################################################

def ESPRIT(Hx, r):
    '''
    Estimation of Signal Parameters via Rotational Invariance 
    Thechniques (ESPRIT).
    
    Roy. R.; Kailath, T. (1989). "Esprit- Estimation of Signal Parameters Via 
    Rotational Invariance Techniques". IEEE Transactions on Acoustics, Speech, 
    and Signal Processing. 37 (7): 984–995.
    
    Parameters
    ----------
    Hx : Hankel matrix built from vector x
    r : Selected order

    Returns
    -------
    zeta : Eigenvalues corresponding Signal parameters.

    '''
    
    U, _, _ = linalg.svd(Hx)
    Ul = U[:-1, :r]; Uf = U[1:, :r]
    Phi = linalg.pinv(Ul) @ Uf
    zeta, _ = linalg.eig(Phi)
    
    return zeta

#############################################################################

def sum_Hank(A):
    '''
    Sum anti-diagonals of a Hankel matrix
    
    Andersson, et. al. "A new frequency Estimation method for equally 
           and unequally spaced data". IEEE Trans. Signal Processing. 
           Vol. 62, NO. 21, Nov 2014

    Parameters
    ----------
    A : Hankel Matrix

    Returns
    -------
    f : vector with the sumed anti-diagonals in each component.

    '''
    N = A.shape[0]
    f = np.zeros(2*N-1, dtype = complex)
    A = np.flipud(A)
    
    for nn in np.arange(-N, N-1, 1):
        f[nn+N] = np.sum(np.diag(A,nn+1))
        
    return f

#############################################################################

def expo_est_blk(Y, r, rho = 0.025, niter = 100):
    '''
    Frequency estimation function by a block data.
    
    Andersson, et. al. "A new frequency Estimation method for equally 
           and unequally spaced data". IEEE Trans. Signal Processing. 
           Vol. 62, NO. 21, Nov 2014

    Parameters
    ----------
    Y : Matrix of data samples, Y = [y_1, ..., y_K] 
    r : order
    rho: parameter
    niter = number of iterations
    
    Returns
    -------
    G : Matrix of approximated data samples.
    zeta: Estimated eigenvalues.
    '''
    H = blkHankel(Y)
    '''
    TODO: ver como combinar el repo de modelOrderEstimation 
    con este.
    if r == None:
        #r =  mos.ModelOrderSelection_Constrain(H, func = 'SAMOS')
        r, _ = mos.SAMOS(H)
    '''    
    p = Y.shape[1]
    n = int((Y.shape[0] - 1)/2.0)
    w = np.append(range(1, n+2), range(n, 0, -1))
    mu = np.ones(Y.shape[0]); M = mu[mu==1].size
    Lambda = np.zeros(((n+1), int((n+1)*p)), dtype = complex)
    
    G = Y
    eps_abs = 1e-2; eps_rel = 1e-4
     
    for ii in range(niter):
        H = blkHankel(G)
        B = H - Lambda/rho
        U, s, V = linalg.svd(B)
        A = U[:,:r] @ np.diag(s[:r]) @ V[:r,:]
        
        for jj in np.arange(p):
            A_aux = A[:, np.arange((n+1)*jj, (n+1)*(jj+1))]
            f = Y[:, jj]
            G[:, jj] = np.linalg.pinv((n+1)/M*np.diag(mu) + rho*np.diag(w)) @ \
                ((n+1)/M*mu*f + sum_Hank(rho*A_aux + Lambda[:, np.arange((n+1)*jj,(n+1)*(jj+1))]))
        
        H = blkHankel(G); R = A-H
        Lambda = Lambda + rho*R
        '''
        TODO: Ver convergencia del ADMM
        S = rho*blkHankel(G-Y)
        
        eps_prim = (2*(n+1)+1)**2*eps_abs + eps_rel*np.maximum(linalg.norm(A, ord = 'fro')**2, linalg.norm(H, ord='fro')**2)
        eps_dual = (2*(n+1)+1)**2*eps_abs + eps_rel*linalg.norm(Lambda, ord='fro')**2
        if (linalg.norm(R, ord = 'fro')**2<eps_prim) & (linalg.norm(S, ord = 'fro')**2<eps_dual):
            break
        '''
    	for tt in np.arange(p):
        	G[:, tt] = sum_Hank(A[:, np.arange((n+1)*tt, (n+1)*(tt+1))])/w
         
    H = blkHankel(G)
    zeta = ESPRIT(H, r)
        
    return H, zeta

##############################################################################

def Spectrum_ShiftZoom(X, t, wc, dw, Q, order = None):
    '''
    Shift and Zoom algorithm for sprectral estimation.
    
    Albert, R. and Galarza, C. "Spectrum estimation using frequency shifting 
    and decimation". IET Signal Processing. 2020.

    Parameters
    ----------
    X : data signal
    t : time array
    wc : center frequency
    dw : Bandwidth
    Q : Decimation factor
    order : Model order
        The default is 5.

    Returns
    -------
    zeta : eigenvalues.

    '''
    Fs = 1/(t[1]-t[0])
    m, p = X.shape
    
    Fcutoff = dw/Fs
    ntaps = 64
    
    shift = np.exp(-1j*wc*t).reshape(t.size,1)
    if  m/Q % 2 == 0:
        n = int(m/Q)-1
    else:
        n = int((m/Q))
    Y = np.zeros((n, p), dtype = complex)
        
    for ii in np.arange(p):
        data_aux = X[:,ii]*shift
        filter_x = myFunc.filt_signal(data_aux, ntaps, Fcutoff, Fs)
        Y[:, ii] = signal.resample_poly(filter_x, 1, Q)[:n]
    
    G, zeta = expo_est_blk(Y, order, niter = 100)
    
    w_h = wc + np.pi * dw
    w_l = wc - np.pi * dw
    
    xi = np.log(zeta[abs(zeta)>1e-10])*Fs/Q + 1j*wc
    xi_aux = [xx if (xx.imag < w_h) & (xx.imag > w_l) else np.NAN for xx in xi]
    zeta_aux = np.exp(np.array(xi_aux)/Fs)
    
    zeta = np.array([z if np.abs(z)<=1 else np.NaN for z in zeta_aux])
    
    return zeta

##############################################################################

def Cadzow(X, rank, tol = 1e-4, MaxIter = 10):
    '''
    Cadzow Algorithm for structured low-rank approximation. See
    
    J. A. Cadzow, "Signal enhancement - A composite property mapping 
                    algorithm," IEEE Trans. Acoust. Speech, Signal Process. 
                    vol 36, no. 1, pp. 42-62, 1988.

    Parameters
    ----------
    X : Input Matrix
    rank : desired rank
    tol : tolerance. The default is 1e-4.
    MaxIter : Maximum number of iteration. The default is 10.

    Returns
    -------
    H_last : Low rank Hankel approximation of X.
    '''
    H0 = X
    for ii in range(MaxIter):
        H = projHankel(H0)
        U, s, Vh = linalg.svd(H)
        H_last = U[:,:rank] @ np.diag(s[:rank]) @ Vh[:rank,:]
        
        if linalg.norm(H_last - H0, ord = 'fro') < tol*linalg.norm(H0, ord = 'fro'):
            break
        H0 = H_last
        
    return H_last
        