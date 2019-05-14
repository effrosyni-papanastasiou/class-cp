"""
This module holds diffent algorithms to compute the CP-CLASS decomposition.
"""
import logging
import time
import numpy as np
from numpy import array, dot, ones, sqrt
from scipy.linalg import pinv, inv
from numpy.random import rand
from sktensor import * 
from sktensor.core import nvecs, norm
from sktensor.ktensor import ktensor
import tensorly as tl
import tensortools as tt
from tensortools.operations import unfold as tt_unfold, khatri_rao

_log = logging.getLogger('CP')
_DEF_MAXITER = 500
_DEF_INIT = 'nvecs'
_DEF_CONV = 1e-5
_DEF_FIT_METHOD = 'full'
_DEF_TYPE = np.float

__all__ = [
    'als',
    'opt',
    'wopt'
]


def als(X, Yl, rank, **kwargs):
    """
    Alternating least-sqaures algorithm to compute the CP decomposition taking into 
    consideration the labels of the set
    Yl -> lx1
    X -> pxuxu
    """

    # init options
    ainit = kwargs.pop('init', _DEF_INIT)
    maxiter = kwargs.pop('max_iter', _DEF_MAXITER)
    fit_method = kwargs.pop('fit_method', _DEF_FIT_METHOD)
    conv = kwargs.pop('conv', _DEF_CONV)
    dtype = kwargs.pop('dtype', _DEF_TYPE)
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    N = X.ndim
    normX = norm(X)
    
    Yl = np.asarray(Yl)
    Yl = np.reshape(Yl, (-1,1))
    normYl = np.linalg.norm(Yl)
    
    U = _init(ainit, X, N, rank, dtype)
    fit = 0
    
    vecX = np.reshape(X, (np.product(X.shape),))
    
    W = ones((rank,1), dtype=dtype)
    
    l = Yl.shape[0]
    p = 240
    D = np.zeros((l,p))
    for i in range(l):
      for j in range(l):
        if i==j: 
          D[i,j] = 1 
    
    for itr in range(maxiter):
        fitold = fit
        for n in range(N):
            Unew = X.uttkrp(U, n)
            # Y is ZtZ
            Y = ones((rank, rank), dtype=dtype)
            for i in (list(range(n)) + list(range(n + 1, N))):
                Y = Y * dot(U[i].T, U[i])
            if n!=1:
                # Updates remain the same for U0,U2
                Unew = Unew.dot(pinv(Y))
            else:
                Ip = np.identity(p)
                IptIp = dot(Ip.T,Ip)
                GtG = np.kron(Y,IptIp)
                vecA = np.reshape(U[1], (np.product(U[1].shape),1))
                #print('vecA shape', vecA.shape)
                GtvecX1 = dot(GtG,vecA)

                L = np.kron(W.T,D)
                LtL = dot(L.T, L)

                Sum1 = inv(GtG + LtL)
                dot0 = dot(L.T,Yl)
                Sum2  = GtvecX1 + dot0
                vecA = dot(Sum1,Sum2)
  
                Unew = np.reshape(vecA, (p,rank))
  
            # Normalize
            if itr == 0:
                lmbda = sqrt((Unew ** 2).sum(axis=0))
            else:
                lmbda = Unew.max(axis=0)
                lmbda[lmbda < 1] = 1
            
            U[n] = Unew / lmbda
        
        # update W 
        AtDt = dot(U[1].T,D.T)
        DA = dot(D,U[1])
        inv1 = inv(dot(AtDt,DA))
        dot2 = dot(AtDt,Yl)
        W = dot(inv1,dot2)

        P = ktensor(U, lmbda)
        A = U[1]
        Ai = A[192:]

        ypred = dot(Ai, W)
        ypred[abs(ypred) > 0.5] = 1
        ypred[abs(ypred) < 0.5] = 0
        
        DAW = dot(DA,W)        
        normDAW = np.linalg.norm(DAW)
        
        if fit_method == 'full':
            normresidual1 = normX ** 2 + P.norm() ** 2 - 2 * P.innerprod(X)
            normresidual2 = normYl ** 2 + normDAW ** 2 - 2 * dot(Yl.T,DAW)
            normresidual = normresidual1 + normresidual2
            #fit = 1 - (normresidual / normX ** 2)
            fit = normresidual
        else:
            fit = itr

        fitchange = abs(fitold - fit)/fitold
        #print('fitchange:',fitchange)

        if itr > 0 and fitchange < conv:
            #print(ypred)
            break
    
    #print('pred:', ypred)
    ypred[abs(ypred) > 0.5] = 1
    ypred[abs(ypred) < 0.5] = 0 
    #print(ypred)

    return P, W, ypred, fit, itr


def opt(X, rank, **kwargs):
    ainit = kwargs.pop('init', _DEF_INIT)
    maxiter = kwargs.pop('maxIter', _DEF_MAXITER)
    conv = kwargs.pop('conv', _DEF_CONV)
    dtype = kwargs.pop('dtype', _DEF_TYPE)
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    N = X.ndim
    U = _init(ainit, X, N, rank, dtype)


def wopt(X, rank, **kwargs):
    raise NotImplementedError()


def _init(init, X, N, rank, dtype):
    """
    Initialization for CP models
    """
    #Uinit = [None for _ in range(N)]
    Uinit = [None for _ in range(N)]
    if isinstance(init, list):
        Uinit = init
    elif init == 'random':
        print('Random Initialization')
        for n in range(1, N):
            Uinit[n] = array(rand(X.shape[n], rank), dtype=dtype)
        print('Ok init')
    elif init == 'nvecs':
        for n in range(1, N):
            Uinit[n] = array(nvecs(X, n, rank), dtype=dtype)
    else:
        raise 'Unknown option (init=%s)' % str(init)
    #for n in range(1, N):
    #        Uinit[n] = array(rand(X.shape[n], rank), dtype=dtype)
    return Uinit

# vim: set et:
