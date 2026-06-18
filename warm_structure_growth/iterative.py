import numpy as np
from scipy.integrate import simpson

def growth_functions(ak,y,Tfs,iters=15):
  '''
  Evaluate the growth functions T^(a,b,...)_k(y,y').
  
  Parameters:
    
    ak: float
      alpha_k = sqrt(2)*v_eq*k/k_eq, where v_eq is the characteristic velocity
      at a_eq.
    
    y: array
      a/a_eq, time steps.
    
    Tfs: callable or list
      The "free streaming transfer function", which is the Fourier transform
      of the velocity distribution in units of the characteristic velocity.
      A list may be passed to use a different Tfs for T^(a), T^(b), ...
      
    iters: int
      Depth of iterative evaluation, which sets how precise the result will be.
      Default is 15.
    
  Returns:
    
    T^(a)_k(y,y'): 2-D array
      "Initial displacement" growth function (axes are y and y').
    
    T^(b)_k(y,y'): 2-D array
      "Initial kick" growth function (axes are y and y').
        
    T^(c,...)_k(y,y'): 2-D arrays
      More "initial displacement" growth functions corresponding to additional
      Tfs that were passed, if more than 2 were passed.
  '''
  
  F = np.log(y/(1.+np.sqrt(1.+y))**2)

  if callable(Tfs):
    Tfs = [Tfs]*2

  # The integral  int dy'' Q(y,y'') R(y'',y') [y''/sqrt(1+y'')]  with Simpson's
  # rule on the (uniform) ln(y) grid is a linear functional of the samples, so
  # it equals a matrix product  (Q * w) @ R  where w_j are the combined Simpson
  # weight and measure at node j. We obtain the exact Simpson weights once (the
  # rule's action on the basis vectors) so this reproduces scipy.simpson while
  # using a single BLAS dgemm.
  lny = np.log(y)
  w = simpson(np.eye(len(y)),x=lny,axis=1) * (y/np.sqrt(1.+y))

  dF = F[:,None]-F[None]

  # prepare T^(b,0)
  # array shape: y, y' (latest to earliest)
  Tfs_ = Tfs[1]
  Tb0 = np.tril(dF*Tfs_(ak*dF))
  np.fill_diagonal(Tb0,0.) # in case Tfs is not well behaved at 0

  # start iteration for T^(b)
  Tb = Tb0
  for i in range(iters-1):
    Tb = Tb0 + 1.5*((Tb*w)@Tb0)

  # evaluate T^(a,...)
  Tbw = Tb*w
  Ta = []
  for Tfs_ in Tfs[:1] + Tfs[2:]:
    Ta0 = np.tril(Tfs_(ak*dF))
    np.fill_diagonal(Ta0,1.) # in case Tfs is not well behaved at 0
    Ta += [Ta0 + 1.5*(Tbw@Ta0)]

  return tuple(Ta[:1] + [Tb] + Ta[1:])