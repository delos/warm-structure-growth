import numpy as np
from scipy.integrate import simpson

def growth(ak,y,Tfs,iters=15):
  '''
  
  Evaluate the growth functions T^(a)_k(y,y') and T^(b)_k(y,y').
  
  Parameters:
    
    ak: alpha_k = sqrt(2)*v_eq*k/k_eq, number.
      Here v_eq is the characteristic velocity at a_eq.
    
    y: a/a_eq, 1-D array of time steps.
    
    Tfs: The "free streaming transfer function", which is the Fourier transform
      of the velocity distribution in units of the characteristic velocity.
      
    iters: (optional, default 15) Depth of iterative evaluation, which sets how
      precise the result will be.
    
  Returns:
    
    T^(a)_k(y,y'): The "initial displacement" growth function.
      2-D array, y and y' are first and second axes, respectively.
        
    T^(b)_k(y,y'): The "initial velocity" growth function.
      2-D array, y and y' are first and second axes, respectively.
  
  '''
  
  F = np.log(y/(1.+np.sqrt(1.+y))**2)
  
  # prepare T^(a,0), T^(b,0)
  # output shape: y, y' (latest to earliest)
  Ta0 = np.tril(Tfs(ak*(F[:,None]-F[None])))
  Tb0 = np.tril((F[:,None]-F[None])*Tfs(ak*(F[:,None]-F[None])))
  np.fill_diagonal(Ta0,1.) # in case Tfs is not well behaved at 0
  np.fill_diagonal(Tb0,0.) # in case Tfs is not well behaved at 0
  
  # start iteration for T^(b)
  # integrand shape is y, y'', y' (latest to earliest)
  Tb = Tb0
  measure = y[None,:,None]/np.sqrt(1.+y[None,:,None])
  for i in range(iters-1):
    Tb = Tb0 + 1.5*simpson(Tb[:,:,None]*Tb0[None]*measure,x=np.log(y),axis=1)
    
  # evaluate T^(a)
  Ta = Ta0 + 1.5*simpson(Tb[:,:,None]*Ta0[None]*measure,x=np.log(y),axis=1)
  
  return Ta, Tb

def P_iso(ak,y,Tfs,iters=15):
  '''
  
  Evaluate n*P_iso(k), i.e. the white noise power spectrum in units of n^-1,
  where n is the mean number density.
  
  Parameters:
    
    ak: alpha_k = sqrt(2)*v_eq*k/k_eq, number.
      Here v_eq is the characteristic velocity at a_eq.
    
    y: a/a_eq, 1-D array of time steps. y[0] is when evolution starts.
    
    Tfs: The "free streaming transfer function", which is the Fourier transform
      of the velocity distribution in units of the characteristic velocity.
      
    iters: (optional, default 15) Depth of iterative evaluation, which sets how
      precise the result will be.
    
  Returns:
    
    n*P: 1-D array as a function of y.
  
  '''
  
  Ta, Tb = growth(ak,y,Tfs,iters=iters)
  
  # integrand shape: y, y' (latest to earliest)
  return 1. + 3.*simpson(Ta*Tb*y[None,:]/np.sqrt(1.+y[None,:]),x=np.log(y),axis=1)

def growth_ad(ak,y,Tfs,dlnDdlny0,iters=15):
  '''
  
  Evaluate T^ad_k(y,y_0), the adiabatic growth function.
  
  Parameters:
    
    ak: alpha_k = sqrt(2)*v_eq*k/k_eq, number.
      Here v_eq is the characteristic velocity at a_eq.
    
    y: a/a_eq, 1-D array of time steps. y[0] is when evolution starts.
    
    Tfs: The "free streaming transfer function", which is the Fourier transform
      of the velocity distribution in units of the characteristic velocity.
      
    dlnDdlny0: The value of d\ln\delta/d\ln a at the initial time, y=y[0].
      Number. Typically this should be 1/ln[sqrt(2) I_2 (k/k_eq) y]
      = 1/ln[I_2 (alpha_k/v_eq) y], with approximately I_2=0.47 as per
      Hu & Sugiyama (1996).
      
    iters: (optional, default 15) Depth of iterative evaluation, which sets how
      precise the result will be.
    
  Returns:
    
    T^ad_k(y,y_0): 1-D array as a function of y.
  
  '''
  
  Ta, Tb = growth(ak,y,Tfs,iters=iters)
  
  return Ta[:,0] + dlnDdlny0*np.sqrt(1.+y[0])*Tb[:,0]
