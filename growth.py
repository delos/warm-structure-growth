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
  integrand_facs = (y[:,None,None]>=y[None,:,None])*(y[None,:,None]>=y[None,None])*y[None,:,None]/np.sqrt(1.+y[None,:,None])
  for i in range(iters-1):
    Tb = Tb0 + 1.5*simpson(Tb0[:,:,None]*Tb[None]*integrand_facs,x=np.log(y),axis=1)
    
  # evaluate T^(a)
  Ta = Ta0 + 1.5*simpson(Tb[:,:,None]*Ta0[None]*integrand_facs,x=np.log(y),axis=1)
  
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
  return 1. + 3.*simpson(Ta*Tb*(y[:,None]>=y[None,:])*y[None,:]/np.sqrt(1.+y[None,:]),x=np.log(y),axis=1)

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

### EXAMPLE USAGE FOLLOWS ###

if __name__ == '__main__':
  
  import matplotlib.pyplot as plt
  
  # Cosmology parameters
  aeq = 0.000295
  keq = 0.01 # Mpc^-1
  
  # Characteristic velocity at matter-radiation equality, in units of c
  veq = 7.239e-5
  
  # Fourier transform of velocity distribution,
  # in units of the characteristic velocity
  Tfs = lambda x: np.exp(-0.5*x**2) # Maxwell-Boltzmann
  #Tfs = lambda x: 3/x**3 * (np.sin(x)-x*np.cos(x)) # uniform momentum sphere
  
  # Grid of times to evaluate at. These are also the time integration steps and
  # must be reasonably dense. The first time is when evolution starts.
  a = np.geomspace(1e-3*aeq,1e3*aeq,61)
  
  # List of k to evaluate at.
  k = np.geomspace(1,1e3,30)
  
  # Calculate isocurvature power spectrum.
  P = np.zeros((len(k),len(a)))
  for i,k_ in enumerate(k):
    P[i] = P_iso(np.sqrt(2)*veq*k_/keq,a/aeq,Tfs) # in units of n^-1
  
  # Plot isocurvature power spectrum.
  plt.figure()
  for i in [30,35,40,45,50,55,60]:
    plt.loglog(k,P[:,i],label='a=%g'%a[i])
  plt.legend()
  plt.xlabel('k (Mpc^-1)')
  plt.ylabel('n P(k)')
  plt.show()
  
  # Calculate adiabatic transfer function.
  T = np.zeros((len(k),len(a)))
  for i,k_ in enumerate(k):
    T[i] = growth_ad(np.sqrt(2)*veq*k_/keq,a/aeq,Tfs,1./np.log(np.sqrt(2)*0.47*k_/keq*a[0]/aeq))/growth_ad(0.,a/aeq,Tfs,1./np.log(np.sqrt(2)*0.47*k_/keq*a[0]/aeq))
  
  # Plot adiabatic transfer function.
  plt.figure()
  for i in [30,35,40,45,50,55,60]:
    plt.semilogx(k,T[:,i],label='a=%g'%a[i])
  plt.legend()
  plt.xlabel('k (Mpc^-1)')
  plt.ylabel('T(k)')
  plt.ylim(0,1)
  plt.show()
  
