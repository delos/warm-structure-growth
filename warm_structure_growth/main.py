import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import CubicSpline
from time import process_time
from . import iterative

def moment_f(n,f,maxf=30.):
  '''n-th velocity moment of f(v), integral |v|^n f(v) d^3v'''
  if callable(f):
    integrand = lambda v: v**(2+n)*f(v)
    vmax = maxf
  else:
    interpolation = CubicSpline(f[0],f[1],bc_type='clamped',extrapolate=True)
    integrand = lambda v: v**(2+n)*interpolation(v)
    vmax = f[0][-1]
  return 4*np.pi*quad(integrand,0.,vmax,)[0]

def fourier_f(x,f,maxf=30.):
  '''Fourier transform of f(v) evaluated at points x'''
  norm = moment_f(0,f,maxf=maxf)
  if callable(f):
    integrand = lambda v: v*f(v)
    vmax = maxf
  else:
    interpolation = CubicSpline(f[0],f[1],bc_type='clamped',extrapolate=True)
    integrand = lambda v: v*interpolation(v)
    vmax = f[0][-1]
  out = np.zeros_like(x)
  for i,x1 in enumerate(x):
    if x1 == 0:
      out[i] = 1.
    else:
      out[i] = 4*np.pi/norm*quad(integrand,0.,vmax,weight='sin',wvar=x1)[0] / x1
  return out

def moment_ff(n,u,f,maxf=30.):
  '''n-th velocity moment of f(|v+u/2|)f(|v-u/2|)'''
  if callable(f):
    integrand = lambda v,mu: v**(2+n)*f(np.sqrt(v**2+u**2/4+mu*v*u))*f(np.sqrt(v**2+u**2/4-mu*v*u))
    vmax = maxf
  else:
    interpolation = CubicSpline(f[0],f[1],bc_type='clamped',extrapolate=True)
    integrand = lambda v,mu: v**(2+n)*interpolation(np.sqrt(v**2+u**2/4+mu*v*u))*interpolation(np.sqrt(v**2+u**2/4-mu*v*u))
    vmax = f[0][-1]
  return 2*np.pi*quad(lambda mu: quad(integrand,0.,vmax,args=(mu,))[0],-1.,1.)[0]

def fourier_ff(x,u,f,maxf=30.):
  '''Fourier transform of f(|v+u/2|)f(|v-u/2|) evaluated at points x'''
  norm = moment_ff(0,u,f,maxf=maxf)
  if callable(f):
    integrand = lambda v,mu: v**2*f(np.sqrt(v**2+u**2/4+mu*v*u))*f(np.sqrt(v**2+u**2/4-mu*v*u))
    vmax = maxf
  else:
    interpolation = CubicSpline(f[0],f[1],bc_type='clamped',extrapolate=True)
    integrand = lambda v,mu: v**2*interpolation(np.sqrt(v**2+u**2/4+mu*v*u))*interpolation(np.sqrt(v**2+u**2/4-mu*v*u))
    vmax = f[0][-1]
  out = np.zeros_like(x)
  for i,x1 in enumerate(x):
    if x1 == 0:
      out[i] = 1.
    else:
      out[i] = 2*np.pi/norm*quad(lambda mu: quad(integrand,0.,vmax,args=(mu,),weight='cos',wvar=mu*x1)[0],-1.,1.)[0]
  return out

class Structure(object):
  '''
  Class for evaluating growth of structure given a velocity distribution and
  cosmological parameters.
  
  Parameters:
    
    a_i: float
      Scale factor at which free streaming starts.
    
    a_f: float
      Latest scale factor we will be interested in. Smaller values lead to
      faster evaluation. Will increase automatically as needed. Default is to
      set as needed.
      
    f: callable or tuple
      Velocity distribution, dN/d^3v. Does not need to be normalized. May be
      entered in two ways:
      - As a table (v,f), where v and f are 1-D arrays.
      - As a callable, f(v). In this case maxf must also be specified.
    
    maxf: float
      Maximum value of the argument of f that we need to consider. Only
      relevant if f is a callable function. Default is maxf=30.
    
    v_scale: float
      Rescale velocities so that the velocity distribution is f(v/v_scale).
      Default is v_scale=1. If f is a callable function and maxf is not
      specified, we assume that f is in terms of a dimensionless velocity, and
      then v_scale must be specified.
    
    v_at_init: bool
      If True, velocities are specified (via f and v_scale) at the initial time
      a_i. Otherwise they are specified at matter-radiation equality.
      Default is v_at_init=False.
    
    a_eq, k_eq: floats
      Scale factor and horizon wavenumber at matter-radiation equality.
      Defaults are a_eq=0.000295 and k_eq=0.01.
      
    max_FT: float
      We tabulate the Fourier transform of f between 1/[rms(v)*max_FT] and
      max_FT/rms(v). Default is max_FT=100. For a very broad velocity
      distribution, max_FT may need to be higher.
    
    N_ft: int
      Number of points on which to tabulate the Fourier transform of f. Default
      is N_ft = 1000.
    
    dlna: float
      Integration time step in ln(a). Default is dlna=0.23.
      
    iters: int
      Depth of iterative evaluation, which sets how precise the results will
      be. Default is iters=15.
      
    verbose: bool
      Default is True; set to False to disable messages.
  '''
  def __init__(self,a_i,a_f=None,f=None,maxf=None,v_scale=None,v_at_init=False,a_eq=0.000295,k_eq=0.01,max_FT=100.,N_ft=1000,dlna=0.23,iters=15,verbose=True):
    self.a_eq = a_eq
    self.k_eq = k_eq
    self.a_i = a_i
    self.a_f = a_f
    self.dlna = dlna
    self.iters = iters
    self.verbose = verbose
    
    if callable(f) and maxf is None and v_scale is None:
      raise Exception('with callable f, we must have at least one of maxf and v_scale')
    maxf = maxf or 30.
    v_scale = v_scale or 1.
    if v_at_init: # scale to a_eq
      v_scale *= a_i/a_eq
    
    self.__generate_FT(f,maxf,v_scale,max_FT,N_ft)
    
    self.__k = None
  
  def __generate_FT(self,f,maxf,v_scale,max_FT,N_ft):
    __t = process_time()
    self.sigma = np.sqrt(moment_f(2,f,maxf=maxf)/(3*moment_f(0,f,maxf=maxf)))
    self.__x = np.geomspace(1./max_FT,max_FT,N_ft)/self.sigma
    self.__T = fourier_f(self.__x,f,maxf=30.)
    
    if np.abs(self.__T[0]-1) > 1e-2 or np.abs(self.__T[-1]) > 1e-2:
      if self.verbose:
        print('Warning: Fourier transformed f(v) ranges from %g to %g. Try increasing max_FT=%g.'%(self.__T[-1],self.__T[0],max_FT))
    
    self.sigma *= v_scale
    self.__x /= v_scale
    
    self.__T_interp = CubicSpline(self.__x,self.__T,bc_type='clamped')
    if self.verbose:
      print('Fourier transformed f(v) in %.2f sec'%(process_time()-__t))
  
  def __generate_TaTb(self,k=None,a_f=None):
    __t = process_time()
    if a_f is not None:
      self.a_f = a_f
    if k is not None:
      self.__k = k
      self.__alpha = np.sqrt(2)*k/self.k_eq
    Na = int(np.round(np.log(self.a_f/self.a_i)/self.dlna))
    agrid = np.geomspace(self.a_i,self.a_f,Na)
    self.__y = agrid/self.a_eq
    # k, y, and y' are first, second, and third axes, respectively
    self.__Ta, self.__Tb = np.zeros((len(self.__k),Na,Na)), np.zeros((len(self.__k),Na,Na))
    for i,alpha in enumerate(self.__alpha):
      self.__Ta[i], self.__Tb[i] = iterative.growth_functions(alpha,self.__y,self.T,iters=self.iters)
    self.__Ta_interp = CubicSpline(self.__y,self.__Ta,axis=1)
    self.__Tb_interp = CubicSpline(self.__y,self.__Tb,axis=1)
    # also get k=0 evolution for transfer functions
    self.__Ta0, self.__Tb0 = iterative.growth_functions(0.,self.__y,self.T,iters=self.iters)
    self.__Ta0_interp = CubicSpline(self.__y,self.__Ta0,axis=0)
    self.__Tb0_interp = CubicSpline(self.__y,self.__Tb0,axis=0)
    if self.verbose:
      print('Evaluated growth functions in %.2f sec'%(process_time()-__t))
  
  def T(self,x):
    '''Fourier transform of the velocity distribution'''
    return np.piecewise(x,[x<self.__x[0],x>self.__x[-1]],[1.,0.,self.__T_interp])
    
  def TaTb(self,a,k=None):
    '''
    Evaluate the growth functions T^(a)_k(y,y') and T^(b)_k(y,y') up to the
    scale factor a.
    
    Parameters:
      
      a: float
        Scale factor at which to evaluate.
      
      k: array
        Wavenumbers in the same units as k_eq. If not specified, we attempt to
        use the last values (saves time).
      
    Returns:
      
      y: 1-D array
        Grid of a/a_eq for integrations
      
      Ta: 2-D array
        T^(a)_k(y,y'). k and y' are first and second axes, respectively.
      
      Tb: 2-D array
        T^(b)_k(y,y'). k and y' are first and second axes, respectively.
    '''
    if k is None or (self.__k is not None and len(k) == len(self.__k) and np.allclose(np.sqrt(2)*k/self.k_eq,self.__alpha,atol=0.)):
      k = None
    if a > self.a_f:
      self.__generate_TaTb(k,a_f=a)
    elif k is not None:
      self.__generate_TaTb(k)
    return self.__y, self.__Ta_interp(a/self.a_eq), self.__Tb_interp(a/self.a_eq)
  
  def TaTb0(self,a):
    '''
    Evaluate the growth functions T^(a)_k(y,y') and T^(b)_k(y,y') up to the
    scale factor a for a reference case with no velocity dispersion.
    
    Parameters:
      
      a: float
        Scale factor at which to evaluate.
      
    Returns:
      
      y: 1-D array
        Grid of a/a_eq for integrations
      
      Ta: 1-D array
        T^(a)_k(y,y') as a function of y'
      
      Tb: 1-D array
        T^(b)_k(y,y') as a function of y'
    '''
    if a > self.a_f:
      self.__generate_TaTb(np.zeros(1),a_f=a)
    return self.__y, self.__Ta0_interp(a/self.a_eq), self.__Tb0_interp(a/self.a_eq)
    
  def P_iso(self,a,k=None):
    '''
    Evaluate n*P_iso(k), i.e. the white noise power spectrum in units of n^-1,
    where n is the mean number density.
    
    Parameters:
  
      a: float
        Scale factor at which to evaluate.
      
      k: array
        Wavenumbers in the same units as k_eq. If not specified, we attempt to
        use the last values (saves time).
    
    Returns:
      
      n*P: array
        Power spectrum as a function of k.
    '''
    y, Ta, Tb = self.TaTb(k=k,a=a)
    return 1. + 3.*simpson(Ta*Tb*y[None,:]/np.sqrt(1.+y[None,:]),x=np.log(y),axis=1)
  
  def growth_ad(self,a,k=None,dlnDdlny0=None):
    '''
    Evaluate T^ad_k(y,y_0), the adiabatic growth function.
    
    Parameters:
  
      a: float
        Scale factor at which to evaluate.
      
      k: array
        Wavenumbers in the same units as k_eq. If not specified, we attempt to
        use the last values (saves time).
        
      dlnDdlny0: array
        The value of d\ln\delta/d\ln a at the initial time, y=y[0], as a
        function of k. Default 1/ln[sqrt(2) I_2 (k/k_eq) y], with I_2=0.47 as
        per Hu & Sugiyama (1996).
      
    Returns:
      
      T^ad_k(y,y_0): 1-D array as a function of k.
    '''
    if dlnDdlny0 is None:
      dlnDdlny0 = 1./np.log(np.sqrt(2)*0.47*(k if k is not None else self.__k)/self.k_eq*self.a_i/self.a_eq)
    y, Ta, Tb = self.TaTb(k=k,a=a)
    return Ta[:,0] + dlnDdlny0*np.sqrt(1.+y[0])*Tb[:,0]
  
  def cutoff_ad(self,a,k=None,dlnDdlny0=None):
    '''
    Evaluate the free-streaming cutoff transfer function for the adiabatic
    modes, i.e. the ratio between density perturbations with the velocity
    distribution to without.
    
    Parameters:
  
      a: float
        Scale factor at which to evaluate.
      
      k: array
        Wavenumbers in the same units as k_eq. If not specified, we attempt to
        use the last values (saves time).
        
      dlnDdlny0: array
        The value of d\ln\delta/d\ln a at the initial time, y=y[0], as a
        function of k. Default 1/ln[sqrt(2) I_2 (k/k_eq) y], with I_2=0.47 as
        per Hu & Sugiyama (1996).
      
    Returns:
      
      T^ad_k(y,y_0): 1-D array as a function of y.
    '''
    if dlnDdlny0 is None:
      dlnDdlny0 = 1./np.log(np.sqrt(2)*0.47*(k if k is not None else self.__k)/self.k_eq*self.a_i/self.a_eq)
    y, Ta, Tb = self.TaTb(k=k,a=a)
    y, Ta0, Tb0 = self.TaTb0(a=a)
    return (Ta[:,0] + dlnDdlny0*np.sqrt(1.+y[0])*Tb[:,0])/(Ta0[0] + dlnDdlny0*np.sqrt(1.+y[0])*Tb0[0])
