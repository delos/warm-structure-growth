import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import CubicSpline
from time import process_time
from .iterative import growth_functions
from .distributions import named_distributions

def __sinc_smallx(x):
  return 1. - x**2/6. + x**4/120. - x**6/5040. + x**8/362880.
def __sinc_largex(x):
  return np.sin(x)/x
def sinc(x):
  '''sin(x)/x'''
  return np.piecewise(x,[x<0.1],[__sinc_smallx,__sinc_largex])

def moment_f(n,f):
  '''n-th velocity moment of f(v), integral |v|^n f(v) d^3v'''
  if callable(f):
    integrand = lambda v: v**(2+n)*f(v)
    vmax = np.inf
  else:
    interpolation = CubicSpline(f[0],f[1],bc_type='clamped',extrapolate=True)
    integrand = lambda v: v**(2+n)*interpolation(v)
    vmax = f[0][-1]
  return 4*np.pi*quad(integrand,0.,vmax,)[0]

def fourier_f(x,f):
  '''Fourier transform of f(v) evaluated at points x'''
  norm = moment_f(0,f)
  if callable(f):
    integrand = lambda v: v*f(v)
    vmax = np.inf
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

def moment_ff(n,u,f):
  '''n-th velocity moment of f(|v+u/2|)f(|v-u/2|)'''
  if callable(f):
    integrand = lambda v,mu: v**(2+n)*f(np.sqrt(v**2+u**2/4+mu*v*u))*f(np.sqrt(v**2+u**2/4-mu*v*u))
    vmax = np.inf
  else:
    interpolation = CubicSpline(f[0],f[1],bc_type='clamped',extrapolate=True)
    integrand = lambda v,mu: v**(2+n)*interpolation(np.sqrt(v**2+u**2/4+mu*v*u))*interpolation(np.sqrt(v**2+u**2/4-mu*v*u))
    vmax = f[0][-1]
  return 2*np.pi*quad(lambda mu: quad(integrand,0.,vmax,args=(mu,))[0],-1.,1.)[0]

def fourier_ff(x,u,f):
  '''Fourier transform of f(|v+u/2|)f(|v-u/2|) evaluated at points x'''
  norm = moment_ff(0,u,f)
  if callable(f):
    integrand = lambda v,mu: v**2*f(np.sqrt(v**2+u**2/4+mu*v*u))*f(np.sqrt(v**2+u**2/4-mu*v*u))
    vmax = np.inf
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
  Class for evaluating growth of perturbations given a velocity distribution
  and cosmological parameters.
  
  Parameters:
    
    a_i: float
      Scale factor at which free streaming starts.
    
    a_f: float
      Latest scale factor we will be interested in. Smaller values lead to
      faster evaluation. Will increase automatically as needed. Default is to
      set as needed.
      
    f: callable or tuple
      Velocity distribution, dN/d^3v. Does not need to be normalized. May be
      entered in three ways:
      - As a table (v,f), where v and f are 1-D arrays.
      - As a callable, f(v).
      - As a string corresponding to a supported named distribution. Supported
        distributions are "Maxwell" (Maxwell-Boltzmann) and "uniform" (uniform
        sphere).
    
    v_scale: float
      Rescale velocities so that the velocity distribution is f(v/v_scale).
      Default is v_scale=1.
    
    v_at_init: bool
      If True, velocities are specified (via f and v_scale) at the initial time
      a_i. Otherwise they are specified at matter-radiation equality.
      Default is v_at_init=False.
      
    m: float
      Field mass in the same units as k_eq (typically Mpc^-1). If m and k_scale
      are both not specified (or None), we use the particle white-noise
      calculation instead.
    
    k_scale: float
      As an alternative to specifying the mass, it is possible to specify the
      characteristic comoving momentum. Note that k_scale = a_eq * m * v_scale.
      If m and k_scale are both not specified (or None), we use the particle
      white-noise calculation instead. If both m and k_scale are given, then
      v_scale (and v_at_init) will be ignored.
    
    n: float
      Number density of particles. Only relevant if m and k_scale are not
      specified. Default is n=1, so that the isocurvature power spectrum is in
      units of 1/n.
    
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
  def __init__(self,a_i,a_f=None,f=None,v_scale=1.,v_at_init=False,m=None,k_scale=None,n=1.,a_eq=0.000295,k_eq=0.01,max_FT=100.,N_ft=1000,dlna=0.23,iters=15,verbose=True):
    self.a_eq = a_eq
    self.k_eq = k_eq
    self.a_i = a_i
    self.a_f = a_f
    self.dlna = dlna
    self.iters = iters
    self.verbose = verbose
    
    self.v_scale = v_scale
    if v_at_init: # scale to a_eq
      self.v_scale *= a_i/a_eq
    
    if m is None and k_scale is None:
      self.m = None
      self.k_scale = None
      self.__generate_FT_particle(f,max_FT,N_ft)
    else:
      self.m = m or k_scale / (self.a_eq * self.v_scale)
      self.k_scale = k_scale or self.a_eq * self.m * self.v_scale
      if m is not None and k_scale is not None:
        self.v_scale = self.k_scale / (self.a_eq * self.m)
      self.__generate_FT_field(f,max_FT,N_ft)
    
    self.n = n
    
    self.__k = None
  
  def __generate_FT_particle(self,f,max_FT,N_ft):
    if isinstance(f,str) and f.lower() in named_distributions:
      f = f.lower()
      self.sigma = np.sqrt(named_distributions[f]['moment_f'](2)/(3*named_distributions[f]['moment_f'](0)))
      self.__Tfs = lambda x: named_distributions[f]['fourier_f'](x*self.v_scale)
      self.sigma *= self.v_scale
      if self.verbose:
        print('Using built-in f(v): %s (sigma_eq=%.3e)'%(f,self.sigma))
    else:
      __t = process_time()
      
      norm_f = moment_f(0,f)
      if (not np.isfinite(norm_f)) or norm_f == 0.:
        raise Exception('Custom f(v) integrates to %g and cannot be normalized.'%norm_f)
      self.sigma = np.sqrt(moment_f(2,f)/(3*norm_f))
      if (not np.isfinite(self.sigma)) or self.sigma == 0.:
        raise Exception('Custom f(v) gives rise to sigma=%g.'%self.sigma)
      
      self.__x = np.geomspace(1./max_FT,max_FT,N_ft)/self.sigma
      self.__Tfs = fourier_f(self.__x,f)
      
      if np.abs(self.__Tfs[0]-1) > 1e-2 or np.abs(self.__Tfs[-1]) > 1e-2:
        print('Warning: Fourier transformed f(v) ranges from %g to %g (should be 0 to 1). Try increasing max_FT=%g.'%(self.__Tfs[-1],self.__Tfs[0],max_FT))
      
      self.sigma *= self.v_scale
      self.__x /= self.v_scale
      
      self.__Tfs_interp = CubicSpline(self.__x,self.__Tfs,bc_type='clamped')
      if self.verbose:
        print('Using custom f(v) (sigma_eq=%.3e); Fourier transformed in %.2f sec'%(self.sigma,process_time()-__t))
      
  def __generate_FT_field(self,f,max_FT,N_ft):
    if isinstance(f,str) and f.lower() in named_distributions:
      f = f.lower()
      self.sigma = np.sqrt(named_distributions[f]['moment_f'](2)/(3*named_distributions[f]['moment_f'](0)))
      self.__Tfs = lambda x: named_distributions[f]['fourier_f'](x*self.v_scale)
      self.__Tfs2 = lambda x,u: named_distributions[f]['fourier_ff'](x*self.v_scale,u/self.k_scale)
      self.__P_iso = lambda u: named_distributions[f]['norm_ff'](u/self.k_scale)
      self.sigma *= self.v_scale
      if self.verbose:
        print('Using built-in f(v): %s (sigma_eq=%.3e)'%(f,self.sigma))
    else:
      raise NotImplementedError('custom f(v) currently not supported for fields')
  
  def __generate_growth(self,k=None,a_f=None):
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
    self.__Ta, self.__Tb, self.__Tc = [np.zeros((len(self.__k),Na,Na)) for i in range(3)]
    for i,alpha in enumerate(self.__alpha):
      if self.k_scale is not None: # field
        g = 0.5 * self.__k[i] / self.k_scale * self.v_scale
        Tfs = [lambda x: np.cos(g*x)*self.Tfs(x),lambda x: sinc(g*x)*self.Tfs(x),lambda x: self.Tfs2(x,self.__k[i])]
        self.__Ta[i], self.__Tb[i], self.__Tc[i] = growth_functions(alpha,self.__y,Tfs,iters=self.iters)
      else: # particle
        self.__Ta[i], self.__Tb[i] = growth_functions(alpha,self.__y,self.Tfs,iters=self.iters)
        self.__Tc[i] = self.__Ta[i]
    self.__Ta_interp = CubicSpline(self.__y,self.__Ta,axis=1)
    self.__Tb_interp = CubicSpline(self.__y,self.__Tb,axis=1)
    self.__Tc_interp = CubicSpline(self.__y,self.__Tc,axis=1)
    # also get k=0 evolution for transfer functions
    self.__Ta0, self.__Tb0 = growth_functions(0.,self.__y,lambda x: x**0,iters=self.iters)
    self.__Ta0_interp = self.__Tc0_interp = CubicSpline(self.__y,self.__Ta0,axis=0)
    self.__Tb0_interp = CubicSpline(self.__y,self.__Tb0,axis=0)
    if self.verbose:
      print('Evaluated growth functions in %.2f sec'%(process_time()-__t))
  
  def Tfs(self,x):
    '''Fourier transform of the velocity distribution'''
    if callable(self.__Tfs):
      return self.__Tfs(x)
    return np.piecewise(x,[x<self.__x[0],x>self.__x[-1]],[1.,0.,self.__Tfs_interp])
  
  def Tfs2(self,x,u):
    '''Fourier transform of the self-convolution of the velocity distribution'''
    if callable(self.__Tfs2):
      return self.__Tfs2(x,u)
    raise NotImplementedError('self-convolution of custom f(v) currently not supported')
    
  def growth(self,a,k=None):
    '''
    Evaluate the growth functions T^(a,b,c)_k(y,y') up to the scale factor a.
    
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
      
      Tc: 2-D array
        T^(c)_k(y,y'). k and y' are first and second axes, respectively.
    '''
    if k is None or (self.__k is not None and len(k) == len(self.__k) and np.allclose(np.sqrt(2)*k/self.k_eq,self.__alpha,atol=0.)):
      k = None
    if self.a_f is None or a > self.a_f:
      self.__generate_growth(k,a_f=a)
    elif k is not None:
      self.__generate_growth(k)
    return self.__y, self.__Ta_interp(a/self.a_eq), self.__Tb_interp(a/self.a_eq), self.__Tc_interp(a/self.a_eq)
  
  def growth0(self,a):
    '''
    Evaluate the growth functions T^(a,b,c)_k(y,y') up to the scale factor a
    for a reference case with no velocity dispersion.
    
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
      
      Tc: 1-D array
        T^(c)_k(y,y') as a function of y'
    '''
    if self.a_f is None or a > self.a_f:
      self.__generate_growth(np.zeros(1),a_f=a)
    return self.__y, self.__Ta0_interp(a/self.a_eq), self.__Tb0_interp(a/self.a_eq), self.__Tc0_interp(a/self.a_eq)
  
  def k_J(self,a):
    '''Jeans wavenumber at scale factor a during matter domination.'''
    return np.sqrt(3*a/self.a_eq)/2. * self.k_eq / self.sigma
  
  def P0_iso(self,k):
    '''
    Evaluate the white noise power spectrum prior to any gravitational growth.
    
    Parameters:
      
      k: array
        Wavenumbers in the same units as k_eq.
    
    Returns:
      
      P: array
        (Dimensionful) power spectrum as a function of k.
    '''
    if self.k_scale is None:
      return 1./self.n
    else:
      return self.k_scale**-3 * self.__P_iso(k)
  
  def P_iso(self,a,k=None):
    '''
    Evaluate P_iso(k), the white noise power spectrum.
    
    Parameters:
  
      a: float
        Scale factor at which to evaluate.
      
      k: array
        Wavenumbers in the same units as k_eq. If not specified, we attempt to
        use the last values (saves time).
    
    Returns:
      
      P: array
        (Dimensionful) power spectrum as a function of k.
    '''
    y, Ta, Tb, Tc = self.growth(k=k,a=a)
    return self.P0_iso(self.__k) * (1. + 3.*simpson(Tc*Tb*y[None,:]/np.sqrt(1.+y[None,:]),x=np.log(y),axis=1))
  
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
    y, Ta, Tb, Tc = self.growth(k=k,a=a)
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
    y, Ta, Tb, Tc = self.growth(k=k,a=a)
    y, Ta0, Tb0, Tc0 = self.growth0(a=a)
    return (Ta[:,0] + dlnDdlny0*np.sqrt(1.+y[0])*Tb[:,0])/(Ta0[0] + dlnDdlny0*np.sqrt(1.+y[0])*Tb0[0])
