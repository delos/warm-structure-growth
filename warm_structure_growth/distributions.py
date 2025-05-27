import numpy as np
from scipy.special import gamma

def fourier_f_Maxwell(x):
  '''Fourier transform of Maxwell-Boltzmann distribution.'''
  return np.exp(-0.5*x**2)

def fourier_ff_Maxwell(x,u):
  '''Fourier transform of self-convolution of Maxwell-Boltzmann distribution.
  Specifically we Fourier transform f(|v+u/2|)f(|v-u/2|).'''
  return np.exp(-0.25*x**2)

def moment_f_Maxwell(n):
  '''n-th velocity moment of Maxwell-Boltzmann distribution. Specifically this
  is integral |v|^n f(v) d^3v.'''
  return 2.**(0.5*(5+n))*np.pi*gamma(0.5*(3+n))

def norm_ff_Maxwell(u):
  '''Integral over self-convolution of Maxwell-Boltzmann distribution.'''
  return np.pi**1.5 * np.exp(-0.25*u**2)

def __fourier_f_uniform_largex(x):
  return 3./x**3 * (np.sin(x)-x*np.cos(x))
def __fourier_f_uniform_smallx(x):
  return 1 - x**2/10. + x**4/280. - x**6/15120. + x**8/1330560. - x**10/172972800.
def fourier_f_uniform(x):
  '''Fourier transform of uniform-sphere distribution.'''
  return np.piecewise(x,[x<0.1],[__fourier_f_uniform_smallx,__fourier_f_uniform_largex])

def __fourier_ff_uniform_smallx(x,u):
  y = (2.-u)*x
  return 1 - (8 + u)*y**2/(80.*(4 + u)) + (12 + u)*y**4/(13440.*(4 + u)) - (16 + u)*y**6/(3.87072e6*(4 + u))
def __fourier_ff_uniform_largex(x,u):
  return 3./x**3 * ( np.sin((1-0.5*u)*x) + x*( 0.5*u - np.cos((1-0.5*u)*x) ) ) / ( (0.25*u+1)*(0.5*u-1)**2 )
def fourier_ff_uniform(x,u):
  '''Fourier transform of self-convolution of uniform-sphere distribution.
  Specifically we Fourier transform f(|v+u/2|)f(|v-u/2|).'''
  x,u = np.broadcast_arrays(x,u)
  out = np.zeros(x.shape)
  small,large = (u<2.)&(x<0.1), (u<2.)&(x>0.1)
  out[small] = __fourier_ff_uniform_smallx(x[small],u[small])
  out[large] = __fourier_ff_uniform_smallx(x[large],u[large])
  return out

def moment_f_uniform(n):
  '''n-th velocity moment of uniform-sphere distribution. Specifically this is
  integral |v|^n f(v) d^3v.'''
  return 4.*np.pi/(3.+n)

def norm_ff_uniform(u):
  '''Integral over self-convolution of uniform-sphere distribution.'''
  return 6.*np.pi**2 * (0.25*u+1)*(0.5*u-1)**2 * (u<2.)

named_distributions = {
  'maxwell':{'moment_f':moment_f_Maxwell,'fourier_f':fourier_f_Maxwell,'fourier_ff':fourier_ff_Maxwell,'norm_ff':norm_ff_Maxwell,},
  'uniform':{'moment_f':moment_f_uniform,'fourier_f':fourier_f_uniform,'fourier_ff':fourier_ff_uniform,'norm_ff':norm_ff_uniform,},
  }