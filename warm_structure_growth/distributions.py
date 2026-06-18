r'''
Analytic velocity/momentum distributions for warm wave (and particle) dark matter.

Each named distribution provides four functions of the (isotropic) velocity
distribution f(v):

  moment_f(n)    : the velocity moment  integral |v|^n f(v) d^3v.
                   Only used to set the dispersion sigma; computed with an
                   amplitude-1 f, so its overall normalization is irrelevant.
  fourier_f(x)   : the 3-D Fourier transform of f, normalized to 1 at x=0,
                   i.e. ftilde(x)/ftilde(0).
  fourier_ff(x,u): the 3-D Fourier transform of the self-convolution
                   f(|v+u/2|) f(|v-u/2|), normalized to 1 at x=0. The transform
                   is taken with the wavevector x PARALLEL to the separation u.
  norm_ff(u)     : the isocurvature power spectrum P^iso(u), i.e. the
                   (un-normalized) self-convolution integral
                   integral f0(|v+u/2|) f0(|v-u/2|) d^3v of the PHYSICALLY
                   normalized field spectrum f0, defined by
                   integral d^3p/(2pi)^3 f0(p) = 1. (This is why, e.g.,
                   norm_ff_uniform(0)=6 pi^2 and norm_ff_Maxwell(0)=pi^1.5 sit on
                   different amplitude footings from moment_f -- both encode the
                   physically normalized f0; see Amin, May & Mirbabayi 2025,
                   arXiv:2506.12131, and Amin & Delos 2025, arXiv:2510.17977.)

Distributions are registered in `named_distributions` after being rescaled to
UNIT VARIANCE (intrinsic sigma = 1) by `_unit_variance`, so that the `v_scale`
argument of `Structure` equals the physical velocity dispersion sigma for every
shape. The functions defined below are the un-rescaled "base" shapes
(e.g. uniform = top-hat of radius 1); the wrapper handles the rescaling.
'''
import numpy as np
from scipy.special import gamma, exp1
from numpy.polynomial.laguerre import laggauss

# ============================ Maxwell-Boltzmann ============================
# base shape f(v) = exp(-v^2/2)

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

# ============================== uniform sphere =============================
# base shape f(v) = Theta(1-v)  (top hat of radius 1)

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
  small,large = (u<2.)&(x<0.1), (u<2.)&(x>=0.1)
  out[small] = __fourier_ff_uniform_smallx(x[small],u[small])
  out[large] = __fourier_ff_uniform_largex(x[large],u[large])
  return out

def moment_f_uniform(n):
  '''n-th velocity moment of uniform-sphere distribution. Specifically this is
  integral |v|^n f(v) d^3v.'''
  return 4.*np.pi/(3.+n)

def norm_ff_uniform(u):
  '''Integral over self-convolution of uniform-sphere distribution.'''
  return 6.*np.pi**2 * (0.25*u+1)*(0.5*u-1)**2 * (u<2.)

# =============================== exponential ===============================
# base shape f(v) = exp(-v)

def fourier_f_exponential(x):
  '''Fourier transform of exponential distribution f(v)=exp(-v).'''
  return 1./(1.+x**2)**2

def moment_f_exponential(n):
  '''n-th velocity moment of exponential distribution, integral |v|^n f(v) d^3v.'''
  return 4.*np.pi*gamma(n+3.)

def norm_ff_exponential(u):
  '''Integral over self-convolution of exponential distribution.'''
  return np.pi**2/24. * np.exp(-u) * (u**2+3.*u+3.)

def __fourier_ff_exponential_smallx(x,u):
  d = u**2+3.*u+3.
  c1 = -(u**4+7.*u**3+27.*u**2+60.*u+60.)/(40.*d)
  c2 = (u**6+11.*u**5+75.*u**4+360.*u**3+1200.*u**2+2520.*u+2520.)/(4480.*d)
  c3 = -(u**8+15.*u**7+147.*u**6+1092.*u**5+6300.*u**4+27720.*u**3+88200.*u**2+181440.*u+181440.)/(967680.*d)
  c4 = (u**10+19.*u**9+243.*u**8+2448.*u**7+20160.*u**6+136080.*u**5+740880.*u**4+3144960.*u**3+9797760.*u**2+19958400.*u+19958400.)/(340623360.*d)
  c5 = -(u**12+23.*u**11+363.*u**10+4620.*u**9+49500.*u**8+451440.*u**7+3492720.*u**6+22619520.*u**5+119750400.*u**4+498960000.*u**3+1536796800.*u**2+3113510400.*u+3113510400.)/(177124147200.*d)
  c6 = (u**14+27.*u**13+507.*u**12+7800.*u**11+102960.*u**10+1184040.*u**9+11891880.*u**8+103783680.*u**7+778377600.*u**6+4929724800.*u**5+25686460800.*u**4+105859353600.*u**3+323805081600.*u**2+653837184000.*u+653837184000.)/(127529385984000.*d)
  x2 = x**2
  return 1. + x2*(c1 + x2*(c2 + x2*(c3 + x2*(c4 + x2*(c5 + x2*c6)))))
def __fourier_ff_exponential_largex(x,u):
  # closed form via prolate-spheroidal coordinates; E1 = exponential integral
  p = x*u/2.
  w = u - 1j*p
  E1 = exp1(w)
  ew = np.exp(-w)
  J1 = np.imag(ew*(w+1.)/w**2)
  G = 2.*np.pi*(u/2.)**3 * ( (2./p)*J1 - (2./p)*np.imag(E1)
        - (4./p**2)*np.real(ew - w*E1)
        + (4./p**3)*np.imag(0.5*ew - 0.5*w*ew + 0.5*w**2*E1) )
  return G / (np.pi/3.*np.exp(-u)*(u**2+3.*u+3.))
def fourier_ff_exponential(x,u):
  '''Fourier transform of self-convolution of exponential distribution.
  Specifically we Fourier transform f(|v+u/2|)f(|v-u/2|).'''
  x,u = np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(u,dtype=float))
  out = np.empty(x.shape)
  small = x < 0.7
  out[small] = __fourier_ff_exponential_smallx(x[small],u[small])
  out[~small] = __fourier_ff_exponential_largex(x[~small],u[~small])
  return out

# ========================= parabolic (Epanechnikov) ========================
# base shape f(v) = (1-v^2) Theta(1-v)

def __fourier_f_parabolic_largex(x):
  return 15.*(3.*np.sin(x)-3.*x*np.cos(x)-x**2*np.sin(x))/x**5
def __fourier_f_parabolic_smallx(x):
  return 1 - x**2/14. + x**4/504. - x**6/33264. + x**8/3459456. - x**10/518918400.
def fourier_f_parabolic(x):
  '''Fourier transform of parabolic (Epanechnikov) distribution f(v)=(1-v^2).'''
  return np.piecewise(x,[x<0.5],[__fourier_f_parabolic_smallx,__fourier_f_parabolic_largex])

def moment_f_parabolic(n):
  '''n-th velocity moment of parabolic distribution, integral |v|^n f(v) d^3v.'''
  return 8.*np.pi/((n+3.)*(n+5.))

def norm_ff_parabolic(u):
  '''Integral over self-convolution of parabolic distribution.'''
  return 15.*np.pi**2/896. * (2.-u)**4 * (3.*u**3+24.*u**2+64.*u+32.) * (u<2.)

def __fourier_ff_parabolic_smallx(x,u):
  D = 3.*u**3+24.*u**2+64.*u+32.
  c2 = (-u**5-8.*u**4-4.*u**3+128.*u**2-128.*u-64.)/(36.*D)
  c4 = (3.*u**7+24.*u**6-56.*u**5-928.*u**4+4144.*u**3-5888.*u**2+2048.*u+1024.)/(25344.*D)
  c6 = (-3.*u**9-24.*u**8+140.*u**7+1600.*u**6-14160.*u**5+45376.*u**4-71872.*u**3+53760.*u**2-10240.*u-5120.)/(9884160.*D)
  c8 = (u**11+8.*u**10-80.*u**9-800.*u**8+11360.*u**7-57344.*u**6+159488.*u**5-266240.*u**4+262400.*u**3-133120.*u**2+16384.*u+8192.)/(1897758720.*D)
  x2 = x**2
  return 1. + x2*(c2 + x2*(c4 + x2*(c6 + x2*c8)))
def __fourier_ff_parabolic_largex(x,u):
  t = x*(u-2.)/2.
  s = np.sin(t); c = np.cos(t)
  num = (u**3*x**3 - u**2*x**4*s + 3.*u**2*x**3*c + 3.*u**2*x**2*s
         + 2.*u*x**4*s - 12.*u*x**3*c - 30.*u*x**2*s + 30.*u*x*c
         + 4.*x**3*c + 24.*x**2*s - 60.*x*c - 60.*s)
  den = x**7*(3.*u**7-56.*u**5+560.*u**3-896.*u**2+512.)
  return 13440.*num/den
def fourier_ff_parabolic(x,u):
  '''Fourier transform of self-convolution of parabolic distribution.
  Specifically we Fourier transform f(|v+u/2|)f(|v-u/2|).'''
  x,u = np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(u,dtype=float))
  out = np.zeros(x.shape)
  small,large = (u<2.)&(x<0.5), (u<2.)&(x>=0.5)
  out[small] = __fourier_ff_parabolic_smallx(x[small],u[small])
  out[large] = __fourier_ff_parabolic_largex(x[large],u[large])
  return out

# ============================ power-law tails ==============================
# base shape f(v) = (1+v^2)^-3  (tail ~ v^-6, finite variance)

def fourier_f_powerlaw(x):
  '''Fourier transform of power-law distribution f(v)=(1+v^2)^-3.'''
  return (1.+x)*np.exp(-x)

def moment_f_powerlaw(n):
  '''n-th velocity moment of power-law distribution, integral |v|^n f(v) d^3v
  (finite for n<3).'''
  return np.pi*gamma((n+3.)/2.)*gamma((3.-n)/2.)

def norm_ff_powerlaw(u):
  '''Integral over self-convolution of power-law distribution.'''
  return 64.*np.pi*(u**2+28.)/(u**2+4.)**4

# Gauss-Laguerre nodes for the large-x branch of fourier_ff (no closed form).
__pl_gl_x,__pl_gl_w = laggauss(64)
def __pl_psi_over_pi(s,u,x):
  # = NUM/den, where the self-convolution FT integrand is pi*(NUM/den)*exp(-s*x);
  # s runs over [1,inf), the radial prolate-spheroidal coordinate.
  c = np.cos(u*x/2.); si = np.sin(u*x/2.)
  NUM = (-128.*s**9*u**2*x**2*si - 768.*s**9*u*x*c + 1536.*s**9*si
         - 192.*s**8*u**3*x**2*c + 384.*s**8*u**2*x*si + 32.*s**7*u**4*x**2*si
         - 1152.*s**7*u**3*x*c + 2304.*s**7*u**2*si - 80.*s**6*u**5*x**2*c
         + 1056.*s**6*u**4*x*si + 40.*s**5*u**6*x**2*si + 2016.*s**5*u**4*si
         - 4.*s**4*u**7*x**2*c + 264.*s**4*u**6*x*si + 1008.*s**4*u**5*c
         + 6.*s**3*u**8*x**2*si + 72.*s**3*u**7*x*c + s**2*u**9*x**2*c
         + 6.*s**2*u**8*x*si + 72.*s**2*u**7*c + 3.*s*u**9*x*c + 3.*u**9*c)
  den = 4.*s**5*u**5*(1024.*s**10+1280.*s**8*u**2+640.*s**6*u**4+160.*s**4*u**6+20.*s**2*u**8+u**10)
  return NUM/den
def __fourier_ff_powerlaw_smallx(x,u):
  c1 = (-u**4/8.-u**2-2.)/(u**2+28.)
  c2 = (u*(u**10+4.*u**8-96.*u**6-896.*u**4-2816.*u**2-3072.)+24.*(u**2+4.)**4*np.arctan(u/2.))/(384.*u**5*(u**2+28.))
  c3 = -(u**2+4.)**4/(46080.*(u**2+28.))
  c4 = (u**2+4.)**4/10321920.
  x2 = x**2
  return 1. + x2*(c1 + x2*(c2 + x2*(c3 + x2*c4)))
def __fourier_ff_powerlaw_largex(x,u):
  # G(x,u) = integral_1^inf 2 pi s pi (NUM/den) exp(-s x) ds, via s = 1 + w/x and
  # Gauss-Laguerre quadrature; normalized by the raw self-convolution C(u).
  s = 1. + __pl_gl_x[:,None]/x[None,:]
  integ = 2.*np.pi*s*np.pi*__pl_psi_over_pi(s,u[None,:],x[None,:])
  G = np.exp(-x)/x*np.sum(__pl_gl_w[:,None]*integ,axis=0)
  C = np.pi**2*(u**2+28.)/(2.*(u**2+4.)**4)
  return G/C
def fourier_ff_powerlaw(x,u):
  '''Fourier transform of self-convolution of power-law distribution.
  Specifically we Fourier transform f(|v+u/2|)f(|v-u/2|).'''
  x,u = np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(u,dtype=float))
  shape = x.shape
  x,u = x.ravel(),u.ravel()
  out = np.empty(x.shape)
  small = x < 0.6
  out[small] = __fourier_ff_powerlaw_smallx(x[small],u[small])
  out[~small] = __fourier_ff_powerlaw_largex(x[~small],u[~small])
  return out.reshape(shape)

# =============================== registration ==============================

def _unit_variance(base):
  '''Wrap a "base" distribution (dict of moment_f/fourier_f/fourier_ff/norm_ff)
  so that the registered distribution has unit variance (sigma=1), i.e. v_scale
  equals the physical velocity dispersion. The base shape g(v) is rescaled to
  g(s*v) with s = sqrt(moment_f(2)/(3 moment_f(0))) its intrinsic dispersion.'''
  s = np.sqrt(base['moment_f'](2)/(3.*base['moment_f'](0)))
  return {
    'moment_f':   lambda n,   b=base,s=s: s**(-(n+3.))*b['moment_f'](n),
    'fourier_f':  lambda x,   b=base,s=s: b['fourier_f'](np.asarray(x)/s),
    'fourier_ff': lambda x,u, b=base,s=s: b['fourier_ff'](np.asarray(x)/s, s*np.asarray(u)),
    'norm_ff':    lambda u,   b=base,s=s: s**3*b['norm_ff'](s*np.asarray(u)),
  }

named_distributions = {name: _unit_variance(base) for name,base in {
  'maxwell':    {'moment_f':moment_f_Maxwell,    'fourier_f':fourier_f_Maxwell,    'fourier_ff':fourier_ff_Maxwell,    'norm_ff':norm_ff_Maxwell,},
  'uniform':    {'moment_f':moment_f_uniform,    'fourier_f':fourier_f_uniform,    'fourier_ff':fourier_ff_uniform,    'norm_ff':norm_ff_uniform,},
  'exponential':{'moment_f':moment_f_exponential,'fourier_f':fourier_f_exponential,'fourier_ff':fourier_ff_exponential,'norm_ff':norm_ff_exponential,},
  'parabolic':  {'moment_f':moment_f_parabolic,  'fourier_f':fourier_f_parabolic,  'fourier_ff':fourier_ff_parabolic,  'norm_ff':norm_ff_parabolic,},
  'powerlaw':   {'moment_f':moment_f_powerlaw,   'fourier_f':fourier_f_powerlaw,   'fourier_ff':fourier_ff_powerlaw,   'norm_ff':norm_ff_powerlaw,},
  }.items()}
