import numpy as np
import iterative
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
  P[i] = iterative.P_iso(np.sqrt(2)*veq*k_/keq,a/aeq,Tfs) # in units of n^-1

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
  T[i] = iterative.growth_ad(np.sqrt(2)*veq*k_/keq,a/aeq,Tfs,1./np.log(np.sqrt(2)*0.47*k_/keq*a[0]/aeq))/iterative.growth_ad(0.,a/aeq,Tfs,1./np.log(np.sqrt(2)*0.47*k_/keq*a[0]/aeq))

# Plot adiabatic transfer function.
plt.figure()
for i in [30,35,40,45,50,55,60]:
  plt.semilogx(k,T[:,i],label='a=%g'%a[i])
plt.legend()
plt.xlabel('k (Mpc^-1)')
plt.ylabel('T(k)')
plt.ylim(0,1)
plt.show()

