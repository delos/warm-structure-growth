# warm-structure-growth
Code for evaluating growth functions and matter power spectra in cosmology, including the effects of potentially finite velocity dispersion and number density in the dark matter.

## Content:

The file [iterative.py](warm_structure_growth/iterative.py) contains the iterative algorithms described in Appendix B of [Amin et al. (2025)](https://arxiv.org/abs/2503.20881). The code in [main.py](warm_structure_growth/main.py) is for convenience.

## Usage:
In Python, start with:

```
import sys
sys.path.append("path/to/warm-structure-growth")
import warm_structure_growth
```

Now pick some parameters, for example:

```
f = lambda v: np.exp(-v**2/2) # Maxwell-Boltzmann velocity distribution
v_eq = 7e-5 # scale velocity at matter-radiation equality, in units of c
a_init = 3e-7 # scale factor at which we start nonrelativistic free streaming
structure = warm_structure_growth.Structure(a_i=a_init,f=f,v_scale=v_eq)
```

Note that the velocity distribution can also be specified as a table. See the docstring (`help(warm_structure_growth.Structure` or in [main.py](warm_structure_growth/main.py)) for this and other options. Now we can evaluate the transfer function for adiabatic modes:

```
import numpy
k = numpy.geomspace(1,1e4,30)
T = structure.cutoff_ad(a=1,k) # power spectrum will be scaled by T^2
```

or the power spectrum of warm white noise:

```
P = structure.P_iso(a=1,k) # this is n*P, where n is the number density
```

## Requirements:

This code requires Python with the `numpy` and `scipy` packages.

## Acknowledgement:

If you use the code, please cite the paper [Amin et al. (2025)](https://arxiv.org/abs/2503.20881).
