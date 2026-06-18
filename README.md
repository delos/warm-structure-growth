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
import numpy
structure = warm_structure_growth.Structure(
    a_i = 3e-7, # scale factor at which we start nonrelativistic free streaming
    f = lambda v: numpy.exp(-v**2/2), # Maxwell-Boltzmann velocity distribution
    v_scale = 7e-5, # scale velocity at matter-radiation equality, in units of c
    n = 5e7, # particle number density in Mpc^-1
    )
```

The velocity distribution can alternatively be supplied as a table and passed as `f=(v,f)`, where `v` and `f` are arrays, or chosen from a set of built-in analytic distributions by passing a string, such as `"maxwell"`. Field (wave dark matter) calculations currently only support the built-in distributions. See the docstring (`help(warm_structure_growth.Structure)` or in [main.py](warm_structure_growth/main.py)) for further options. Now we can evaluate the transfer function for adiabatic modes:

```
k = numpy.geomspace(1,1e4,30)
T = structure.cutoff_ad(a=1.,k=k) # power spectrum will be scaled by T^2
```

or the power spectrum of warm white noise:

```
P = structure.P_iso(a=1.,k=k)
```

Wavenumbers are in units of 1/Mpc by default, but this can be changed by specifying a custom `k_eq` when instantiating `warm_structure_growth.Structure`. The unit of `k` will be the same as the unit of `k_eq`.

### Advanced usage

`n` (number density) is only relevant in particle mode. To use field mode (wave dark matter), instead pass `m` (field mass) or `k_scale` (characteristic field momentum), or both, in which case `v_scale` is ignored. `m` and `k_scale` are in the same units as `k_eq`. Note that field calculations currently only support the built-in distributions.

## Requirements:

This code requires Python with the `numpy` and `scipy` packages.

## Acknowledgement:

If you use the code, please cite the paper [Amin et al. (2025)](https://arxiv.org/abs/2503.20881).
