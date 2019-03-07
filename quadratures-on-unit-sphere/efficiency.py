'''
Script to create a plot of the efficiency factor of different
quadrature schemes on the sphere. Efficiency factor taken from

@article{McLaren1963,
author = {McLaren, A. D.},
doi = {10.2307/2003998},
issn = {00255718, 10886842},
journal = {Mathematics of Computation},
number = {84},
pages = {361--383},
publisher = {American Mathematical Society},
title = {{Optimal Numerical Integration on a Sphere}},
url = {http://www.jstor.org/stable/2003998},
volume = {17},
year = {1963}
}

'''

import numpy as np
import matplotlib.pyplot as plt
import glob
# import matplotlib2tikz

# Lebedev grid loading and efficiency
leb_list = sorted(glob.glob('PointDistFiles/lebedev/*'))
T_leb = np.zeros(len(leb_list))
N_leb = np.zeros(len(leb_list))
for (i, leb_file) in enumerate(leb_list):
    N_leb[i] = sum(1 for line in open(leb_file))
    T_leb[i] = float(leb_file[-7:-4])
E_leb = (T_leb + 1)**2 / (3.0 * N_leb)

# Spherical designs loading and efficiency
# Points taken from http://web.maths.unsw.edu.au/~rsw/Sphere/
sph_list = sorted(
    glob.glob('PointDistFiles/sphdesigns/WomersleySym/ss*'))  # symmetric
# sph_list = sorted(glob.glob('PointDistFiles/sphdesigns/WomersleyNonSym/sf*')) ## non-symmetric
# Points taken from http://neilsloane.com/sphdesigns/dim3/
#sph_list = sorted(glob.glob('PointDistFiles/sphdesigns/HardinSloane/hs*'))
T_sph = np.zeros(len(sph_list))
N_sph = np.zeros(len(sph_list))
for (i, sph_file) in enumerate(sph_list):
    N_sph[i] = float(sph_file[-5:])
    T_sph[i] = float(sph_file[-9:-6])
E_sph = (T_sph + 1)**2 / (3.0 * N_sph)

plt.plot(T_leb, E_leb, 'r-x', lw=2, label='Lebedev')
plt.plot(T_sph, E_sph, 'kx', lw=2, label=r'Spherical $t$-design')
# Note that E=2/3 for Gaussian prod.
plt.plot([0, 140], 2 * [2.0 / 3], 'b-', lw=4, label='Gaussian product')
plt.ylim([0, 1.1])
plt.xlim([0, 140])
plt.xlabel(r'$p$')
plt.ylabel(r'$E$')
plt.legend(loc='lower right')
plt.show()
# Possibility to save using matplotlib2tikz
# matplotlib2tikz.save('../Graphics/efficiency.tex',figurewidth='\\figurewidth')
