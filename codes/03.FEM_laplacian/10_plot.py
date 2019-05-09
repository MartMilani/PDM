
from __future__ import print_function
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from dolfin import *
import scipy
import numpy as np
import healpy as hp
from deepsphere import utils

plt.rcParams['figure.figsize'] = (17, 5)


# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

spectral_content = dict()
nsides = [4, 8, 16]
for nside in nsides:
    lmax = 3 * nside - 1
    N = np.cumsum(np.arange(1, 2*lmax+2, 2))[-1]
    eig_vectors = np.load("10_eig_vectors/eig_vectors_{}.npy".format(nside))
    
    
# ---------------------------------------------------------
    ne = 16
    for ind in range(ne):
        hp.mollview(eig_vectors[:, ind],
                    title='Eigenvector {}'.format(ind),
                    nest=False,
                    sub=(ne//4, 4, ind+1),
                    max=np.max(np.abs(eig_vectors[:, :ne])),
                    min=-np.max(np.abs(eig_vectors[:, :ne])),
                    cbar=False,

                    rot=(0,0,0))
    with utils.HiddenPrints():
        hp.graticule();


# ---------------------------------------------------------
    cl = np.empty((N, lmax+1))

    for i in range(N):
        eigenvector = eig_vectors[:,i]
        cl[i] = hp.sphtfunc.anafast(eigenvector, lmax=lmax, iter=8)

    spectral_content[nside] = np.empty((lmax+1, lmax+1))
    start = 0
    for ell in range(lmax+1):
        end = start + (2 * ell + 1)
        spectral_content[nside][ell] = np.sum(cl[start:end,:], axis=0)
        start = end

fig1, axes = plt.subplots(1, len(nsides))
fig2, ax2 = plt.subplots()

for ax, (nside, sc) in zip(axes, spectral_content.items()):
    sc = sc / sc[0, 0]
    im = ax.imshow(sc, cmap=plt.cm.gist_heat_r)
    ax.set_title(rf'$N_{{side}}={nside}$')

    energy_in = np.diag(sc)
    ax2.plot(energy_in, 'o', label=rf'$N_{{side}}={nside}$')

ax2.legend();
plt.show()

# ---------------------------------------------------------
eig_values = np.load('10_eig_values/eig_values_16.npy')

plt.plot(eig_values[:50], 'b.')
plt.plot(eig_values[:50], 'b-')
for idx in range(7):
    plt.text(idx**2, eig_values[idx**2] + 0.01, 'l = {}'.format(idx));
plt.savefig('FEM_eigenvalues')
plt.show()



