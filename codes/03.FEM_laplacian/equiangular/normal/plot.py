
from __future__ import print_function
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from dolfin import *
import scipy
import numpy as np
from deepsphere import utils
import pyshtools


plt.rcParams['figure.figsize'] = (17, 5)



def to_array(f, bw):

    """From a 1-d vector to a 2D grid necessary to initiate a pyshtools.SHGrid object"""
    height, width = 2*bw, 2*bw
    array = np.zeros((height, width))  # shape=(longitude, latitude)
    f = np.append([f[0]]*(2*bw-1), f)  # correct! the first line is the North pole repeated 2bw times

    # now we need to undo the meshgrid
    assert f.size == array.size
    for n, fx in enumerate(f):
        j = n%width
        i = n//width
        array[i, j] = fx
    return array

# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

spectral_content = dict()
bws = [8, 16, 32]

for bw in bws:
    lmax = bw-1  # I'm limited by the Driscoll/Healy sampling theorem
    npix = 2*bw*(2*bw-1)+1
    N = np.cumsum(np.arange(1, 2*lmax+2, 2))[-1]  # how many eigenvectors do I calculated in FEM.py
    eig_vectors = np.load("eig_vectors/eig_vectors_{}.npy".format(bw))


# ---------------------------------------------------------

    cl = np.empty((N, lmax+1))

    for i in range(N):
        eigenvector = eig_vectors[:,i]

        # ---------ANAFAST ON THIS SAMPLING DOES NOT WORK ANYMORE-------
        ### cl[i] = hp.sphtfunc.anafast(eigenvector, lmax=lmax, iter=8)
        eig_array = to_array(eigenvector, bw)
        g = pyshtools.SHGrid.from_array(eig_array)
        clm = g.expand(normalization='unnorm')
        cl[i] = clm.spectrum()

    spectral_content[bw] = np.empty((lmax+1, lmax+1))
    start = 0
    for ell in range(lmax+1):
        end = start + (2 * ell + 1)
        spectral_content[bw][ell] = np.sum(cl[start:end,:], axis=0)/ np.sum(cl[start:end,:])
        start = end

fig1, axes = plt.subplots(1, len(bws))
fig2, ax2 = plt.subplots()

for ax, (bw, sc) in zip(axes, spectral_content.items()):
    sc = sc / sc[0, 0]
    im = ax.imshow(sc, cmap=plt.cm.gist_heat_r)
    ax.set_title(rf'$bw={bw}$')

    energy_in = np.diag(sc)
    ax2.plot(energy_in, 'o', label=rf'$bw={bw}$')

ax2.legend();
plt.show()
fig1.savefig('normal_FEM')
fig2.savefig('normal_FEM_diagonal')
# ---------------------------------------------------------
eig_values = np.load('eig_values/eig_values_32.npy')

plt.plot(eig_values[:50], 'b.')
plt.plot(eig_values[:50], 'b-')
for idx in range(7):
    plt.text(idx**2, eig_values[idx**2] + 0.01, 'l = {}'.format(idx));
plt.savefig('FEM_eigenvalues')
plt.show()
