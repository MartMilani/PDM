import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

import scipy.sparse
from mpl_toolkits.mplot3d import Axes3D

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import scipy as sp
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp


nside = 4


# eigendecomposition
# Create eigensolver
npix = 12*nside**2
N = npix


L = scipy.sparse.load_npz('HEALPix/10_matrices/stiffness_matrix_{}.npz'.format(nside))
B = scipy.sparse.load_npz('HEALPix/10_matrices/mass_matrix_{}.npz'.format(nside))

reordering_mask = np.load("HEALPix/15_reordering_masks/reordering_mask_{}.npy".format(nside))

eig_values, eig_vectors = scipy.linalg.eigh(L.toarray(), B.toarray())
eig_values_normalized = eig_values/np.max(eig_values)


eig_vectors_INV = np.linalg.inv(eig_vectors)


# just for check
L_reconstructed = eig_vectors@np.diag(eig_values)@eig_vectors_INV
L_reconstructed = L_reconstructed[reordering_mask]
L_reconstructed = L_reconstructed[:, reordering_mask]
L = L[reordering_mask]
B = B[reordering_mask]
L = L[:, reordering_mask]
B = B[:, reordering_mask]
B_inv = scipy.sparse.linalg.inv(B)
assert np.max(B_inv@L - L_reconstructed)<1e-5


signal = np.zeros(12*nside**2)
signal[0] = 1

def subplotsphere(fig, signal, coords, tri, j):
    ax = fig.add_subplot(2,2,j+1, projection='3d')
    M = np.max(signal)
    for simplex in tri.simplices:
        triangle = a3.art3d.Poly3DCollection([coords[simplex]])
        triangle.set_color(colors.rgb2hex([np.max(signal[simplex])/M, 0,0]))
        triangle.set_edgecolor('k')
        ax.add_collection3d(triangle)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
def diffuse(x):
    return eig_vectors@np.diag(1/(1+100*eig_values_normalized))@eig_vectors_INV@x

fig = plt.figure(figsize=(15,15))

signal = np.zeros(12*nside**2)
signal[0] = 1
indexes = range(nside**2 * 12)
x, y, z = hp.pix2vec(nside, indexes)
coords = np.vstack([x, y, z]).transpose()
coords = np.asarray(coords)
tri = ConvexHull(coords)  # just for plotting

for j in range(4):
    subplotsphere(fig, signal, coords, tri, j)
    
    # diffusing 3 times
    for i in range(1):
        signal = abs(diffuse(signal))

plt.show()
plt.savefig('FEM_filter')