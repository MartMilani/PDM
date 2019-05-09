
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


def float_almost_equal(a, b):
        tol = 1e-5
        return (a > (b - tol)) and (a < (b + tol))

    
def almost_equal(matrix, item):
    mask = np.zeros(np.alen(matrix), dtype='bool')    
    for i, row in enumerate(matrix):
        sub_array = np.zeros(len(item))
        for j, (x, y) in enumerate(zip(row, item)):
            sub_array[j] = float_almost_equal(x,y)
        if sub_array.all():
            mask[i] = True
    return mask
    
    
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
    
    # Define mesh, function space
    mesh = Mesh("09_meshes/HEALPix_{}.xml".format(nside))
    global_normal = Expression(("x[0]", "x[1]", "x[2]"), degree=1)
    mesh.init_cell_orientations(global_normal)
    V = FunctionSpace(mesh, "Lagrange", 1)
    dofs_x = V.tabulate_dof_coordinates()
    
    indexes = range(nside**2 * 12)
    x, y, z = hp.pix2vec(nside, indexes)
    coords = np.vstack([x, y, z]).transpose()
    coords = np.asarray(coords)
    reordering_mask = np.empty(np.alen(coords), dtype='int')
    indexes = np.arange(nside**2 * 12)
    for i, item in enumerate(coords):
        reordering_mask[i] = int(indexes[almost_equal(dofs_x, item)][0])
    assert len(np.unique(reordering_mask)) == 12*nside**2
    np.save("reordering_mask_{}".format(nside), reordering_mask)
    print("done")
    