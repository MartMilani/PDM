
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
        tol = 1e-6
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
bws = [4, 8, 16]
for bw in bws:

    # Define mesh, function space
    mesh = Mesh("meshes/equi_{}.xml".format(bw))
    global_normal = Expression(("x[0]", "x[1]", "x[2]"), degree=1)
    mesh.init_cell_orientations(global_normal)
    V = FunctionSpace(mesh, "Lagrange", 1)
    dofs_x = V.tabulate_dof_coordinates()

    alpha = np.arange(2 * bw) * np.pi / bw
    beta = np.arange(2 * bw) * np.pi / (2. * bw)

    theta, phi = np.meshgrid(*(beta, alpha),indexing='ij')
    ct = np.cos(theta).flatten()
    st = np.sin(theta).flatten()
    cp = np.cos(phi).flatten()
    sp = np.sin(phi).flatten()
    x = st * cp
    y = st * sp
    z = ct
    coords = np.vstack([x, y, z]).T
    coords = np.asarray(coords, dtype=np.float32)
    coords = coords[2*bw-1:]
    npix = 2*bw*(2*bw-1)+1
    reordering_mask = np.empty(npix, dtype='int')
    indexes = np.arange(npix)
    for i, item in enumerate(coords):
        reordering_mask[i] = int(indexes[almost_equal(dofs_x, item)][0])
    assert len(np.unique(reordering_mask)) == npix
    reordering_mask = np.flip(reordering_mask)

    np.save("15_reordering_masks/reordering_mask_{}".format(bw), reordering_mask)
    print(reordering_mask)
    print("done")
