
from __future__ import print_function
from dolfin import *
import scipy
import numpy as np


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



# Define mesh, function space
mesh = Mesh("meshes/imbalanced.xml")
global_normal = Expression(("x[0]", "x[1]", "x[2]"), degree=1)
mesh.init_cell_orientations(global_normal)
V = FunctionSpace(mesh, "Lagrange", 1)
dofs_x = V.tabulate_dof_coordinates()

coords = mesh.coordinates()
npix = np.alen(coords)
reordering_mask = np.empty(npix, dtype='int')
indexes = np.arange(npix)
for i, item in enumerate(coords):
    reordering_mask[i] = int(indexes[almost_equal(dofs_x, item)][0])
assert len(np.unique(reordering_mask)) == npix

# ------- WHY DO I NEED THIS? I DON'T UNDERSTAND --------
reordering_mask = np.flip(reordering_mask)
# ---------------------------------------------------------

np.save("reordering_mask_{}", reordering_mask)
print(reordering_mask)
print("done")
