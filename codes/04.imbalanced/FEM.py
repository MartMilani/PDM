
from __future__ import print_function
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from dolfin import *
import scipy
import numpy as np




# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

spectral_content = dict()


N = 25  # how many eigenvectors do I calculate

# Define mesh, function space
mesh = Mesh("meshes/imbalanced.xml")
global_normal = Expression(("x[0]", "x[1]", "x[2]"), degree=1)
mesh.init_cell_orientations(global_normal)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define basis and bilinear form
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx
b = dot(u, v)*dx
# Assemble stiffness form
A = PETScMatrix()
B = PETScMatrix()
assemble(a, tensor=A)
assemble(b, tensor=B)

A_mat = as_backend_type(A).mat()
A_sparray = csr_matrix(A_mat.getValuesCSR()[::-1], shape=A_mat.size)

npix = 1025
scipy.sparse.save_npz('matrices/stiffness_matrix.npz', A_sparray)

B_mat = as_backend_type(B).mat()
B_sparray = csr_matrix(B_mat.getValuesCSR()[::-1], shape=B_mat.size)
scipy.sparse.save_npz('matrices/mass_matrix.npz', B_sparray)

# Create eigensolver
eigensolver = SLEPcEigenSolver(A, B)
eigensolver.parameters['spectrum'] = 'target real'
eigensolver.parameters['tolerance'] = 1.e-4
eigensolver.parameters['maximum_iterations'] = 10000

# Compute all eigenvalues of A x = \lambda x
print("Computing eigenvalues. This can take a minute.")
eigensolver.solve(N)

print('Done. Saving results...')
file = File("eigenvectors/eigenvectors.pvd")

eig_vectors = np.ndarray((npix, N), dtype='float')
eig_values = np.ndarray(N, dtype='float')
for i in range(N):
    # Extract largest (first) eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(i)

    # Initialize function and assign eigenvector
    u = Function(V)
    u.vector()[:] = rx
    eig_vectors[:, i] = u.compute_vertex_values()
    eig_values[i] = r
    file << (u, i)
