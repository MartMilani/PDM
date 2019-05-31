
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




# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

spectral_content = dict()
bws = [32]
for bw in bws:
    npix = 2*bw*(2*bw-1)+1
    lmax = bw
    N = np.cumsum(np.arange(1, 2*lmax+2, 2))[-1]  # how many eigenvectors do I calculate

    # Define mesh, function space
    mesh = Mesh("../../normal/meshes/equi_{}.xml".format(bw))
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

    # Create eigensolver
    import scipy.linalg

    B_lumped_inverse = np.diag(1./np.sum(B.array(), axis=1))
    A_array = A.array()
    M = B_lumped_inverse@A_array
    M = scipy.sparse.csc_matrix(M)

    eig_values, eig_vectors = scipy.sparse.linalg.eigs(M, N, which='SR', tol=1e-4)

    print('Done. Saving results...')
    file = File("eigenvectors/eigenvectors_{}.pvd".format(bw))

    for i in np.argsort(eig_values):
        r = eig_values[i]
        rx = eig_vectors[:,i]

        # Initialize function and assign eigenvector
        u = Function(V)
        u.vector()[:] = rx
        eig_vectors[:, i] = u.compute_vertex_values()
        eig_values[i] = r
        file << (u, i)
    np.save("eig_vectors/eig_vectors_{}".format(bw), eig_vectors)
    np.save("eig_values/eig_values_{}".format(bw), eig_values)
