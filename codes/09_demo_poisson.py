"""This demo program solves Poisson's equation

    - div grad u(x, y, z) = f(x, y, z)

on the unit sphere with source f given by

    f(x, y, z) = x
"""

from dolfin import *
import scipy
from scipy.sparse import csr_matrix

for nside in [32, 64, 128]:
    # Create mesh and define function space
    meshname = "09_meshes/HEALPix_{}.xml".format(nside)
    mesh = Mesh(meshname)
    global_normal = Expression(("x[0]", "x[1]", "x[2]"), degree=1)
    mesh.init_cell_orientations(global_normal)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression("x[0]", degree=4)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution and save stiffness matrix!
    u = Function(V)
    A = assemble(a)
    A_mat = as_backend_type(A).mat()
    A_sparray = csr_matrix(A_mat.getValuesCSR()[::-1], shape=A_mat.size)
    scipy.sparse.save_npz('stiffnessmatrix_{}.npz'.format(nside), A_sparray)

    # solve(a == L, u)

    # Save solution in pvd format
    # file = File("poisson_{}.pvd".format(nside))
    # file << u
