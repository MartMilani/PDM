"""
Heat equation
"""

from __future__ import print_function
from fenics import *
import numpy as np
import scipy
from scipy.sparse import csr_matrix

nside = 2

# Create mesh and define function space
# BIG PROBLEM: I HAVE NO IDEA OF HOW THE PIXELS ARE ORDERED!
meshname = "09_meshes/HEALPix_{}.xml".format(nside)
mesh = Mesh(meshname)
global_normal = Expression(("x[0]", "x[1]", "x[2]"), degree=1)
mesh.init_cell_orientations(global_normal)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# forcing term
f = Expression("0", degree=1)

# initial condition
u_n = interpolate(Expression("pow(x[0], 2)", degree=1), V)

dt = 0.10
a =  u*v*dx + dt*dot(grad(u), grad(v))*dx
L = (u_n + dt*f)*v*dx
    
# Time-stepping
u = Function(V)

num_steps = 20

# Create VTK file for saving solution
vtkfile = File('ft03_heat_results/time.pvd')

t = 0

A = assemble(a)
A_mat = as_backend_type(A).mat()
A_sparray = csr_matrix(A_mat.getValuesCSR()[::-1], shape=A_mat.size)
scipy.sparse.save_npz('stiffnessmatrix_{}.npz'.format(nside), A_sparray)

for n in range(num_steps):

    # Compute solution
    solve(a == L, u)

    # Update previous solution
    u_n.assign(u)

    # Save solution in pvd format
    
    vtkfile << (u, t)
    
    t += dt
