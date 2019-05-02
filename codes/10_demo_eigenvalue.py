"""
This program illustrates basic use of the SLEPc eigenvalue solver for
a standard eigenvalue problem.
"""

# Copyright (C) 2007 Kristian B. Oelgaard
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008.
# Modified by Marie Rognes, 2009.
#
# First added:  2007-11-28
# Last changed: 2009-10-09

# Begin demo
from __future__ import print_function
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

from dolfin import *


nside = 4

# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

# Define mesh, function space
mesh = Mesh("09_meshes/HEALPix_{}.xml".format(nside))
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
eigensolver = SLEPcEigenSolver(A)
eigensolver.parameters['spectrum'] = 'target real'
eigensolver.parameters['tolerance'] = 1.e-14
# Compute all eigenvalues of A x = \lambda x
print("Computing eigenvalues. This can take a minute.")
eigensolver.solve()

for i in range(9):
    # Extract largest (first) eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(i)

    print("Smallest eigenvalue: ", r)

    # Initialize function and assign eigenvector
    u = Function(V)
    u.vector()[:] = rx

    file = File("eigenvector_{}.pvd".format(i))
    file << u
