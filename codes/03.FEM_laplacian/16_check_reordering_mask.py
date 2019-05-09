
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
nsides = [8]
for nside in nsides:
    lmax = 3 * nside - 1
    N = np.cumsum(np.arange(1, 2*lmax+2, 2))[-1]
    
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
    eigensolver = SLEPcEigenSolver(A, B)
    eigensolver.parameters['spectrum'] = 'target real'
    eigensolver.parameters['tolerance'] = 1.e-3
    eigensolver.parameters['maximum_iterations'] = 100
    # Compute all eigenvalues of A x = \lambda x
    print("Computing eigenvalues. This can take a minute.")
    eigensolver.solve(N)
    
    
    print('Done. Extracting results...')
    

    eig_vectors = np.ndarray((12*nside**2, N), dtype='float')
    eig_values = np.ndarray(N, dtype='float')
    ne = 16
    for i in range(ne):
        # Extract largest (first) eigenpair
        r, c, rx, cx = eigensolver.get_eigenpair(i)
        
        # ----- keeping the dof ordering -----
        eig_vectors[:, i] = np.asarray(rx)
        eig_values[i] = r
   
    for ind in range(ne):
        hp.mollview(eig_vectors[:, ind],
                    title='Eigenvector {}'.format(ind),
                    nest=False,
                    sub=(ne//4, 4, ind+1),
                    max=np.max(np.abs(eig_vectors[:, :ne])),
                    min=-np.max(np.abs(eig_vectors[:, :ne])),
                    cbar=False,

                    rot=(0,0,0))
    with utils.HiddenPrints():
        hp.graticule();
    plt.show()
    
    # ---------- reordering ----------
    reordered_mask = np.load('reordering_mask_{}.npy'.format(nside))
    eig_vectors = eig_vectors[reordered_mask]
    # --------------------------------
    
    ne = 16
    for ind in range(ne):
        hp.mollview(eig_vectors[:, ind],
                    title='Eigenvector {}'.format(ind),
                    nest=False,
                    sub=(ne//4, 4, ind+1),
                    max=np.max(np.abs(eig_vectors[:, :ne])),
                    min=-np.max(np.abs(eig_vectors[:, :ne])),
                    cbar=False,

                    rot=(0,0,0))
    with utils.HiddenPrints():
        hp.graticule();
    plt.show()



      
    