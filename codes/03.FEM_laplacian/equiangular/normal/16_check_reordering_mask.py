
from __future__ import print_function
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from dolfin import *
import scipy
import numpy as np
import pyshtools
from deepsphere import utils
    
# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

def to_array(f, bw):

    """From a 1-d vector to a 2D grid necessary to initiate a pyshtools.SHGrid object"""
    height, width = 2*bw, 2*bw
    array = np.zeros((height, width))  # shape=(longitude, latitude)
    f = np.append([f[0]]*(2*bw-1), f)  # correct! the first line is the North pole repeated 2bw times

    # now we need to undo the meshgrid
    assert f.size == array.size
    for n, fx in enumerate(f):
        j = n%width
        i = n//width
        array[i, j] = fx
    return array

spectral_content = dict()
spectral_content_reordered = dict()
bws = [4, 8]
for bw in bws:
    lmax = bw-1
    N = np.cumsum(np.arange(1, 2*lmax+2, 2))[-1]
    npix = 2*bw*(2*bw-1)+1
    # Define mesh, function space
    mesh = Mesh("meshes/equi_{}.xml".format(bw))
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
    

    eig_vectors = np.ndarray((npix, N), dtype='float')
    eig_values = np.ndarray(N, dtype='float')        


    

    for i in range(N):
        # Extract largest (first) eigenpair
        r, c, rx, cx = eigensolver.get_eigenpair(i)
        
        # ----- keeping the dof ordering -----
        eig_vectors[:, i] = np.asarray(rx)
        eig_values[i] = r
    
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    cl = np.empty((N, lmax+1))
    spectral_content[bw] = np.empty((lmax+1, lmax+1))
    for i in range(N):
        eigenvector = eig_vectors[:,i]
        # ---------ANAFAST ON THIS SAMPLING DOES NOT WORK ANYMORE-------
        ### cl[i] = hp.sphtfunc.anafast(eigenvector, lmax=lmax, iter=8)
        eig_array = to_array(eigenvector, bw)
        g = pyshtools.SHGrid.from_array(eig_array)
        clm = g.expand(normalization='unnorm')
        cl[i] = clm.spectrum()

   
    start = 0
    for ell in range(lmax+1):
        end = start + (2 * ell + 1)
        spectral_content[bw][ell] = np.sum(cl[start:end,:], axis=0)/ np.sum(cl[start:end,:])
        start = end
    
    # ---------------------------------------------------------
    # ---------- reordering ----------
    reordered_mask = np.load('15_reordering_masks/reordering_mask_{}.npy'.format(bw))
    eig_vectors = eig_vectors[reordered_mask]
    
    
    # ---------------------------------------------------------
    # ---------------------------------------------------------
 
    
    cl_reordered = np.empty((N, lmax+1))
    spectral_content_reordered[bw] = np.empty((lmax+1, lmax+1))
    for i in range(N):
        eigenvector = eig_vectors[:,i]
        # ---------ANAFAST ON THIS SAMPLING DOES NOT WORK ANYMORE-------
        ### cl[i] = hp.sphtfunc.anafast(eigenvector, lmax=lmax, iter=8)
        eig_array = to_array(eigenvector, bw)
        g = pyshtools.SHGrid.from_array(eig_array)
        clm = g.expand(normalization='unnorm')
        cl_reordered[i] = clm.spectrum()

    
    start = 0
    for ell in range(lmax+1):
        end = start + (2 * ell + 1)
        spectral_content_reordered[bw][ell] = np.sum(cl_reordered[start:end,:], axis=0)/ np.sum(cl_reordered[start:end,:])
        start = end
        
        

fig1, axes = plt.subplots(1, len(bws))
fig2, ax2 = plt.subplots()

for ax, (bw, sc) in zip(axes, spectral_content.items()):
    sc = sc / sc[0, 0]
    im = ax.imshow(sc, cmap=plt.cm.gist_heat_r)
    ax.set_title(rf'$bw={bw}$')

    energy_in = np.diag(sc)
    ax2.plot(energy_in, 'o', label=rf'$bw={bw}$')

ax2.legend();
plt.show()


    

fig3, axes = plt.subplots(1, len(bws))
fig4, ax2 = plt.subplots()

for ax, (bw, sc) in zip(axes, spectral_content_reordered.items()):
    sc = sc / sc[0, 0]
    im = ax.imshow(sc, cmap=plt.cm.gist_heat_r)
    ax.set_title(rf'$bw={bw}$')

    energy_in = np.diag(sc)
    ax2.plot(energy_in, 'o', label=rf'$bw={bw}$')

ax2.legend()
plt.show()
      
    