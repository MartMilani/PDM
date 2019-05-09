
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

    
for nside in [4, 8, 16]:
    lmax = 3 * nside - 1
    N = np.cumsum(np.arange(1, 2*lmax+2, 2))[-1]
    # Define mesh, function space
    mesh = Mesh("09_meshes/HEALPix_{}.xml".format(nside))
    global_normal = Expression(("x[0]", "x[1]", "x[2]"), degree=1)
    mesh.init_cell_orientations(global_normal)
    V = FunctionSpace(mesh, "Lagrange", 1)
    dofs_x = V.tabulate_dof_coordinates()
    import pdb
    pdb.set_trace()
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
    scipy.sparse.save_npz('10_matrices/stiffness_matrix_{}.npz'.format(nside), A_sparray)

    plt.spy(A_sparray)
    plt.show()

    B_mat = as_backend_type(B).mat()
    B_sparray = csr_matrix(B_mat.getValuesCSR()[::-1], shape=B_mat.size)
    scipy.sparse.save_npz('10_matrices/mass_matrix_{}.npz'.format(nside), B_sparray)

    # Create eigensolver
    eigensolver = SLEPcEigenSolver(A, B)
    eigensolver.parameters['spectrum'] = 'target real'
    eigensolver.parameters['tolerance'] = 1.e-4
    eigensolver.parameters['maximum_iterations'] = 1000
    # Compute all eigenvalues of A x = \lambda x
    print("Computing eigenvalues. This can take a minute.")
    eigensolver.solve(N)
    print('Done. Saving results...')
    file = File("10_eigenvectors/eigenvectors_{}.pvd".format(nside))

    eig_vectors = np.ndarray((12*nside**2, N), dtype='float')
    for i in range(N):
        # Extract largest (first) eigenpair
        r, c, rx, cx = eigensolver.get_eigenpair(i)

        # Initialize function and assign eigenvector
        u = Function(V)
        u.vector()[:] = rx
        eig_vectors[:, i] = u.compute_vertex_values()
        file << (u, i)
    np.save("eig_vectors_{}".format(nside), eig_vectors)

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


# ---------------------------------------------------------
    cl = np.empty((N, lmax+1))

    for i in range(N):
        eigenvector = eig_vectors[:,i]
        cl[i] = hp.sphtfunc.anafast(eigenvector, lmax=lmax, iter=8)

    spectral_content = np.empty((lmax+1, lmax+1))
    start = 0
    for ell in range(lmax+1):
        end = start + (2 * ell + 1)
        spectral_content[ell] = np.sum(cl[start:end,:], axis=0)
        start = end

    fig1, axes = plt.subplots()
    fig2, ax2 = plt.subplots()


    spectral_content = spectral_content / spectral_content[0, 0]
    im = axes.imshow(spectral_content, cmap=plt.cm.gist_heat_r)
    axes.set_title(rf'$N_{{side}}={nside}$')

    energy_in = np.diag(spectral_content)
    ax2.plot(energy_in, 'o', label=rf'$N_{{side}}={nside}$')

    ax2.legend();
    plt.show()


# ---------------DEEPSPHERE V2 ----------------------------


nsides = [4, 8, 16]
sigmas = [0.1003, 0.02561, 0.00647, 0.001628]
deepsphere_thresholded_graphs = []
spectral_content = dict()

k= 0.01
 
for nside, sigma in zip(nsides, sigmas):

    lmax = 3 * nside - 1

    n_harmonics = np.cumsum(np.arange(1, 2*lmax+2, 2))[-1]
    full_graph = utils.full_healpix_graph(nside, dtype=np.float64, std=sigma)  # in NEST order
    
    # sparsifying the graph
    W = full_graph.W.copy()
    W[W < k] = 0
    deepsphere_thresholded_graphs.append(pg.graphs.Graph(W))
    graph = deepsphere_thresholded_graphs[-1]
    print("Threshold: ", k)
    print("Number of neighbours: ", np.mean(np.sum(W>0, axis=1)))
    
    graph.compute_fourier_basis(n_eigenvectors=n_harmonics)

    cl = np.empty((n_harmonics, lmax+1))
    for i in range(n_harmonics):
        eigenvector = hp.reorder(graph.U[:, i], n2r=True)
        # alm = hp.sphtfunc.map2alm(eigenvector)
        cl[i] = hp.sphtfunc.anafast(eigenvector, lmax=lmax, iter=8)

    spectral_content[nside] = np.empty((lmax+1, lmax+1))
    start = 0
    for ell in range(lmax+1):
        end = start + (2 * ell + 1)
        spectral_content[nside][ell] = np.sum(cl[start:end,:], axis=0)
        start = end

fig1, axes = plt.subplots(1, len(nsides))
fig2, ax2 = plt.subplots()

for ax, (nside, sc) in zip(axes, spectral_content.items()):
    sc = sc / sc[0, 0]
    im = ax.imshow(sc, cmap=plt.cm.gist_heat_r)
    ax.set_title(rf'$N_{{side}}={nside}$')
    energy_in = np.diag(sc)
    ax2.plot(energy_in, 'o', label=rf'$N_{{side}}={nside}$')

ax2.legend();


