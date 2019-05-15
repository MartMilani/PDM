## 2019-05-14 Meeting sphere

### Frédérick

Done:
* ModelNet40
	* aligned shapes: Cohen 85, Esteves 89, Jiang 91, ours 87
	* randomly rotated shapes: -20% (Esteves -10%) (train on aligned, test on rotated)
	* DeepSphere v2 doesn't improve over DeepShere v1 (as for SHREC-17)
	* 90° rotation should be exact on healpix
* GHCN
	* many unknown values
	* temperature is kinda ok, rain is kinda ok, snow has very small coverage
		* temperature and rain have best coverage
	* all days in 4 years with temperature every day (subset) => 1198 stations (out of 56k in total)
		* superset (stations with values only once) => 17k
	* how to deal with missing values? mask? create different graphs? interpolate?
		* => start with the same graph with interpolation of missing values
	* cleaning: clearly out-of-range temperatures
* equiangular
	* changing weights along axis doesn't correct the geometry to the sphere

Next:
* exactly rotating a healpix map (e.g., by 90°) should give exactly the same results
* choose a set of stations that have enough measurements (say 75% of the days)
	* inpaint the missing stations with the PyGSP
* equiangular on hold while Martino finds a good way to build the graph

### Charles

Done:
* factorizing the graph in horizontal and vertical
	* same filters on both graphs, concatenate the output => equivariant to 90° rotation
	* mean pooling at the end to get invariance (from equivariance)
	* CIFAR-10: 2D CNN is best, then factored graph, then grid graph
* full spectrum: from 2D CNN (4 graphs, different filters) to most constrained (grid graph, same filters)

Next:
* test the different symmetries:
	* 1 graph => invariant to rotations / mirroring / flip
	* 2 graphs (horizontal & vertical) => equivariant to 90° rotations, invariant to 180° rotations, mirroring
		* unty the filters to even loose equivariance (then sum instead of concatenate)
	* 4 graphs (up, down, right, left) => equivariant to 90° & 180° rotations / mirroring
		* unty the filters to even loose equivariance (then sum instead of concatenate)
* hopefully the best symmetries for AID are different from CIFAR-10

Longer term:
* learn the symmetries through the network (depth-wise)

### Martino

Done:
* tested the FEM Laplacian
	* the "true Laplacian" is L' = B^-1 L (i.e., the operator)
		* we need to solve the problem B^-1 L to find the true Laplacian
	* eigendecomposition of L' is close to the spherical harmonics
		* converges (but not as good as graph Laplacian)
	* B has the same sparsity structure as L (given by the mesh)
	* B^-1 is dense, so L' = B^-1 L will be dense as well
	* filtering with the FEM (y = L' x = B^-1 L x) involves solving L x = B y (that is how the FEM solves any PDE)
* for use in a CNN, we'd have to backprop through a solver
	* People did it, see OptNet: Differentiable Optimization as a Layer in Neural Networks, https://arxiv.org/abs/1703.00443
* FEM need an embedding of the mesh

Next:
* Can we threshold L' = B^-1 L, as we do for L_graph?
	* When is B needed? When is a dense L' necessary?
* get the best of the graph Laplacian on the equiangular sampling
* FEM Laplacian better than graph Laplacian on non-regular samplings?
	* we saw that the graph Laplacian cannot integrate vertex' weights (to accommodate varying sampling density)
	* can the FEM fix that? Probably

Long term:
* When does the Laplacian need to be dense?
* Other operators than the laplacian?
* Good basis for learning to solve PDEs.
	* FEM is good for diffusion (e.g., Laplacian), finite volume is better when there is more transport (fluids, Navier-Stokes)
	* finite volume is discrete, as finite difference
