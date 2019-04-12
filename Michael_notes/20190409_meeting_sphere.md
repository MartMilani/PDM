## 2019-04-09 Meeting sphere

### Frédérick

Done:
* use the TF input pipeline => 20% gain in training time
* Nside = 64 or 128 => same performance with the same architectures as 32
	* 6 layers: 128->64->32->16->8->4 (shall see the whole sphere)
	* number of hops to cross the sphere: sqrt(3) * Nside => sqrt(3) * 128 = 221
	* filters are spatially smaller => shall we increase K?
	* does it mean that a resolution of 32 is sufficient?
	* shapes look quite recognizable, even for Nside = 8 => Nside = 32 might be more than enough
* 2 to 6 layers: F1 stayed at 76 on val set
* cross-entropy augments on the validation set
* random translations => not better on val set, but less overfitting
* read and summarize Equivariant Multi-View Networks from Esteves
	* SOTA on 3D datasets: multi-view methods are working best
	* project colors on images from multiple views
	* how is occlusion treated differently from the sphere?
	* change of coordinates for the 2D CNNs to be invariant to rotations
	* symmetry group of the icosahedron is the larger discrete subgroup of SO(3) (12, 20, 60 rotations)
	* 2D CNN as feature extractors, then group equivariant convolution for rotations on the icosahedron
	* performance is much better than everything else
* projection from the outside of the sphere
	* can leverage part of sphere, as objects won't cover the whole sphere
	* influence of the sampling (projection on the equator or pole)

Next:
* try with better graphs
	* ask Martino to generate a good one for Nside=32
	* do the same with the equirectangular? we don't understand it well enough yet
* understand why the loss augments on the validation set (while it converges on the training set)
	* not due to the absence of the additive l2 regularization
	* search on Google
	* indication of over-fitting?
	* bad learning rate?
* as we overfit, try with more regularization
	* try without and with more l2 on the weights
	* dropout on convolutional layers: drop filters, or feature maps (look online)
		* implement as an element-wise product with a masking matrix (the K weights of a filter should be set to zero)
* try with Nside = 8, 16
	* performance vs resolution should be a curve that saturates at some point
	* look at the confusion matrix (use the function from sklearn) => do some classes become indistinguishable?
* put the spheres (equirectangular, Cohen's equiangular, healpix) in the PyGSP

### Charles

Done:
* scale in [-0.8, 0.8] doesn't change much (was chance before)
* weight initialization probably did not change much
* tried padding and cropping to avoid border effects
	* much slower (as implementation is hand-made)
	* not much improvement => 1% augmentation
* pytorch geometric
	* different batching (assumes a different graph per sample)
* diagonals on the grid graph => currently running
* factor graph paper => we expect a performance drop from anisotropic filters on CIFAR-10

Next:
* pytorch geometric: reshape the matrix (from #batches signals on the same graph, to #batches graphs in parallel)
* augment the number of filters to have the same number of parameters
* need to include diagonals to compare with 2D CNN with exactly the same spatial extent
	* K=2 with diagonals = 3x3, K=3 with diagonals = 5x5
	* without diagonals, graph kernels have a diamond shape
	* Weight on the diagonal? 1/sqrt(2)? Should preserve equivariance.
		* Look at what Renata & Pascal did
* get the code to build the hexa grid and do the interpolation (ask the first author)

### Martino

Done:
* Nobile thinks in terms of FEM
	* their functions is a weighted sum of localized basis functions
	* their Laplacian is also a matrix => filters as polynomials of the FEM Laplacian
	* define a quadrature rule that solves the integral, then try to sparsify it
	* in the end, all those methods want to approximate an integral with a sparse matrix
* Much better convergence with smaller stddev of the kernel
	* capture higher frequencies due to less averaging
	* sparse graph is just a numerical approximation (by setting small values to zero)
* B&N study asymptotic behavior
	* the constant in front of the stddev matters
* sparsification
	* set stddev at 0.01 for Nside = 16
	* cut values at 4 sigmas (threshold of epsilon = exp(-4) = 0.018)
	* gives 30 neighbors on average
	* that is sufficient, we only lose the last 2-3 eigenvectors compared to full graph (which has 3k neighbors)
	* good up to lmax = 2 Nside
	* SHT from healpix is good up to 3 Nside (over 0.92) => still some room for improvements
* with a good FT, we can also be equivariant
	* as rotation is a change in the m, but it stays in the subspaces
	* problem: the GFT is O(Npix^2) = O(Nside^4), which is more expansive than the SHT

Next:
* study how to optimally set sigma
	* better plot (log?) or a quantitative measure for alignment (we're now too good to see a difference)
	* for a fixed Nside, what is the impact of sigma?
		* Is there a trade-off between approximating low and high frequencies?
				NOT REALLY
		* Small sigma can capture high frequencies, large sigma approximates the integral better.
			WHY A LARGE SIGMA SHOULD APPROXIMATE THE INTEGRAL BETTER? I DON'T AGREE
	* setting sigma as a function of Nside, does it now converge as we increase Nside?
		* should the number of neighbors (or, number of neighbors closer than 4 sigma) increase as Nside increases? YES
		* if it can stay constant, what was the issue with our initial graph construction? Simply not enough neighbors? (we had 8)
				TRIED WITH 14 (more or less double as before) AND GET THE SAME SHIT -> IT CAN'T STAY CONSTANT
	* Can we theoretically motivate an optimal sigma, for a given finite and deterministic sampling? NO IDEA WHERE TO START.
		* B&N might not be the optimal framework to think about that, as all the results are asymptotic.
* study the connection with finite elements / differences
	* hard to integrate the knowledge that we know the manifold
	* see if mesh-free methods might be more appropriate (ask Nobile for a reference)
	* start on the circle: the matrices from finite elements and finite differences are the same as the graph Laplacian
	* what happens with a non-uniform sampling of the circle?
* finish the B&N pointwise convergence proof
	* done, we only need to understand the formula for the radius of the pixels
	* if no answer from author, we can ask Cardoso (he also knows the healpix authors)
* play with the platonic solids: those are the sole 5 samplings that are regular
	* postponed for now
