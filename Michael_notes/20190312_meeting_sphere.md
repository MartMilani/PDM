## 2019-03-12 Meeting sphere

### Frédérick

Done:
* adapted our model to be close to that of Cohen
	* non-local kernels => K=5 and sqrt(12) nside / 2
		* 400k for Cohen, 59k and 170k four ours
		* in general better accuracy with K=5
	* they use batch norm as well as well
	* same number of feature maps
	* average pooling instead of their integration over SO(3)
* runtime per batch (32 samples of 12k pixels): cohen is 0.4s per batch, ours is 0.12 for K=15, and 0.58s for K=56
	* wall time (with time.perf_counter)
* performance around 70 or 75%
	* but 59% with the evaluation script
* Cohen without augmentation looses (precision from 0.701 to 0.669)
* large label distribution: 5 classes are much larger (55 classes in total)

Next:
* fix evaluation
* ask implementation to Esteves (e.g., to know the number of parameters, and compare with same conditions)
* test ours with augmentation (rotation shouldn't change much, translation should)
* test ours with equirectangular sampling
* should have one "most comparable" configuration, and one that works best (eventually some in-between for explanations)

### Charles

Done:
* found a simple architectures: 2 conv, pooling, 2 conv, fully connected
	* 86% on CIFAR-10
* same architecture with the graph
* pooling works, i.e., doesn't destroy 2D structure
* Presented his understanding and critique of Renata's work
	* differently from them, we'll treat invariance through the relation with the continuous world
	* why would you use dynamic pooling when you know the manifold?
	* I don't see a good motivation for the stat layer (plus it's empirically useless)
		* averaging is well motivated: aggregation of evidence, variance reduction
	* badly written, not much motivation for design choices
	* treat equivariance in the discrete domain => easier in the continuous domain

Next:
* tweak learning rate (and other hyper-parameters) to converge and obtain reasonable performance
* test with normalized laplacian (shouldn't make much of a difference)

Long term:
* non-uniform sampling of images
	* require continuous data, but arbitrary rotations as well (synthetic or polynomial interp?)
* directed graphs
* biomedical images

### Martino

Done:
* not much structure in optimized Fourier basis of the non-uniform ring
* the sampled eigenfunctions Y should be full rank => that'll give the sampling theorem
	* the continuous ones form a basis for l^2(R) / S^2
	* the dot product of the discrete versions only depend on the sampling
	* hypothesis: the closer to uniform, the closer they are to a basis for R^n
* orthogonality: on the graph (in R^n) or on the manifold
	* that was a source of confusion (if the SHT was the dot product with the sampled eigenfunctions)
* two extreme samplings of the ring:
	1. uniform: basis, full spectrum
	2. all points at the same position: only the constant frequency can be captured

Next:
* Look at Y^T Y on the non-uniform ring
	* Y^T Y is the dot product on the graph
	* from best (uniform) to worst
* Test the convergence in the setting of Belkin-Nyogi: full graph, kernel width, uniform sampling?
	* it doesn't converge as we do now => why?
	* who said that it's good to set the Gaussian kernel width to the average distance? Shi-Malik? Only relevant for clustering?
* What is the optimal sampling (or its properties) if uniform is not realizable?
	* hypothesis: as close as constant distance between the points?
* continue with the theory
	* what is the relation between the orthogonality of Y and the sampling?
	* Y seems to be our target when searching for U, i.e., discrete dot product should be close to the continuous one
	* Does the optimization scheme proposed by Nath find the optimal U given Y?
		* What's optimal? Conflicting goals: orthogonality (forms a valid Laplacian) and fidelity (close to Y)
