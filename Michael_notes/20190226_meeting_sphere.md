## 2019-02-26 Meeting sphere

### Frédérick

* SHREC'17 is currently being transformed to a spherical representation on the HEALPix sampling
	* uses the code from Cohen => dist, sin, cos on the surface and convex hull (6 features)
* next: adapt DeepSphere to take this data rather than the cosmological maps
* use the metrics from Cohen or Esteves
	* be sure we can compare to their published results and the SOTA at https://shapenet.cs.stanford.edu/shrec17/#results
* verbatim DeepSphere doesn't run on the GDK
	* due to CUDA update?
	* test again on my environment

### Charles

* plutôt lu (et bossé sur la PyGSP)
* currently following a pytorch tutorial
* start by implementing a 2D CNN
* choose an architecture
	* not too large (training time and memory)
	* close to SOTA
	* tried-and-tested (not our own)
	* look at https://paperswithcode.com/task/image-classification
* dataset: CIFAR-10 or CIFAR-100
	* MNIST is too easy and ImageNet is too large
* give him access to GPU machine

### Martino

* Martino: what to put as the eigenvalues?
* there's a thm that says that the diagonalization of a circulant matrix results in the classical Fourier basis
	* https://en.wikipedia.org/wiki/Circulant_matrix
	* probably an iff => no way to get the classical Fourier basis without a circulant matrix
* the basis formed by the sampled sines and cosines is not orthogonal if the sampling is not uniform
	* not orthogonal in the Euclidean space, but with a metric
* Nath's optimization problem: min_W || LU - L Lambda ||
	* linear operator from W to L
	* redundancy of Lambda and L (same information in a different basis) => they help the optimization
		* also, U is not a simple change of basis as it's not necessarily orthonormal
	* will give the closest orthonormal U to the desired U
	* find the closest orthonormal Fourier basis: min_U || U - V || s.t. U^T U = I
* by trying to align the eigenspaces, we lose the eigenvalues
* what if we give the Lambda with the non-orthonormal U?
	* not sure that will give a valid L
	* if not, remove the negative weights in the wrong L => gives the closest orthonormal U?
