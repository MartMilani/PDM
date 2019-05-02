## 2019-04-30 Meeting sphere

* No meeting next week (Nathanaël and Michaël at ICLR)
* Think about the story / structure you want to give to your project
* For DeepSphere poster:
	* Equivariance through projection: chair in 3D and on the sphere (Frédérick)
	* Better correspondence: eigenvalues, diagonal matrices, subspace alignment (Martino)
	* Equivariance is an unnecessary price: results on SHREC-17 (performance, #parameters, speed)

### Frédérick

Done:
* Literature review
	* 2D CNNs: they don't care about equivariance => more interested in distortions (invariant )
		* Mostly for omni-directional images
	* Cohen and Esteves are the only ones to use the SHT
	* Graphs: Renata (equirectangular), and first paper from Bruna (random sampling)
* Datasets
	* Omni-directional
		* not need to be equivariant to rotation because of gravity
		* No real 360° datasets: either projected 2D, or badly created datasets
	* climate
		* segmentation, large enough, labels (2% not background), equirectangular, 768*1152=1M pixels
		* Papers
			* Segmenting and Tracking Extreme Climate Eventsusing Neural Networks (defined the task)
			* Gauge Equivariant Convolutional Networks and the Icosahedral CNN, Cohen
				* sell it as a scalable alternative to S2CNN
			* Spherical CNNs on unstructured grids
		* ask Jiang to have access to the pre-processing and raw data (for us to change the sampling)
	* panoramic scenes
		* from Stanford indoor
		* same as models, from the outside
* Graph from equiangular sampling
	* better 3D embedding with geodesic than Euclidean distance
* the core issue is that the scalar product on the sphere is not equal to the scalar product in R^n
	* `<x,y>_S^2 = x^t M y != x^t y`, M depends on the sampling (M=I for HEALPix because of equal-area)
	* we know the weights / metric M for the equiangular sampling
		* can compare the eigenvectors with the spherical harmonics
		* can we integrate that knowledge in our construction of the graph / Laplacian?

Next:
* try on ModelNet40 (should be easy to accommodate the code according to Frédérick)
* start on GHCN => we'll need to show non-uniform sampling at some point
* scale the distances in one direction so that the embedded manifold looks like a sphere
* kernel: 1/d or exp(-d) ?

### Charles

Done:
* AID: 2D CNN works better with larger size at last layer
* Global average pooling (77%) better than FC (67%)
	* 3x3 before GAP, 2200 parameters
	* more overfitting before GAP
* Did not find datasets with texture and feature
* Microsoft video: invariance at the end => yeah, pooling is equivariance to invariance
* what are the symmetry groups of the graph convolution?
	* pure graph => permutation of the vertices
	* "geometric graph", i.e., graph as a sampled manifold (here the manifold is the 2D Euclidean space)
		* continuous rotation
		* flip around any axis
		* any other?
		* some permutations don't have equivalent in the Euclidean space
		* local deformations? look at the gauge paper by Cohen, and scattering networks (or further work?) by Bruna & Mallat

Next:
* CIFAR with less data
* test with multiple graphs: horizontal and vertical (factored graph), two vertical graphs (up and down directions)
* main goal: understand the symmetries of graph convolution when the graph is a discretized Euclidean space

### Martino

Done:
* convergence proof for HEALPix
* good news: there is code to generate the FEM Laplacian

Next:
* work on the equiangular sampling
	* our ideas are overfitted to HEALPix, where `<x,y>_S^2 = <x,y>_R^n = x^t y`
	* easier to draw results now that we know what to look for and have good intuitions on HEALPix
* start to play with the FEM Laplacian
