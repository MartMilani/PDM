## 2019-04-25 Meeting sphere

### Charles

* test with structure (generated shapes) and texture (texture bank)
* test with a mix of invariant and 2D CNN layers

### Frédérick

* make it work on the equirectangular sampling
	* work with Martino to build a good graph
	* look at the 3D embedding of the sphere => it should give intuitions on how to set the weights
* which task / dataset next?
	* Martino: medical spherical images
	* omni-directional images: mostly projected images
* SHREC-17: one sphere signal per intersection to alleviate shadowing

### Martino

* check Frédérick's equirectangular graph
* beyond the Heat kernel Laplacian [Belkin, Nyiogy] approximation of the Laplace-Beltrami
	* FEM Laplacian => much better approximation of the spectrum
	* geometric Laplacian
* our only result so far is to have empirically improved the DeepSphere graph, and a better understanding of the importance of sigma
* it would be nice to have a practical recipe: if we don't know the manifold (hence we cannot check the eigenvectors), how to set sigma?
