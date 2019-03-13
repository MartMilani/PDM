## 2019-03-05 Meeting sphere

### Charles

Done:
* Trained on CIFAR-10
* Difficulties to get SOTA: around 90% where SOTA is around 96%
* Results often reported with data augmentation
	* horizontal flip
	* random crop

Next
* smaller net for speed (VGG has 11 layers)
* no augmentation at first: we want to compare the 2D CNN with the graph CNN on the simplest architecture
	* then we can investigate various tricks
* goal: try with GraphConv instead of 2DConv as soon as possible

### Frédérick

Done:
* DeepSphere runs on SHREC-17
	* 72% accuracy on validation (Cohen is around 0.7 too)
	* same architecture: 3 GraphConv + FC
* Differences: BN, number of feature maps per layer, size of the filters
	* easy to set the number of feature maps to the same
	* two ways to compare filter sizes: same spatial extent or same number of parameters
	* pooling: Cohen does bandwidths 64 => `(2*64)^2=16k` to 16 => `(2*16)^2=1k` to 10 => `(2*10)^2=400` pixels
		* the closest we can do is Nside 32 => `12*32^2=12k` to 8 => `12*8^2=768` to 4 => `12*8^2=192` pixels
		* cannot set them equal
* Data
	* Cohen and Esteves: grid 4x 64^2 = 16k pixels => corresponds to Nside = 32 => 12k pixels
	* object is at the center of the sphere
	* augmentation: random rotation and translation before projecting on the sphere
		* not done yet for DeepSphere => either do it for DeepShere or don't do it for them (ideally both)
		* no augmentation for them is currently running
* simplified model on their github: they did it afterwards and claim it has the same performance

Next:
* evaluation with nodejs script
* compare with Cohen and Esteves on the same data (features ok, augmentation) and same architecture (number of parameters, BN, pooling, etc.)
* tweak our architecture and try to beat them
* compare computational time (seconds per epoch or per sample) => we should beat them easily here
* make DeepShere work on the equirectangular sampling
	* easier than theirs on HEALPix
	* will be an asset to democratize its usage (the more sampling schemes we provide out-of-the-box, the better)
	* how to build the graph? NN or Renata style?

Longer term (next experience after SHREC-17):
* large cosmo experience => will not scale (full sphere is 12M points, feeding partial sky to their CNN will be painful)
* try the small cosmo experience (100 maps of Nside = 64 => 16k pixels)
	* can control difficulty by adding random noise
	* need to transform the data to the equirectangular sampling for the spherical CNNs
* weather non-uniform sampling: do we even really want the manifold to be a sphere? need more theoretical understanding
* climate data

### Martino

Done:
* used a solver to solve the problem proposed by Nath
	* problem is indeed quadratic / linear
* found the weights for a regular ring
* infinite number of solutions: every circulant matrix will have the DFT as its basis
	* even: return a banded structure
	* odd: full matrix
* conclusion: solver works! Will be a useful building block
* read the three papers
	* Pasdeloup, Gripon, Rabbat: diffuse white noise, the covariance matrix will have the same eigendecomposition as the Laplacian
		* In the end, the problem is to find the eigenvalues => characterized the space of possible eigenvalues as a convex polytop
		* For their problem in practice, you never know exactly the eigenvectors (signals used to create the graph are noisy)
	* Shafipour, Segarra, Marques: similar approach, but not with white noise
	* Vassilis & Nath

Next:
* try the algorithm on a non-uniformly sampled ring
	* set the goal as the sampled sines and cosines => harder as this basis is not orthonormal
* what is the basis we want to optimize for?
	* we now know it's not the sampled eigenfunctions
	* what we want: the dot product between the sampled signal and the elements of the basis gives the FT of the continuous signal
	* should maybe integrate some measure (it's probably some sort of weighting scheme)
* study more the quadrature formulae (on ring or sphere)

Long term goal:
* exact correspondence between the dot product and continuous integral
* if impossible, a convergence result as we increase the number of vertices
