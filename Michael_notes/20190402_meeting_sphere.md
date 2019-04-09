## 2019-04-02 Meeting sphere

### Frédérick

* modified the code to use less RAM
	* there's a bug as we don't get the same results
	* needed for larger Nside => much more samples
* more depth (3 layers to 5 and 6) and K=5, augmentation (only translation)
	* 80% on validation set
	* does not beat Cohen / Esteves yet
	* overfit much more easily
	* l2 regulation doesn't seem to change much
	* validation loss starts to augment, while accuracy and f1 keeps going done
		* Nath: can be due to the network being over-confident => wrong hard predictions are much penalized by loss but not accuracy
* we currently use BatchNorm and Dropout (1%)
	* dropout only on the FC (which is 100 x 55) => shouldn't have much impact
	* do dropout on the feature maps? we do that already on the last layer
	* spatial dropout? Drop pixels.
	* For sure, don't drop the learned polynomial weights.
* multiple definitions of the equirectangular sampling
	* simplest: equidistance in theta, phi (what I did)
	* Cohen uses another equirectangular sampling: angles between rings are not constant?
	* topologically connected like a cylinder, with vertices from the last rings connected to the ones in front

Next:
* do the pre-processing in the training pipeline? no, keep the two datasets
	* use tf.data, see https://www.tensorflow.org/guide/datasets
	* mesh to spherical signal (with augmentation) takes 0.5 seconds => way too slow
	* 31k shapes => 93k spherical signals (with 3 random rotations and translations for each), 31k signals (without augmentation)
* project from the outside of the sphere?
	* more difficult problem: omniscient observer => point-wise observer
	* similar setting as omni-directional imaging and cosmology
	* could be interesting to compare those two settings on the same data
* more depth, smaller filters, more aggressive coarsening / pooling
	* it's a global task, we want to go down to a size of 1, so go smoother (e.g., Nside = 64, 32, 16, 8, 4, 2, 1)
* put the spheres (equirectangular, Cohen's equiangular, healpix) in the PyGSP

### Charles

* works better without dropout on convolutions (dropout was don across space and features)
* Nath's scale (0.8 instead of 1) => doesn't converge faster, but can go higher
	* to be confirmed, might be due to chance
	* Nath: without the scale, filters are mostly combinations of diracs and low-pass
* accuracy is now at 71% (drop from 74% might be due to not doing augmentation anymore)
* slow training: 4-5 hours to get a feeling on K40
	* as a test for renku?
	* Nath might have some spare hours at CSCS
	* Google colab? free GTX 1080 / TPU
	* Kaggle? free GTX 1080
	* pytorch geometric: optimized GPU kernel
* hexagons
	* found the code

Next:
* continue on the points from last week

### Martino

* depth valley in sparsity
	* Nath: problem is that there is two parameters at play (number of neighbors, and sigma)
	* correct distance with geodesic distances => same effect
	* Nside = 8 => sigma = 1 => weight = 0.1 on the other side of the sphere => way too large
	* sigma should be linked to the number of neighbors?
	* tweak the constant in front of sigma by hand
	* look at the distribution of edge weights => by sparsifying, we cut the tail
		* if the tail is at 0.1, that's too much
		* sparsification is really a numercial approximation: droping the small weights in the computation of a weighted sum
		* maybe use a threshold instead of kNN
	* large sigma to capture well the low frequencies, small sigma needed to capture higher frequencies
		* at the limit of infinite sigma => fully connected graph, cannot know where we are, all the eigenvalues are merged
* study of special samplings (platonic solids) still to be done
* minimum radius
	* not divided by 2
	* found a formula from reverse-engineering HEALPix code
	* ask the authors
	* continue by assuming the formula is true
* meeting Nobile
	* two steps: see if he can point us to what they've done in their field on the general idea, if not, go more specific

Next:
* test how a fully connected graph with multiple sigmas converges DONE
* finish the B&N pointwise convergence proof TODO
* play with the platonic solids: those are the sole 5 samplings that are regular TODO
