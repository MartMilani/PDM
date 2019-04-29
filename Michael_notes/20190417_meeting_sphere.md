## 2019-04-11 Meeting sphere

### Frédérick

Done:
* we were not using the graphs from Martino due to a bug
	* no idea what improved the convergence speed (only data loading was changed)
* good graphs (correct std and #neighbors) doesn't improve much
* Nside = 8 is still ok, F1 = 78 (vs 81 with Nside=32), Nside = 2 doesn't work anymore
	=> we'll do this experiment in cosmo (Nside=32 will never work)
* reduce overfitting
	* l2 regularization => more l2 gave worse results
	* dropping filters didn't help
	* dropout on last FC helped a little bit
	* similar results by keeping the distance feature only (dropping the other 5 features)
	* add triplet loss, used by Esteves (https://en.wikipedia.org/wiki/Triplet_loss)
		* push the anchor closer to the positive example, and farther from the negative example
		* TF implementation takes embedding and label from current batch
		* F1 increases by 1%
		* loss better suited to the retrieval task
	* augmentation
		* noise did worse (too much noise)
		* 3 translations and 3 rotations => loss on validation augments slower
* conclusion on SHREC-17
	* same performance as Cohen and Esteves (a bit better in classification, less in retrieval)
	  => equivariance to third rotation is useless, invariance is sufficient
		* sklearn accuracy = 80% (78% Cohen), sklearn F1 = 80.5% (78.5% Cohen), script F1 0.705 (0.699 Cohen), mAP 0.675 (0.676 Cohen, 0.685 Esteves)
	* less parameters: 190k (K=4, 6 layers), 400k, 500k
		* can probably reduce ours without much performance loss
	* speed gain: 2h (Cohen 30h, Esteves to be measured)
		* can probably speed up ours without much performance loss
	* we showed that we can perform much faster, at no performance cost
* test rotation invariance
	* random rotations as inputs doesn't improve => confirmation of rotation invariance
* currently constructing out of sphere
* pygsp graphs being prepared

Next:
* summarize the current results (figures, tables), ready for report and presentation
* run Esteves to know how long it takes
* make a performance vs speed / #parameters analysis
	* Cohen and Esteves are two points
	* we have same perf with 6 layers
	* can we be even faster? what do we loose?
* on which dataset should we work next?
	* goal: are there other (fast CNN) we should compare to?
	* survey the existing spherical CNNs (Michaël sends a list of papers)
		* question: does it scale?
		* question: is there an interesting dataset / task?
	* to keep in mind
		* would be nice to show a dense same-equivariant task (SHREC-17 and cosmo are global invariant)
		* demonstrate that we work on multiple samplings, even arbitrary ones (non-uniform sampling density, part of sphere)
	* ideas
		* semantic segmentation of omnni-directional images? (like https://niessner.github.io/Matterport?)
			* that's probably better represented as a cylinder than a sphere (because of gravity)
		* planetary data? such as temperature, humidiy, etc. on the Earth
		* natural spheres S²: planets, atoms, human scalp (EEG, MEG), others?
	* make a proposition of the next task to tackle
* compare healpix vs equirectangular
	* put the spheres (equirectangular, Cohen's equiangular, healpix) in the PyGSP first, we'll look into how good the equirectangular is
	* as we project the shapes ourselves, it's a good dataset to compare multiple samplings

### Charles

Done:
* conclusion on CIFAR-10: as expected, natural image classification is not invariant to rotation
* Test on aerial images
	* 1M parameters => no overfitting yet
	* 20h to train (78 epochs) => 4x slower than 2D CNN
	* a bit better performance than 2D CNN
		* 82% with graph
		* 80% with 2D CNN with same architecture
		* SOTA: 89% (train with 50%), 86% (train with 20%)
	* large images (600 x 600) => down-sample to experiment faster?
	* no clear train and test split (20% testing, 80% training, some did the contrary)
* DeepMind / Dielman paper: equivariance & invariance helps for small models, less so for large models
* adding diagonals doesn't change the results

Next:
* play / tweak a bit to understand better
* do on the grid what Frédérick did on the sphere: show that for a task that is rotation invariant, rotation invariant processing is sufficient
	* to be compared with equivariant processing followed by invariant pooling / summarization
	* three levels: 2D CNN (no special treatment), HexaConv / harmonic networks (rotation equivariant processing), ours (rotation invariant processing)
* look at the datasets from Sander Dielman's work:
	* Plankton (global prediction)
	  https://www.kaggle.com/c/datasciencebowl
	* Galaxies (global prediction)
	  https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge
	* Massachusetts  buildings (dense prediction)
	  https://www.cs.toronto.edu/~vmnih/data/

Long-term:
* on graphs, we can only do same-equivariance and invariance
* we don't choose the symmetries, the domain (graph) gives them to us
	* could those be said to be intrinsic symmetries of the domain?
	* we can "engineer the domain" to get relevant symmetries (e.g., breaking the grid in factor and directions)
* play with other domains, such as the cylinder

### Martino

Done:
* at Nside=1, the pixels are not the icosahedron
	* the faces are faces of the icosahedron, but the pixels are not at the center => likely due to the iso-latitude constraint
* it starts to stop working once the eigenvalues don't form steps (it becomes continuous)
* augmenting Nside diminish the error in the low frequencies, and goes higher in the high frequencies => convergence
	* sigma manually set, full graph thresholded at 0.1
* quite sensible to sigma (a factor two can change quite a bit)
* main issue with the DeepShere way is the number of neighbors, not the sigma
* optimal sigma is not the same for the full or thresholded graph
	* that is strange, as the sparsification is just a numerical approximation
	* cutting distances greater than n sigma is the same as thresholding at a fixed value after the exponential
* sigma is for the convergence, #neighbors for the numerical approximation
* number of neighbors should increase as Nside increases
* sigma is more or less a function of the distance
* looked at the three other Laplacian convergence proofs
	* are they all in the B&N setting (unknown manifold)? yes
	* one added convergence speed to B&N
	* Coifman is more general (similarity on a set instead of manifold), but not as strong (not spectral)
	* last compared the different Laplacians (normalized, combinatorial)

Next:
* Test with sigma as the average distance (as in DeepSphere), but with more neighbors (e.g., 20 instead of 8)
* Can we theoretically motivate an optimal sigma, for a given finite and deterministic sampling?
	* Nath: there is no optimal sigma (just competing objectives)
	* that should ideally lead to an lmax, e.g., lmax = 2Nside, below which the error is smaller than epsilon
	  => sampling theorem
	* probably requires moving away from the B&N setting
		* finite elements / differences? Martino studied that, Nobile can be helpful
		* (discrete) exterior calculus? pointed out by the topologists, see Hirani's thesis "Discrete Exterior Calculus"
		* differential geometry?
		* quadrature formulas? (there was a paper on fast filtering on the sphere with localized filters, not sure there was a proof)
	* main problem: we don't know how to attack that...
* Nath: quantify the error for each frequency (given a sampling and a sigma)
	* inject the fact that we know the manifold to born the error
	* risk: error bound might be large and useless
* study the connection with finite elements / differences
