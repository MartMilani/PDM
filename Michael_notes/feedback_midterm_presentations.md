# Feedback on midterm presentations

All of you:
* look more at the audience, not at the slides
* figures: larger fonts, legends, labeled axes

## Charles

* time: 44 to 9 => 25 minutes (with questions)
* good explanation of equivariance
* you could say that the orientation is arbitrary for some datasets
* good: more weight sharing by exploiting more symmetries
* citations: [Cohen et al., 2018] for example
* pooling for invariance => mention that we do that for translation on images already
* group convolution cons: more params to learn?
* log-polar represention: rotation becomes shift
* work more on the pros & cons
	* slow => depends compared to what (2D conv is well optimized) for example it's much faster on the sphere)
* larger fonts for figures
* architecture: didn't we use global average poolin (GAP)?
* mention the task: classification of images
* view from top => images with rotational symmetry
	* give some examples: biology, aerial images, something else?
* number of examples: CIFAR is 60k, AID is ??
	* with enough data, the NN will learn the equivariance
* add the example of the rabbit becoming a duck only with rotation

## Frédérick

* time: 12 to 37 => 25 minutes
* equivariant in translation => to translation
* less text on the slides
* DeepSphere advantages: faster, but also more flexible (part of sphere, any sampling, non-uniform samplings)
* good discussion of the sampling schemes
* SHREC-17
	* explain the task (show some examples of classes)
	* why do you use a sphere to represent 3D models?
	* briefly explain what are the Cohen and Esteves models
* good to have a quick summary / take-home message
	* equivariance is not necessary => invariance is sufficient => equivariance is an unnecessary price to pay
* training curves: export from TF, plot with matplotlib with proper labels, titles, axis, etc.
* in addition, or maybe rather than, training curves, show test accuracy (that's a single number to draw a conclusion from)
* good conclusion (sufficiently equivariant, info in low frequencies)
	* I would repeat the main point of the quick summary (equivariance can be an unnecessary price to pay)
* "next steps" => "future work"

## Martino

* time: 40 to 7 => 27 minutes
* we don't need L^k to be sparse as we can compute L^k x recursively, i.e., we never compute L^k
* slide 10: why fixed number of neighbors? => computational efficiency
* show the spherical harmonics for people to understand why eigenvalues are grouped
* Defferrard et al. => Nathanaël et al. ;-) (or put both)
* define N_side
* try to put less text
* try to be more visual (for example, show the spherical harmonics, the FEM test functions, etc.)
