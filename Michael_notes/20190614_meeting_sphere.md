## 2019-06-04 Meeting sphere

### Frédérick

* mostly report
	* benchmark part almost done: SHEC'17, ModelNet40, GHCN
	* did not have time for Extreme weather
* baseline of predicting the current day: MSE goes from 9 => 134
* 8.2 with order 5, 10.88 with order 0

Next:
* global regression: predict the time-of-year
* continue with the report

### Charles

* Did the blog post

Next:
* Clean the code and polish the repo. Make it public.
* Prepare the presentation.

For the summer, after the project:
* Test the equivariant architecture

Feedback blog:
* Suggested title: Exploiting symmetries (in images) with graph neural networks
* Style needs improvement.
* Good figures.
* You could mention equivariance to localization (generalization of translation).
* AID is not anisotropic. The classification task is invariant to rotation. Alternatively, the orientation of the images is arbitrary.
* Good story.

### Martino

* worked on the report
* lumped mass is proportional to the pixel area
* lumping breaks the correspondence of the FEM Laplacian
* graph Laplacian on equiangular is even worse
* equivariance error: 10^-8 relative error for graph, 0.1 for FEM
	* rotation around the south-north axis is perfect on graph (that is the only discrete symmetry)

Next:
* Normalize the eigenvalues in [0, 1] before evaluating the filtering function.
* Even less uniform than equiangular? random sampling?
* Measure the equivariance error with continuous signals (synthesized in the spherical spectral domain).
	* Rotation in the spectrum is given by Wiener D-function.
	* Not sure if worth it.
* Another example filtering: denoising with smoothness prior.
* Future direction for FEM: filtering (in the signal processing sense) is limited to functions of the Laplacian. FEM can solve any PDE.
