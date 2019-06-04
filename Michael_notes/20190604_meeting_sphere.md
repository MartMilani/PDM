## 2019-06-04 Meeting sphere

### Frédérick

* Data:
	* 1573 stations have temperature every day for 4 years
	* superset 12k
	* graph built from Euclidean distance
* Check if structure is helping
	* future T from past n temps
	* 84% for order 0 => 73% for order 4
	* mean absolute error: 2.4°C => 2.1 °C
	* R^2 (https://en.wikipedia.org/wiki/Coefficient_of_determination): 0.89 => 0.92
* How to evaluate?
	* predicting 100 if the value varies from 97 and 103 is a small relative error, but a big mistake
	* we can only predict statistical fluctuations, not the real future
* Extreme event dataset:
	* We don't have the same dataset, but another one derived from CAM5

Next:
* do it for multiple orders, make a plot
* compare with a very simple baseline (like predicting the same temperature as the day before)
* play a bit to try to improve
* Nath could launch the climate extreme event dataset on CSCS.
* Send Alexandre's code to Frédérick
* prioritize the report

### Charles

* Started to write the blog post
* Fixing the non-equivariant convolution

Next:
* be sure that the NN is truly equivariant
* continue on the blog post

## Martino

* Made progress on the report
	* Write first what he did, then context
* FEM filtering
	* diffusion with the lumped matrix is quite bad on HEALPix

Next:
* Measure the equivariance error (small experiment to validate the work).
	* That experiment was done by Cohen as well.
* Plot the lumped mass matrix on the sphere to see where the small and large values are.
	* What is the interpretation of this matrix? Does it have to do with the pixel's areas?
	* Prepare a notebook that shows the current result. We'll look at it at AIcosmo.
