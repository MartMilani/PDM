Meeting deepshere 28 May 2019


# Frederic

### What has been done?
* Found the extreme weather dataset
	- https://extremeweatherdataset.github.io/
* Did not find another dataset with irregular sampling...
* Work on the report

### Next week
* Preparation of the extreme weather dataset to run with interpolation at Nside=32
* Report


# Charles

### What has been done?
* Plan for the report
	https://docs.google.com/document/d/1ATEYPlUrEjIJ8KccP4Rxj25fFkKTKwGCjSiG3XGGBnA/edit
* Problem: the network with 4 graphs is not invariant to rotation
	Nathanael: probably, the problem comes from the filter inter graphs -> do not concatenate the outputs of the different graphs

### Next week
* Go on with the blog post
* Try to fix the equivariance


# Martino

### What has been done?
* Storyline
* Discussion with 
	- FEM allows for computation outside of the original sampling
* Filtrage avec B^{-1}L?
	- Mass lumping: approximate B^{-1} with a diagonal matrix B_d...
	- Then filter with L'= B_d^{-1/2} L B_d^{-1/2}


### Next week
	- Check the filtering with FEM
	- Go further with the report
