# Meeting Deepsphere 22.05.19


### Charles

Done:
* Random walk matrix VS graph laplacian (similar performance on undirected graph)
* Adding directed graphs for the vertical axis (undirected on the horizontal) -> improve performance, but still worse than traditional CNN
* 4 directed graphs -> improve slightly the performance but not much more, still lower than a CNN

* For HEID, add direction does not help

Conclusion isotropy VS anisotropy is good/bad depending on the data

Next:
* Start with the blogpost, structure + figures
* Augment the number of parameters of the 4 graphs to get to the level of the 2D CNN.
* Compare mean VS concatenation to check invariance VS equivariance


### Martino

Done:
* Making a spectral analysis of the equirectangular sampling
	- Using a full graph with Gaussian kernel, there is some alignment, at least the for small l.
	- Using the FEM, the alignment is very good!

FEM is very promising for irregular sampling

Next:
* Can we threshold L' = B^-1 L, as we do for L_graph?  -> TRY MASS LUMPING
	* When is B needed? When is a dense L' necessary?
* Can we compute g(B^{-1}L) by only using B and L? Can we find a scalable algorithm?
* Study this: g(B^{-1}L) \neq g_1(B^{-1}) g_2(L) = g_3(B) g_2(L), where g_3(x) = g_1(1/x)
* Write a plan/structure/story for the report -> what is missing?


### Frédéric

Done:
* Why are we losing so much performance sur Modelnet 40 when we do some rotation of the input?
	- Rotations of 90 degrees give exactly the same result for DeepSphere
	- The mistake comes very likely from the fact that the classes are very similar when projected onto the sphere. Example a desk is very close to a table when projected. The sphere is not giving us a stable representation for the object.
	- If we augment our dataset with rotation, we drop way less...
	- Without augmentation, we drop 20%
	- If we augment only one rotation, (setting of Esteves), we drop as much as Esteves. He has a different sampling.
* GHCN
	- Problems of interpolation -> after interpolation, we have some T_min> T_max
	- It is hard to make a meaningful problem.
		- He did from T_min to T_max (I believe it is not a very good problem)
		- Temperature to precipitation does not work
		- Guess future temperatures -> predict day T instead of T+1
	- Problem what task could we solve???? Not enough data. @Michael

Next:
* Work on the climate patterns segmentation problem
* Find another dataset with an irregular sampling
