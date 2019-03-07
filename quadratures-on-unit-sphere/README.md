# README #

### What is this repository for? ###

* Python implementation of several quadrature schemes for integrals on the unit sphere in R^3. See [essay](http://people.maths.ox.ac.uk/beentjes/Essays/) for a more detailed exposure.
* Current quadrature schemes:

1. Lebedev quadrature
2. Spherical designs
3. Cartesian product of a trapezoidal and Gauss-Legendre quadrature
4. Monte Carlo quadrature

### How do I get set up? ###

* Quadrature schemes can be loaded from sphere_quadrature.py
* In order to use spherical designs or Lebedev quadrature run bash script get_quadrature_grids.sh to automatically download the quadrature grids from the different online sources (+/-250 MB, could take a while)
* To use the 3D visualisation of the quadrature grids [Mayavi](http://docs.enthought.com/mayavi/mayavi/) needs to be installed

### How do I use the Python scripts? ###

* sphere_quadrature: contains the quadrature 
* point_dist: visualisation of the quadrature grids on the unit sphere
* efficiency: plot of McLaren efficiency measure
* test_functions: test suite of several functions to see relative convergence of quadratures used in [essay](http://people.maths.ox.ac.uk/beentjes/Essays/)

### Who do I talk to? ###

* Casper Beentjes [beentjes@maths.ox.ac.uk](mailto:beentjes@maths.ox.ac.uk)