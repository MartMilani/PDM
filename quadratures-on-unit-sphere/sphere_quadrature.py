'''
Quadrature schemes for integration over the unit sphere in R^3
I = 1/(4*pi) int_[0,2*pi]int_[0,pi] f(theta,phi)sin(phi)dphi dtheta

note that this is a normalised integral (factor 1/(4*pi))

Available quadratures:
    Monte Carlo (clustered nodes, uniform nodes)
    Lebedev quadrature
    Spherical designs
    Cartesian product one-dimensional quadratures (Trapezoidal+Gauss-Legendre)

Literature:
    Monte Carlo
        @book{Press2007,
        address = {New York, NY, USA},
        author = {Press, William H and Teukolsky, Saul A and Vetterling, William T and Flannery, Brian P},
        edition = {3},
        isbn = {0521880688, 9780521880688},
        publisher = {Cambridge University Press},
        title = {{Numerical Recipes 3rd Edition: The Art of Scientific Computing}},
        year = {2007}
        }
    Lebedev (and other papers by Lebedev)
        @article{Lebedev1975,
        author = {Lebedev, V. I.},
        doi = {10.1016/0041-5553(75)90133-0},
        issn = {00415553},
        journal = {USSR Computational Mathematics and Mathematical Physics},
        month = {jan},
        number = {1},
        pages = {44--51},
        title = {{Values of the nodes and weights of ninth to seventeenth order gauss-markov quadrature formulae invariant under the octahedron group with inversion}},
        url = {http://www.sciencedirect.com/science/article/pii/0041555375901330},
        volume = {15},
        year = {1975}
        }
    Spherical designs (overview)
        @article{Brauchart2015,
        author = {Brauchart, Johann S. and Grabner, Peter J.},
        doi = {10.1016/j.jco.2015.02.003},
        issn = {0885064X},
        journal = {Journal of Complexity},
        month = {jun},
        number = {3},
        pages = {293--326},
        title = {{Distributing many points on spheres: Minimal energy and designs}},
        url = {http://www.sciencedirect.com/science/article/pii/S0885064X15000205},
        volume = {31},
        year = {2015}
        }
'''

import numpy as np
from math import pi
import glob

## MONTE CARLO METHOD
# If method = 'cluster' we look at the two 1D integrals over [0,pi]x[0,2pi]
# If method = 'uniform' we look at uniformly distributed points on S^2
def monte_carlo_sphere(func, N=1000, method = 'uniform', **kwargs):
    r = np.random.rand(N,2)
    ## Construct random points
    if method == 'cluster':
        phi = r[:,0]*pi
    else:
        phi = np.arccos(2*r[:,0]-1)
    theta = r[:,1]*2*pi
    ## Integration 
    f_mean = 0
    if method == 'cluster':
        V = 2*pi**2 # 'Volume' of integration domain [0,pi]x[0,2pi]
        for n in range(N):
            f_mean += func(phi[n], theta[n], **kwargs)*np.sin(phi[n])
    else:
        V = 4*pi # 'Volume' of integration domain S^2
        for n in range(N):
            f_mean += func(phi[n], theta[n], **kwargs)
    f_mean = f_mean / N
    I = f_mean * V/(4*pi) # Normalisation MC integral
    ## Variance estimation
    Var = 0
    if method == 'cluster':
        for n in range(N):
            Var += (func(phi[n], theta[n], **kwargs)*np.sin(phi[n])-f_mean)**2
    else:
        for n in range(N):
            Var += (func(phi[n], theta[n], **kwargs)-f_mean)**2
    Var = Var / (N-1) ## Unbiased estimator sample variance
    VarI = Var * V**2 / (N*(4*pi)**2) ## MC integral variance estimator
    return (I, VarI**0.5)

## Lebedev Quadrature
def lebedev(func, N=11, **kwargs):
    coord = np.loadtxt('PointDistFiles/lebedev/lebedev_%03d.txt'% N)
    phi = coord[:,1]*pi/180; theta = coord[:,0]*pi/180 + pi
    w = coord[:,2]
    I = 0
    for (i,x) in enumerate(phi):
        I += w[i]*func(phi[i],theta[i],**kwargs)
    return I

## Spherical t-designs
'''
Three different sets of spherical designs available under switch author
'Hardin': set by Hardin & Sloane up to t=21
'WomersleySym': set by Womersley, symmetrical grid (exact integration odd spherical harmonics), up to t=311
'WomersleySym': set by Womersley, non-symmetrical, up to t=180 
'''
def sph_design(func, t=5, author='Hardin', **kwargs):
    if author == 'WomersleySym':
        if not glob.glob('PointDistFiles/sphdesigns/WomersleySym/ss%03d.*'%t):
            return (float('nan'),float('nan'))
        coord = np.loadtxt(glob.glob('PointDistFiles/sphdesigns/WomersleySym/ss%03d.*'%t)[0])
    elif author == 'WomersleyNonSym':
        if not glob.glob('PointDistFiles/sphdesigns/WomersleyNonSym/sf%03d.*'%t):
            return (float('nan'),float('nan'))
        coord = np.loadtxt(glob.glob('PointDistFiles/sphdesigns/WomersleyNonSym/sf%03d.*'%t)[0])
    else:
        if not glob.glob('PointDistFiles/sphdesigns/HardinSloane/hs%03.*.txt'%t):
            return (float('nan'),float('nan'))
        coord = np.loadtxt(sorted(glob.glob('PointDistFiles/sphdesigns/HardinSloane/hs%03d.txt'%t))[0])
    xx = coord[:,0]; yy = coord[:,1]; zz = coord[:,2]
    phi = np.arccos(zz)
    theta = np.arctan2(yy,xx) + pi
    N = len(phi)
    I = 0
    for (i,x) in enumerate(phi):
        I += func(phi[i],theta[i],**kwargs)
    I = I/N # Normalisation
    return (I,N)

## Gauss-Legendre 1D integration
def gauss_legendre(func, N=40, x_low=0, x_up=pi, **kwargs):
    (x_sample, w) = np.polynomial.legendre.leggauss(N)
    # Transform to [x_low,x_up]
    x_sample = (x_up-x_low)*x_sample/2 + (x_up + x_low)/2
    I = 0
    for (i,x) in enumerate(x_sample):
        I += w[i]*func(x, **kwargs)
    I = (x_up-x_low)/2 * I
    return I

## Gauss 1D integration
def trapezoidal(func, N=40, x_low=0, x_up=pi, **kwargs):
    x_sample = np.linspace(x_low,x_up,N)
    f = np.zeros(N)
    for (i,x) in enumerate(x_sample):
        f[i] = func(x, **kwargs)
    I = np.trapz(f,x_sample)
    return I

## Gaussian product quadrature using Trapezoidal for azimuthal direction and Gauss-Legendre for polar angle
def prod_quad(func, N=20, M=40, **kwargs):
    ## Create function to integrate over sphere in [0,pi]x[0,2pi]
    def func_sphere(phi,theta,**kwargs): return func(phi,theta,**kwargs)*np.sin(phi)
    ## Outer integration over theta using trapezoidal
    I = trapezoidal(lambda x: gauss_legendre(func_sphere, N=N,x_low=0,x_up=pi,theta=x,**kwargs),
            N=M,x_low=0,x_up=2*pi)
    ## Return normalised integral
    return I/(4*pi)
