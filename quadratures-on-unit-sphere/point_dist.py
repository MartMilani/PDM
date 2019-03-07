## Visualisation of the different quadrature grids on the sphere in R^3
from mayavi import mlab
import numpy as np
import pickle
import pdb

## Function to add orientation axis to the visualisation
def add_axes():
    fig = mlab.gcf()
    ax = mlab.orientation_axes(fig.children[0])
    ax.text_property.italic=True
    ax.text_property.font_size=80
    ax.text_property.font_family='times'
    ax.axes.normalized_label_position = [2,2,2]

## Plot parameters
pt_size = 0.05 # point size on the sphere
sphere_color = (0.75,0.75,0.75)
## Create a unit sphere in R^3
r = 1.0
pi = np.pi; cos = np.cos; sin = np.sin
phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j] # complex step to make endpoint inclusive
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)

## Create random points wrongly clustered, [0,pi]x[0,2pi] distrubtion
f1=mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
tmp = mlab.mesh(x , y , z, color=sphere_color,figure=f1)

N = 240 #Number of points
rand = np.random.rand(N,2)
phi = rand[:,0]*pi
theta = rand[:,1]*2*pi

xx = r*sin(phi)*cos(theta)
yy= r*sin(phi)*sin(theta)
zz = r*cos(phi)

mlab.points3d(xx, yy, zz, scale_factor=pt_size)
add_axes()

## Create random points uniformly distributed, S^2 distribution
f2=mlab.figure(2, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
mlab.mesh(x , y , z, color=sphere_color)

N = 240
rand = np.random.rand(N,2)
phi = np.arccos(2*rand[:,0]-1)
theta = rand[:,1]*2*pi

xx = r*sin(phi)*cos(theta)
yy= r*sin(phi)*sin(theta)
zz = r*cos(phi)

mlab.points3d(xx, yy, zz, scale_factor=pt_size)
add_axes()

## Create points such that equispaced in azimuthal angle and Gauss-Legendre in polar angle
f3=mlab.figure(3, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
mlab.mesh(x , y , z, color=sphere_color)

N = 3; M = 2*N+1
phi = np.repeat(0.5*pi*np.polynomial.legendre.leggauss(N)[0] + 0.5*pi, M)
theta = np.tile(np.linspace(0,2*pi,M),N)

xx = r*sin(phi)*cos(theta)
yy= r*sin(phi)*sin(theta)
zz = r*cos(phi)

mlab.points3d(xx, yy, zz, scale_factor=pt_size)
add_axes()

## Create points for spherical t-designs (includes Platonic solids)
f5=mlab.figure(4, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
mlab.mesh(x , y , z, color=sphere_color)

N = 1130 # Number of points in spherical design
t = 47 # Order of spherical design
## Load one of the following possibilities
#coord = np.loadtxt('PointDistFiles/sphdesigns/HardinSloane/hs%03d.%05d'%(t,N))
#coord = np.loadtxt('PointDistFiles/sphdesigns/WomersleyNonSym/sf%03d.%05d'%(t,N))
coord = np.loadtxt('PointDistFiles/sphdesigns/WomersleySym/ss%03d.%05d'%(t,N))

xx = coord[:,0]; yy = coord[:,1]; zz = coord[:,2]
mlab.points3d(xx, yy, zz, scale_factor=pt_size)
add_axes()

## Create points for Lebedev quadratures
f6=mlab.figure(5, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
mlab.mesh(x , y , z, color=sphere_color,figure=f6)

N = 7
coord = np.loadtxt('PointDistFiles/lebedev/lebedev_%03d.txt'% N)
phi = coord[:,1]*pi/180; theta = coord[:,0]*pi/180

xx = r*sin(phi)*cos(theta)
yy= r*sin(phi)*sin(theta)
zz = r*cos(phi)

mlab.points3d(xx, yy, zz, scale_factor=pt_size)
add_axes()

## Only view a subset of the options
N = [1,2,3,4]
for n in N:
    mlab.close(n)

mlab.show()
