from sphere_quadrature import *
import numpy as np
import matplotlib.pyplot as plt

cos = np.cos; sin = np.sin; tanh = np.tanh; exp = np.exp;
## Test functions
x = lambda phi,theta: sin(phi)*cos(theta) 
y = lambda phi,theta: sin(phi)*sin(theta)
z = lambda phi,theta: cos(phi)
def f_1(p,t,a=1):
    return 1 + x(p,t) + y(p,t)**2 + x(p,t)**2*y(p,t) \
        +x(p,t)**4+y(p,t)**5+x(p,t)**2*y(p,t)**2*z(p,t)**2

def f_2(p,t,a=1):
    return 0.75*exp(-0.25*((9*x(p,t)-2)**2+(9*y(p,t)-2)**2+(9*z(p,t)-2)**2)) \
        + 0.75*exp(-((9*x(p,t)+1)**2/49+(9*y(p,t)+1)/10+(9*z(p,t)+1)/10)) \
        + 0.5*exp(-0.25*((9*x(p,t)-7)**2+(9*y(p,t)-3)**2+(9*z(p,t)-5)**2)) \
        - 0.2*exp(-((9*x(p,t)-4)**2+(9*y(p,t)-7)**2+(9*z(p,t)-5)**2))

def f_3(p,t,a=9):
    return (1+tanh(-a*(x(p,t)+y(p,t)-z(p,t))))/(1.0*a)
    
def f_4(p,t,a=9):
    return (1+np.sign(-a*(x(p,t)+(y(p,t)-z(p,t)))))/(1.0*a)

def f_5(p,t,a=9):
    return (1+np.sign(-a*(pi*x(p,t)+y(p,t))))/(1.0*a)

a = 12 # Function parameter
## Exact integral values
f_1_exact = 216*pi/35
f_2_exact = 6.6961822200736179253
f_3_exact = 4*pi/a; f_4_exact = 4*pi/a; f_5_exact = 4*pi/a

## Function to test
f_test = f_2
f_exact = f_2_exact/(4*pi)

## Monte Carlo
N_mc = np.ceil(np.logspace(1,4,10)).astype(np.int64)
E_mc1 = np.zeros(len(N_mc))
E_mc2 = np.zeros(len(N_mc))
for (i,n) in enumerate(N_mc):
    E_mc1[i] = abs(f_exact-monte_carlo_sphere(f_test,n,method='uniform',a=a)[0])
    E_mc2[i] = abs(f_exact-monte_carlo_sphere(f_test,n,method='cluster',a=a)[0])

## Lebedev testing
n_leb = np.concatenate((range(3,33,2),[35,41,47,53,59,65,71,77,83,89,95,101,107,113,119,125,131]))
E_leb = np.zeros(len(n_leb))
N_leb = np.zeros(len(n_leb))
for (i,n) in enumerate(n_leb):
    coord = np.loadtxt('PointDistFiles/lebedev/lebedev_%03d.txt'% n)
    N_leb[i] = len(coord[:,1])
    E_leb[i] = abs(f_exact-lebedev(f_test,n,a=a))

## Spherical design testing
t_sph = np.array(range(1,100))
E_sph = np.zeros(len(t_sph))
N_sph = np.zeros(len(t_sph))
for (i,t) in enumerate(t_sph):
    (f_sph,N) = sph_design(f_test,t,author='WomersleySym',a=a)
    E_sph[i] = abs(f_exact-f_sph); 
    if E_sph[i] == 0.0: E_sph[i] = 1e-16
    N_sph[i] = N
E_sph = E_sph[~np.isnan(N_sph)]
N_sph = N_sph[~np.isnan(N_sph)]; E_sph = E_sph[np.argsort(N_sph)]; N_sph = np.sort(N_sph)

## Cartesian product testing
M_prod = np.arange(2,74,14);
N_prod1 = 2*M_prod**2; N_prod2 = 2*M_prod*(M_prod+1)
E_prod1 = np.zeros(len(M_prod)); E_prod2 = np.zeros(len(M_prod))
for (i,m) in enumerate(M_prod):
    E_prod1[i] = abs(f_exact-prod_quad(f_test,m,2*m,a=a))
    E_prod2[i] = abs(f_exact-prod_quad(f_test,m,2*m-1,a=a))

## Plotting results
plt.loglog(N_prod1,E_prod1/abs(f_exact),'b-x',lw=3,label=r'Gaussian product ($2M$)')
plt.loglog(N_prod2,E_prod2/abs(f_exact),'b--x',lw=3,label=r'Gaussian product ($2M$+1)')
plt.loglog(N_leb,E_leb/abs(f_exact),'r-x',lw=3,label=r'Lebedev')
plt.loglog(N_sph,E_sph/abs(f_exact),'k-x',lw=3,label=r'Spherical design')
plt.loglog(N_mc,E_mc1/abs(f_exact),'g-x',lw=3,label=r'Monte Carlo 1')
plt.loglog(N_mc,E_mc2/abs(f_exact),'g--x',lw=3,label=r'Monte Carlo 2')
plt.xlabel(r'$N$')
plt.ylabel(r'$e['+f_test.__name__+']$')
plt.legend(loc='upper right')
plt.ylim([1e-17,1e0])
plt.show()

# Possibility to export figure to Tikz
#import matplotlib2tikz
#matplotlib2tikz.save('../Graphics/'+f_test.__name__+'.tex', figurewidth='\\figurewidth')

''' TEST FUNCTION CONTOUR PLOTTING (MAYAVI NEEDED)
from mayavi import mlab
# Plotting contours of test function
phi, theta = np.mgrid[0:pi:251j, 0:2*pi:251j]
X = x(phi,theta); Y = y(phi,theta); Z = z(phi,theta)
mlab.figure(1,fgcolor=(0,0,0),bgcolor=(1,1,1),size=(800,700))
#    phi, theta = np.mgrid[0:pi:201j, 0:2*pi:201j]
#    X = x(phi,theta); Y = y(phi,theta); Z = z(phi,theta)
contours = mlab.mesh(X,Y,Z,scalars=f_test(phi,theta,a=a),colormap='Set1')
setattr(contours,'enable_contours',True)
contours.update_pipeline()
mlab.mesh(X[::10],Y[::10],Z[::10],color=(0.85,0.85,0.85))
ax=mlab.orientation_axes()
ax.text_property.italic=True
ax.text_property.font_size=80
ax.text_property.font_family='times'
ax.axes.normalized_label_position = [2,2,2]
mlab.show()
'''
