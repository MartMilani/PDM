## Short script to reshape coordinates from Neil Sloane's spherical design

import numpy as np
import glob

for file in glob.glob('PointDistFiles/sphdesigns/HardinSloane/hs*'):
    coord = np.loadtxt(file)
    coord = np.reshape(coord, (int(len(coord)/3), 3))
    np.savetxt(file, coord)
