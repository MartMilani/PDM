import matplotlib.pyplot as plt
import numpy as np


def kernel(x, t):
    return np.exp(-x**2/(4*t))


xx = np.linspace(0, 2, 100)
t = 0.5
plt.plot(xx, kernel(xx, t))
for x in np.linspace(0, 2, 11):
    if x:
        """if x<0.3:
            plt.plot([x, x], [0, kernel(x, t)], 'r--')"""

        plt.plot([x, x], [0, kernel(x, t)], 'b--')
plt.show()
