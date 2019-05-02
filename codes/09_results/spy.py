import scipy.sparse
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt


sparse_matrix = scipy.sparse.load_npz('stiffnessmatrix_16.npz')
plt.spy(sparse_matrix)
plt.show()
