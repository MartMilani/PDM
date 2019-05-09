import scipy
import numpy as np
from scipy.optimize import minimize
from scipy import spatial
import seaborn as sns
import matplotlib.pyplot as plt


__SAVE_FIG__ = True
__FIG_DIR__ = '04_figs/'
__FORMAT__ = 'eps'


def laplacian(W):
    return np.diag(np.sum(W, axis=0)) - W


def normalized_laplacian(W):
    N = np.alen(W) + 1
    D_diag = np.sum(W, axis=0)
    D_minus12 = np.diag([d**(-0.5) for d in D_diag])
    return np.eye(N - 1) - D_minus12@W@D_minus12


def buildW(flat_w_sup, N):
    W = np.zeros((N, N))
    shift = 0
    for i in range(N-1):
        row_length = N-i-1
        W[i, i+1:] = flat_w_sup[shift:shift+row_length]
        shift += row_length
    W = W+W.T
    return W


def O_LU(W, U):
    """W is a matrix NxN"""
    L = laplacian(W)
    return L@U


def O_LambdaU(lambda_, U):
    return U @ np.diag(lambda_)


def loss_function(Wsup_and_lambda, U, N):
    flat_w_sup = Wsup_and_lambda[:-N]
    W = buildW(flat_w_sup, N)
    lambda_ = Wsup_and_lambda[-N:]
    return np.linalg.norm(O_LU(W, U) - O_LambdaU(lambda_, U), ord='fro')


def harmonics(thetas):
    N = len(thetas)
    assert N % 2 == 1, 'N has to be odd to construct the last eigenvector'
    U = np.zeros((N, N))
    k = 0
    for i in range(N):
        if i == 0:
            U[:, 0] = 1. / (np.sqrt(2)*np.pi)
        elif i % 2:
            k += 1
            U[:, i] = np.sin(k * thetas) / (np.pi)
        else:
            U[:, i] = np.cos(k * thetas) / (np.pi)
    return U


def main():

    # number of vertices on the ring graph
    # WARNING: it has to be an odd number for the construction of the correct harmonics
    N = 21

    # building the pointset
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    thetas += np.random.rand(N)/(30*np.pi/N)
    thetas.sort()
    plt.plot(thetas)
    plt.show()
    assert len(thetas) == N, len(thetas)
    pointset = np.zeros((N, 2))
    pointset[:, 0] = np.cos(thetas)
    pointset[:, 1] = np.sin(thetas)
    plt.plot(pointset[:, 0], pointset[:, 1], 'ro')
    plt.axis('equal')
    title = 'Sampling'
    plt.title(title)
    if __SAVE_FIG__:
        plt.savefig(__FIG_DIR__+title, format=__FORMAT__)
    plt.show()

    # buiding the initial graph
    distance_matrix = spatial.distance.cdist(pointset, pointset)
    W0 = np.exp(-distance_matrix / np.mean(distance_matrix))
    for i in range(np.alen(W0)):
        W0[i, i] = 0
    print("Initial adjacency matrix:\n", W0)

    W0 = np.random.rand(N, N)

    # calculating the circle armonics of the circle
    U = harmonics(thetas)

    # storing the upper triangular adjacency matrix in a flat vector
    w = np.ndarray(0)
    for i in range(np.alen(W0)-1):
        w = np.append(w, W0[i, i+1:])
    assert len(w) == N*(N-1)/2

    # initializing the lambdas
    lambdas = np.ones(N)/N

    # initializing the vector of w, lambdas
    X0 = np.concatenate([w, lambdas])

    # defining the bounds on the variables w, lambda
    lb = np.zeros(len(w) + N)
    ub = np.ones(len(w) + N) * np.inf
    bounds = scipy.optimize.Bounds(lb=lb, ub=ub)

    # defining the constraint on the lambdas
    A = np.zeros(len(w) + N)
    A[len(w):] = 1
    b = 1
    cons = [{"type": "eq", "fun": lambda x: A @ x - b}]  # Ax == b

    # performing the minimization step
    result = minimize(loss_function, X0, args=(U, N), bounds=bounds,
                      constraints=cons)

    # retrieving the results
    Wsup_and_lambda = result.x

    flat_w_sup = Wsup_and_lambda[:-N]
    W = buildW(flat_w_sup, N)
    lambda_ = Wsup_and_lambda[-N:]

    # printing the results
    print("Final adjacency matrix:\n", W)
    print("Final lambdas:\n", lambda_)
    print("Initial Loss = ", loss_function(X0, U, N))
    print("Final Loss = ", loss_function(Wsup_and_lambda, U, N))

    sns.heatmap(W0)
    title = 'initial weights'
    plt.title(title)
    if __SAVE_FIG__:
        plt.savefig(__FIG_DIR__+title, format=__FORMAT__)
    plt.show()

    sns.heatmap(W)
    title = 'optimal weights'
    plt.title(title)
    if __SAVE_FIG__:
        plt.savefig(__FIG_DIR__+title, format=__FORMAT__)
    plt.show()

    eigvalues0, eigenvectors0 = np.linalg.eig(laplacian(W0))
    eigvalues, eigenvectors = np.linalg.eig(laplacian(W))
    eigenvectors = eigenvectors[:, np.argsort(eigvalues)]
    eigenvectors0 = eigenvectors0[:, np.argsort(eigvalues0)]

    sns.heatmap(abs(eigenvectors.T @ U))
    title = 'harmonics @ final U, ordered by eigenvalues'
    plt.title(title)
    if __SAVE_FIG__:
        plt.savefig(__FIG_DIR__+title, format=__FORMAT__)
    plt.show()

    print("||initial eigenvectors @ harmonics.T|| = ",
          np.linalg.norm(eigenvectors0 @ (U.T), ord='fro'))
    print("||final eigenvectors @ harmonics.T|| = ",
          np.linalg.norm(eigenvectors @ (U.T), ord='fro'))
    # ---------------------------------------------------
    # Does comparing the two quantities above make sense?
    # ---------------------------------------------------

    plt.plot(eigenvectors[:, :3])
    title = 'First three eigenvectors of the optimal Laplacian'
    plt.title(title)
    if __SAVE_FIG__:
        plt.savefig(__FIG_DIR__+title, format=__FORMAT__)
    plt.show()

    plt.plot(np.sort(eigvalues), 'bo', label='eigenvalues of the optimal solution')
    plt.plot(np.sort(lambda_), 'ro', label='Lambdas')
    title = 'confront eigenvalues'
    plt.legend()
    if __SAVE_FIG__:
        plt.savefig(__FIG_DIR__+title, format=__FORMAT__)
    plt.show()


if __name__ == '__main__':
    main()
