import scipy
import numpy as np
from scipy.optimize import minimize
from scipy import spatial
from scipy.linalg import circulant


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
    for i in range(N):
        W[i, i:] = flat_w_sup[shift:shift+N-i]
        shift += N-i
    W = W+W.T
    for i in range(N):
        W[i, i] /= 2
    return W


def O_LU(W, U):
    """W is a matrix NxN"""
    L = laplacian(W)
    return L@U


def O_Lambda(lambda_, U):
    return np.diag(lambda_)@U


def loss_function(Wsup_and_lambda, U, N):
    flat_w_sup = Wsup_and_lambda[:-N]
    W = buildW(flat_w_sup, N)
    lambda_ = Wsup_and_lambda[-N:]
    return np.linalg.norm(O_LU(W, U) - O_Lambda(lambda_, U))


def harmonics(N):
    sigma = 0.002
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    thetas += sigma * np.random.randn(N)
    U = np.zeros((N, N))
    k = 0
    for i in range(N):
        if i == 0:
            U[:, i] = 0
        if i % 2 == 1:
            k += 1
            U[:, i] = np.sin(k * thetas) / np.sqrt(np.pi)
        if i % 2 == 0:
            U[:, i] = np.cos(k * thetas) / np.sqrt(np.pi)
    return U


def main():
    N = 4
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    assert len(thetas) == N, len(thetas)
    pointset = np.zeros((N, 2))
    pointset[:, 0] = np.cos(thetas)
    pointset[:, 1] = np.sin(thetas)
    vect = spatial.distance.cdist(pointset[0].reshape((1, 2)), pointset)
    weights = np.exp(-vect / np.mean(vect))
    W0 = circulant(weights)
    print(W0)
    U = harmonics(N)
    w = np.ndarray(0)
    for i in range(np.alen(W0)):
        w = np.append(w, W0[i, i:])
    assert len(w) == N*(N+1)/2
    lambdas = np.ones(N)
    X0 = np.concatenate([w, lambdas])
    lb = np.zeros(len(w) + N)
    ub = np.ones(len(w) + N) * np.inf
    bounds = scipy.optimize.Bounds(lb=lb, ub=ub)
    A = np.zeros(len(w) + N)
    A[len(w):] = 1
    b = 1
    cons = [{"type": "eq", "fun": lambda x: A @ x - b}]  # Ax == b
    result = minimize(loss_function, X0, args=(U, N), bounds=bounds,
                      constraints=cons)
    Wsup_and_lambda = result.x
    flat_w_sup = Wsup_and_lambda[:-N]
    W = buildW(flat_w_sup, N)
    lambda_ = Wsup_and_lambda[-N:]
    print(W)
    print(lambda_)


if __name__ == '__main__':
    main()
