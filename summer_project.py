from numpy.polynomial.polyutils import as_series
from numpy.polynomial import chebyshev
from numpy.linalg import eig
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy import sparse
import scipy as sp



def get_hamiltonian(n, w, t, periodic=False):
    hamiltonian = np.zeros((n, n))
    #    hamiltonian.fill(0)
    for i in range(n):
        hamiltonian[i, i] = random.uniform(-w, w)
    for i in range(n - 1):
        hamiltonian[i + 1, i] = t
        hamiltonian[i, i + 1] = t
    if periodic:
        hamiltonian[n-1,0] = t
        hamiltonian[0, n-1] = t
    return hamiltonian

def find_min_max_eigenvals(hamiltonian):
    e_val, matrix_v = np.linalg.eigh(hamiltonian)
    eigen_max = np.max(e_val)
    eigen_min = np.min(e_val)
    return eigen_min, eigen_max

def get_state(num_sites, k):
    """ Returns a state with a particle in site `k`"""
    vector_i = np.zeros(num_sites)
    vector_i[k] = 1
    return vector_i

def create_alpha_n(alpha_0, alpha_1, j, hamiltonian):
    all_alphas = np.ndarray(shape=(j, len(alpha_0)))
    all_alphas[0], all_alphas[1] = alpha_0, alpha_1
    for i in range(2, j):
        all_alphas[i] = 2 * hamiltonian @ all_alphas[i - 1] - all_alphas[i - 2]
    return all_alphas

def calculate_moments(alpha_0, d, hamiltonian, alpha_1, num_moments):
    alpha = create_alpha_n(alpha_0, alpha_1, num_moments, hamiltonian)
    mu = np.ones(num_moments)
    for i in range(1, num_moments):
        mu[i] = np.dot(alpha_0, alpha[i])
    return mu

def main():
    n = 10000 # size of matrix
    w = 0.0 # potential energy
    t = -1.0 # hopping

    j = 1000 # number of moments

    EPSILON = 1.0 # Margin in Chebyshev expansion
    hamiltonian = get_hamiltonian(n, w, t) # avg_matrix(n, w, t, n_samples)

    eigen_min, eigen_max = find_min_max_eigenvals(hamiltonian)
    a = (eigen_max - eigen_min) / (2 - EPSILON)
    b = (eigen_max + eigen_min) / 2
    rescaled_hamiltonian = (hamiltonian - b * np.eye(n)) / a

    # creating all vecs-
    alpha_0 = get_state(n, 0)
    alpha_1 = rescaled_hamiltonian @ alpha_0

    c = calculate_moments(alpha_0, n, rescaled_hamiltonian, alpha_1, j)
    c[0] = 0.5  # because of the factor 2 in the function

    # create function -
    x = np.linspace(-1, 1, 1000)
    moment_sum = np.polynomial.chebyshev.chebval(x, c)
    f = (2 / np.pi * np.sqrt(1 - x ** 2)) * moment_sum

    g= f*a +b

    plt.plot(x, f)
    plt.show()

if __name__ == '__main__':
    main()