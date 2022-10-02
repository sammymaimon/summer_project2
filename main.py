from numpy.polynomial.polyutils import as_series
from numpy.polynomial import chebyshev
from scipy.sparse.linalg import eigsh
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy import sparse
import scipy as sp


def get_hamiltonian(n, w, t):
    hamiltonian = sp.sparse.lil_matrix((n ** 3, n ** 3))
    for i in range(n):
        ip1 = (i + 1) % n
        for j in range(n):
            jp1 = (j + 1) % n
            for k in range(n):
                kp1 = (k + 1) % n
                hamiltonian[i * n * n + j * n + k, i * n * n + j * n + k] = random.uniform(-w, w)
                hamiltonian[ip1 * n * n + j * n + k, i * n * n + j * n + k] = t
                hamiltonian[i * n * n + j * n + k, ip1 * n * n + j * n + k] = t
                hamiltonian[i * n * n + jp1 * n + k, i * n * n + j * n + k] = t
                hamiltonian[i * n * n + j * n + k, i * n * n + jp1 * n + k] = t
                hamiltonian[i * n * n + j * n + kp1, i * n * n + j * n + k] = t
                hamiltonian[i * n * n + j * n + k, i * n * n + j * n + kp1] = t
    return hamiltonian


# def get_hamiltonian(n, w, t, periodic=False):
#     hamiltonian = np.zeros((n, n))
#     #    hamiltonian.fill(0)
#     for i in range(n):
#         hamiltonian[i, i] = random.uniform(-w, w)
#     for i in range(n - 1):
#         hamiltonian[i + 1, i] = t
#         hamiltonian[i, i + 1] = t
#     if periodic:
#         hamiltonian[n-1,0] = t
#         hamiltonian[0, n-1] = t
#     return hamiltonian

def find_min_max_eigenvals(hamiltonian):
    eigen_val, eigen_vec = sp.sparse.linalg.eigsh(hamiltonian)
    eigen_ma = np.max(eigen_val)
    eigen_mi = np.min(eigen_val)
    return eigen_mi, eigen_ma


def eigen_values(hamiltonian, n, EPSILON):  # rescaling hamiltonian
    eigen_min, eigen_max = find_min_max_eigenvals(hamiltonian)
    a_diff = eigen_max - eigen_min
    b_sum = eigen_max + eigen_min
    a = a_diff / (2 - EPSILON)
    b = b_sum / 2
    hamiltonian_b_diff = np.diff([hamiltonian, b * np.eye(n ** 3)])
    rescaled_hamiltonian = sparse.csr_matrix(hamiltonian_b_diff[0]) / a
    return rescaled_hamiltonian


def get_state(n, k):
    """ Returns a state with a particle in site `k`"""
    vector_i = np.zeros(n**3)
    vector_i[k] = 1
    return vector_i


def create_alpha_0_and_alpha_1(n, rescaled_hamiltonian, k):
    # creating all vecs-
    alpha_0 = get_state(n, k)
    alpha_1 = rescaled_hamiltonian @ alpha_0
    return alpha_1, alpha_0


def create_alpha_n(alpha_0, alpha_1, j, rescaled_hamiltonian):
    all_alphas = np.ndarray(shape=(j, len(alpha_0)))
    all_alphas[0], all_alphas[1]= alpha_0, alpha_1
    for i in range(2, j):
        all_alphas[i] = 2 * rescaled_hamiltonian @ all_alphas[i - 1] - all_alphas[i - 2]
    return all_alphas


def calculate_moments(alpha_0, alpha_1, rescaled_hamiltonian, j):
    alpha = create_alpha_n(alpha_0, alpha_1, j, rescaled_hamiltonian)
    mu = np.ones(j)
    for i in range(1, j):
        mu[i] = np.dot(alpha_0, alpha[i])
    return mu


def density_of_states(alpha_0, alpha_1, j, rescaled_hamiltonian):
    # create function -
    x = np.linspace(-1, 1, 1000)
    c = calculate_moments(alpha_0, alpha_1, rescaled_hamiltonian, j)
    c[0] = 0.5  # because of the factor 2 in the function
    moment_sum = np.polynomial.chebyshev.chebval(x, c)
    f = (2 / np.pi * np.sqrt(1 - x ** 2)) * moment_sum
    return f, x


def average_densities(n_runs, n, w, t, EPSILON, j):
    l = np.random.choice(range(n ** 3), size=10, replace=False)
    # hamiltonian = get_hamiltonian(n, w, t)
    # rescaled_hamiltonian = eigen_values(hamiltonian, n, EPSILON)
    average_density_final = np.zeros(1000)
    for k in l:
        y = np.zeros(1000)
        for i in range(n_runs):
            hamiltonian = get_hamiltonian(n, w, t)
            rescaled_hamiltonian = eigen_values(hamiltonian, n, EPSILON)
            alpha_1, alpha_0 = create_alpha_0_and_alpha_1(n, rescaled_hamiltonian, k)
            f, x = density_of_states(alpha_0, alpha_1, j, rescaled_hamiltonian)
            y += f
        print("site " + str(k) + " done")
        average_density_final += y / (n_runs)
    return average_density_final/10, x


def main():
    n = 15  # size of matrix
    t = 1.0  # hopping
    w = 1.5 * t  # potential energy
    j = 70  # number of moments
    EPSILON = 1.0  # Margin in Chebyshev expansion
    n_runs = 40  # number of iterations

    #hamiltonian = get_hamiltonian(n, w, t)  # avg_matrix(n, w, t, n_samples)

    # eigen_min, eigen_max = find_min_max_eigenvals(hamiltonian)
    # a = (eigen_max - eigen_min) / (2 - EPSILON)
    # b = (eigen_max + eigen_min) / 2
    # rescaled_hamiltonian = (hamiltonian - b * np.eye(n**3)) / a

    # creating all vecs-
    # alpha_0 = get_state(n**3, 0)
    # alpha_1 = rescaled_hamiltonian @ alpha_0

    # c = calculate_moments(alpha_0, n, rescaled_hamiltonian, alpha_1, j)
    # c[0] = 0.5  # because of the factor 2 in the function

    # # create function -
    # x = np.linspace(-1, 1, 1000)
    # moment_sum = np.polynomial.chebyshev.chebval(x, c)
    # f = (2 / np.pi * np.sqrt(1 - x ** 2)) * moment_sum

    #
    # g= f*a +b
    #
    # plt.plot(x, f)
    # plt.show()

    average_spectrum, x = average_densities(n_runs, n, w, t, EPSILON, j)
    np.savetxt('n' + str(n) + '_j' + str(j) + '_runs' + str(n_runs) + '.dat', np.c_[x, average_spectrum])
    plt.plot(x, average_spectrum)
    plt.show()


if __name__ == '__main__':
    main()
