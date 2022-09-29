from numpy.polynomial.polyutils import as_series
from numpy.polynomial import chebyshev
from numpy.linalg import eig
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy import sparse
import scipy as sp
from scipy.sparse.linalg import eigsh


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

# def get_eigen_values_for_matrix(hamiltonian):
#     e_val, e_vec = eig(hamiltonian)
#     return e_val, e_vec

def find_min_max_eigenvals(hamiltonian):
    # e_val, e_vec = get_eigen_values_for_matrix(hamiltonian)
    eigen_val, eigen_vec = sp.sparse.linalg.eigsh(hamiltonian)
    eigen_ma = max(eigen_val)  # sp.sparse.linalg.eigsh(hamiltonian, k=1, which='LM')[0]
    eigen_mi = min(eigen_val)  # sp.sparse.linalg.eigsh(hamiltonian, k=1, which='SM')[0]
    return eigen_mi, eigen_ma


def get_state(num_sites, k):
    """ Returns a state with a particle in site `k`"""
    vector_i = np.zeros(num_sites)
    vector_i[k] = 1
    return vector_i

def create_alpha_0_and_alpha_1(n, rescaled_hamiltonian, k):
    # creating all vecs-
    alpha_0 = get_state(n ** 3, k)
    alpha_1 = rescaled_hamiltonian @ alpha_0
    return alpha_1, alpha_0

# def create_alpha_n(alpha_0, alpha_1, j, hamiltonian):
#     all_alphas = np.ndarray(shape=(j, len(alpha_0)))
#     all_alphas[0], all_alphas[1] = alpha_0, alpha_1
#     for i in range(2, j):
#         all_alphas[i] = 2 * hamiltonian @ all_alphas[i - 1] - all_alphas[i - 2]
#     return all_alphas

def create_alpha_2(alpha_0, alpha_1, hamiltonian):
    alpha_2 = 2 * hamiltonian @ alpha_1 - alpha_0
    return alpha_2


def calculate_moments(alpha_0, alpha_1, hamiltonian, j):
    mu = np.ones(j)
    alpha_int= np.copy(alpha_0)
    alpha_int1 =  np.copy(alpha_1)
    mu[1] = np.dot(alpha_int, alpha_int1)
    alpha_2 = create_alpha_2(alpha_0, alpha_1, hamiltonian)
    for i in range(2,j):
        alpha_int = np.copy(alpha_int1)
        alpha_int1 = np.copy(alpha_2)
        alpha_2 = create_alpha_2(alpha_int, alpha_int1, hamiltonian)
        mu[i] = np.dot(alpha_int, alpha_int1)
        if np.sum(alpha_1) == np.sum(alpha_2):
            print("match")
    print('mu', mu)
    return mu


def density_of_states(alpha_0, alpha_1, j, hamiltonian):
    # create function -
    x = np.linspace(-1, 1, 1000)
    c = calculate_moments(alpha_0, alpha_1,  hamiltonian, j)
    c[0] = 0.5  # because of the factor 2 in the function
    moment_sum = np.polynomial.chebyshev.chebval(x, c)
    f = (2 / np.pi * np.sqrt(1 - x ** 2)) * moment_sum
    print(c[3])
    return f, x


def eigen_values(hamiltonian, n, EPSILON):  # rescaling hamiltonian
    eigen_min, eigen_max = find_min_max_eigenvals(hamiltonian)
    a_diff = eigen_max - eigen_min
    b_sum = eigen_max + eigen_min
    a = a_diff / (2 - EPSILON)
    b = b_sum / 2
    hamiltonian_b_diff = np.diff([hamiltonian, b * np.eye(n ** 3)])
    rescaled_hamiltonian = sparse.csr_matrix(hamiltonian_b_diff[0]) / a
    return rescaled_hamiltonian



def average_of_densities(n_runs, n, w, t, EPSILON, j):
    l = np.random.choice(range(n ** 3), size=10, replace=False)
    hamiltonian = get_hamiltonian(n, w, t)
    rescaled_hamiltonian = eigen_values(hamiltonian, n, EPSILON)
    complete_loop = np.zeros(1000)
    for k in l:
        y = np.zeros(1000)
        alpha_1, alpha_0 = create_alpha_0_and_alpha_1(n, rescaled_hamiltonian, k)
        for i in range(n_runs):
            f, x = density_of_states(alpha_0, alpha_1, j, hamiltonian)
            y += f
        print("site " + str(k) + " done")
        average_density = y / n_runs
        complete_loop += average_density
    return complete_loop/len(l), x


def main():
    n = 3  # size of matrix
    t = 1  # hopping
    j = 20  # number of moments
    w = 0 * t  # potential energy
    n_runs = 1 # realizations of disorder
    EPSILON = 1.0  # Margin in Chebyshev expansion

    hamiltonian = get_hamiltonian(n, w, t)  # avg_matrix(n, w, t, n_samples)

    # eigen_min, eigen_max = find_min_max_eigenvals(hamiltonian)
    # a = (eigen_max - eigen_min) / (2 - EPSILON)
    # b = (eigen_max + eigen_min) / 2
    # rescaled_hamiltonian = (hamiltonian - b*np.eye(n**3) ) / a
    #
    # # creating all vecs-
    # alpha_1, alpha_0 = create_alpha_0_and_alpha_1(n, rescaled_hamiltonian)
    # f, x = density_of_states(alpha_0,n, rescaled_hamiltonian, alpha_1 , j)
    average_spectrum, x = average_of_densities(n_runs, n, w, t, EPSILON, j)
    np.savetxt('n' + str(n) + '_j' + str(j) + '_runs' + str(n_runs) + '.dat', np.c_[x, average_spectrum])
    plt.plot(x, average_spectrum)
    plt.show()


if __name__ == '__main__':
    main()
