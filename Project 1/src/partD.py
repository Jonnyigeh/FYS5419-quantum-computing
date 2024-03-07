import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from partA import init_basis # e det vits? nei
from partC import pauli_x, pauli_y, pauli_z

# Constants
Hx = 2.0
Hz = 3.0
epsilon_vals = [0.0, 2.5, 6.5, 7.0]    # diagonal elements in the non-interacting Hamiltonian

def init_2bit_basis() -> np.ndarray:
    """
    Set the basis for the qubits, using the four dimensional
    computational basis |00> = [1,0,0,0], |01> = [0,1,0,0], |10> = [0,0,1,0] and |11> = [0,0,0,1].
    """
    q00 = np.array([1, 0, 0, 0])       # |00>
    q01 = np.array([0, 1, 0, 0])       # |01>
    q10 = np.array([0, 0, 1, 0])       # |10>
    q11 = np.array([0, 0, 0, 1])       # |11>
    
    return q00, q01, q10, q11

def H(lmbda: float) -> np.ndarray:
    """
    Returns the Hamiltonian matrix for the system.
    """
    X = pauli_x()
    Z = pauli_z()
    H0 = np.diag(epsilon_vals)
    HI = Hx * np.kron(X,X) + Hz * np.kron(Z,Z)

    return H0 + lmbda * HI



if __name__ == "__main__":
    # Initialize the basis
    q00, q01, q10, q11 = init_2bit_basis()
    # Calculate the eigenvalues and eigenvectors
    lmbda = np.linspace(0.1, 1, 10)
    eigvals = np.zeros((len(lmbda), 4))
    eigvecs = np.zeros((len(lmbda), 4, 4))
    den_matrix = np.zeros((len(lmbda), 4, 4))
    den_a = np.zeros((len(lmbda), 2, 2))
    den_b = np.zeros_like(den_a)
    von_neumann_entropies = np.zeros(len(lmbda))
    # Finds the density matrix for the lowest eigenstate (E_0) for various interaction strengths
    for i, l in enumerate(lmbda):
        eigvals[i], eigvecs[i] = np.linalg.eigh(H(l))
        den_matrix[i] = np.outer(eigvecs[i].T[0], eigvecs[i].T[0])
        # Find the reduced density matrix for subsystem A by Peijun's journal (www.peijun.me/reduced-density-matrix-and-partial-trace.html)
        den_b[i] = np.trace(den_matrix[i].reshape(2,2,2,2), axis1=0, axis2=2)
        den_a[i] = np.trace(den_matrix[i].reshape(2,2,2,2), axis1=1, axis2=3)
        l_b = np.linalg.eigvalsh(den_b[i])
        von_neumann_entropies[i] = -np.sum(l_b * np.log2(l_b))
        
    # Plot the von Neumann entropy
    sns.set_theme()
    fig, ax = plt.subplots()
    ax.plot(lmbda, von_neumann_entropies, label=r"$S(\rho_B)$", color="blue", linestyle="--")
    ax.scatter(lmbda, von_neumann_entropies, color="tomato", marker="x")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Entanglement entropy")
    ax.set_title(r"Von Neumann entropy as a function of $\lambda$ in subsystem B")
    ax.legend()
    fig.savefig('../doc/figs/von_neumann_entropy.pdf')
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()
    


    # Plot the energy levels
    sns.set_theme()
    fig, ax = plt.subplots()
    for i in range(len(eigvals[0,:])):
        ax.plot(lmbda, eigvals[:,i], linestyle="--", label=rf"$E_{i}$")
        ax.scatter(lmbda, eigvals[:,i], color="tomato", marker="x")
    ax.set_xlabel(r"$\lambda $")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(r"Energy eigenvalues as a function of $\lambda$")
    ax.legend()
    # fig.savefig('../doc/figs/energy_eigenvalues_twoqubit.pdf')
    # plt.show()

