import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from partA import init_basis

# Constants
E1 = 0
E2 = 4
V11 = 3
V12 = 0.2
V21 = 0.2
V22 = - V11

def H() -> np.ndarray:
    """
    Returns the Hamiltonian matrix for the system.
    """
    H0 = np.array([[E1, 0], [0, E2]])
    H1 = np.array([[V11, V12], [V21, V22]])

    return H0 + H1

def H_pauli(lmbda: float) -> np.ndarray:
    """
    Returns the Hamiltonian matrix, written using the Pauli matrices
    """
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    Eps = (E1 + E2) / 2
    Omega = (E1 - E2) / 2
    c = (V11 + V22) / 2
    omega_z = (V11 - V22) / 2   
    omega_x = V21

    H0 = Eps * np.eye(2) + Omega * Z
    H1 = c * np.eye(2) + omega_z * Z + omega_x * X

    return H0 + lmbda * H1

if __name__ == "__main__":
    # Initialize the basis
    q0, q1 = init_basis()
    lmbda = np.linspace(0, 1, 11)
    eigvals = np.zeros((len(lmbda), 2))
    eigvecs = np.zeros((len(lmbda), 2, 2))
    # Calculate the eigenvalues and eigenvectors
    for i, l in enumerate(lmbda):
        hamiltonian = H_pauli(l)
        eigvals[i], eigvecs[i] = np.linalg.eigh(hamiltonian)

    
    # Plot the energy levels and the avoided crossing
    sns.set_theme()
    fig, ax = plt.subplots()
    ax.plot(lmbda, eigvals[:, 0], label=r"$E_0$", color="blue", linestyle="--")
    ax.plot(lmbda, eigvals[:, 1], label=r"$E_1$", color="red", linestyle="--")
    ax.scatter(lmbda, eigvals[:, 0], color="blue", marker="x")
    ax.scatter(lmbda, eigvals[:, 1], color="red", marker="x")
    ax.annotate(r"$\lambda = \frac{2}{3}$", xy=(2/3, 2.3), xytext=(2/3, 3), arrowprops=dict(color="black", arrowstyle="->"), ha="center")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Energy")
    ax.set_title("Energy levels as a function of $\lambda$," + "\n" + "highlighting the avoided crossing")
    ax.legend()
    fig.savefig('../doc/figs/energy_eigenvalues_onequbit.pdf')
    plt.show()    
