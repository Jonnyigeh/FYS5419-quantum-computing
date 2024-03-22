import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from partC import pauli_x, pauli_y, pauli_z

def j1_hamiltonian(epsilon: float = 1.0,
                   V: float = 1.0):
    """Returns the Hamiltonian matrix for J=1 case
    """
    H = np.array([[-epsilon, 0, -V], [0, 0, 0], [-V, 0, epsilon]])

    return H

def j1_hamiltonian_pauli(V: float = 1.0):
        """Returns the Hamiltonian matrix for J=1 case, expressed in the Pauli basis
        """
        I = np.eye(2)
        X = pauli_x()
        Y = pauli_y()
        Z = pauli_z()
        first_term = 0.5 * (
            np.kron(Z, I) + np.kron(I, Z)
        )
        V_term = -V / 2 * (
            np.kron(X, X) - np.kron(Y, Y)
        )
    
        return first_term + V_term

def j2_hamiltonian(epsilon: float = 1.0,
                  V: float = 1.0,
                  W: float = 0.0):
    """Returns the Hamiltonian matrix for J=2 case
    """
    H = np.array([[-2 * epsilon, 0, np.sqrt(6) * V, 0, 0],
                  [0, -epsilon + 3 * W, 0, 3 * V, 0],
                  [np.sqrt(6) * V, 0, 4 * W, 0, np.sqrt(6) * V],
                  [0, 3 * V, 0, epsilon + 3 * W, 0],
                  [0, 0, np.sqrt(6) * V, 0, 2 * epsilon]])

    return H

def j2_hamiltonian_pauli(epsilon: float = 1.0,
                            V: float = 1.0):
    """Returns the Hamiltonian matrix for J=2 case, expressed in the Pauli basis
    """
    I = np.eye(2)
    X = pauli_x()
    Y = pauli_y()
    Z = pauli_z()
    eps_term = -epsilon * (
        np.kron(Z, I) + np.kron(I, Z)
    )
    V_term = np.sqrt(6) * V / 2 * (
        np.kron(X, I) + np.kron(I, X) + np.kron(I, Z) @ np.kron(X, I) - np.kron(I, X) @ np.kron(Z, I)
    )

    return eps_term + V_term


if __name__ == "__main__":
    # Setting up dictionaries for the eigenvalues
    j1_eigenvalues = {}
    j2_eigenvalues = {}
    V = np.round(np.linspace(0, 2, 11), 1)
    for v in V:
        j1_h = j1_hamiltonian(V=v)
        j2_h = j2_hamiltonian(V=v)
        j1_eigs, j1vecs = np.linalg.eigh(j1_h)
        j2_eigs, j2vecs = np.linalg.eigh(j2_h)
        j1_eigenvalues[str(v)] = j1_eigs[0]
        j2_eigenvalues[str(v)] = j2_eigs[0]
    
    # Plotting the eigenvalues for increasing V parameters
    sns.set_theme()
    fig, ax = plt.subplots()
    ax.plot(V, j1_eigenvalues.values(), label="J=1", color="blue", linestyle="--")
    ax.scatter(V, j1_eigenvalues.values(), color="blue", marker="x")
    ax.plot(V, j2_eigenvalues.values(), label="J=2", color="red", linestyle="--")
    ax.scatter(V, j2_eigenvalues.values(), color="red", marker="x")
    ax.set_xlabel("V")
    ax.set_ylabel("Eigenvalues")
    ax.set_title("Eigenvalues for increasing V")
    ax.legend()
    plt.show()
    # fig.savefig('../doc/figs/energy_eigenvalues_j1_j2.pdf')
    exit()
    # Save the eigenvalues to a json file
    import json
    with open("../doc/j1_eigenvalues.json", "w") as file:
        json.dump(j1_eigenvalues, file)
    with open("../doc/j2_eigenvalues.json", "w") as file:
        json.dump(j2_eigenvalues, file)
    
    
    