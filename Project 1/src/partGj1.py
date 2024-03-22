import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from partA import init_basis, hadamard_gate
from partF import j1_hamiltonian_pauli as H_j1
from partC import VQE, R_y, pauli_x, pauli_z

def H(V: float = 1.0):
    """
    Returns the Hamiltonian matrix for the 2-qubit Lipkin model
    """
    X = pauli_x()
    Z = pauli_z()
    
    return -Z - V * X
    
class VQE_lipkin(VQE):
    def __init__(self, 
                     V: float = 1.0,
                     n_iter: int = 100,
                     learning_rate: float = 0.1,
                     phi: float = None):
            """Initialize the VQE algorithm for the 2-qubit (reduced) Lipkin model for J = 1
            -----------
            Parameters:
            V : float
                Interaction strenght parameter
            """
            super().__init__(n_iter=n_iter, learning_rate=learning_rate, phi=phi)
            self.q0, self.q1 = init_basis()
            self.V = V
            self.H = H(V=V)

        
    def initialize_algorithm(self):
        """
        Starts the VQE algorithm.
        """
        print(f"VQE algorithm initialized for V = {self.V:.1f}")
        self.initialize_ansatz()                # Initialize the ansatz wavefunction
        self.optimization_scheme()              # Run the optimization scheme
    

    def Z(self, qubit: np.ndarray) -> np.ndarray:
            """
            Returns the Z operator acting on the qubit
            """
            Z = pauli_z()

            return Z @ qubit
        
    def X(self, qubit: np.ndarray) -> np.ndarray:
            """
            Returns the X operator acting on the qubit
            """
            X = pauli_x()

            return X @ qubit

    def measure_qubit(self, ansatz: np.ndarray, n_shots=1000) -> np.array:
        """
        Measures the qubit in the Z-basis
        """
        
        prob_z = np.abs(ansatz)**2
        prob_x = np.abs(hadamard_gate(ansatz))**2
    
        measurements_x = np.zeros(n_shots)
        measurements_z = np.zeros(n_shots)
        for i in range(n_shots):
            measurements_z[i] = np.random.choice([0, 1], p=prob_z)
            measurements_x[i] = np.random.choice([0, 1], p=prob_x)

        return measurements_z, measurements_x
        
    def energy(self, angle: float=0, ansatz: np.ndarray=None, n_shots=1000) -> float:
        """
        Returns the energy of the ansatz wavefunction
        """
        if ansatz is None:
            ansatz = self.temp_ansatz(angle)
        
        z_m, x_m = self.measure_qubit(ansatz, n_shots=n_shots)
        z_exp = (n_shots - 2 * np.sum(z_m)) / n_shots
        x_exp = (n_shots - 2 * np.sum(x_m)) / n_shots
        
        first_term = - z_exp
        second_term = -self.V * x_exp
        energy = first_term + second_term
        
        return energy

    def temp_ansatz(self, phi: float):
        """
        Updates the ansatz wavefunctino with the new parameter phi
        """
        new_state = R_y(2 * float(phi)) @ self.q0

        return new_state

    def optimization_scheme(self):
        from scipy.optimize import minimize
        angles = self.phi
        res = minimize(self.energy, angles, method="Powell", options={"disp": True, "maxiter": self.n_iter}, tol=1e-12)
        self.gs_energy_estimate = res.fun
        self.phi = float(res.x)

if __name__ == "__main__":
    import json
    with open("../doc/j1_eigenvalues.json", "r") as file:
        j1_eigenvalues = json.load(file)

    V_list = np.round(np.linspace(0, 2, 11), 1)
    est_j1_eigenvalues = {}
    vqe = VQE_lipkin(V=0.6)
    eigval, eigvecs = np.linalg.eigh(vqe.H)
    e = vqe.energy(ansatz=eigvecs[0])
    
    for v in V_list:
        vqe = VQE_lipkin(V=v)
        eigval, eigvecs = np.linalg.eigh(vqe.H)
        vqe.initialize_algorithm()
        est_j1_eigenvalues[str(v)] = vqe.gs_energy_estimate
        print(f"Exact eigenvalue through diagonalization: {j1_eigenvalues[str(v)]:.4f}" + "\n")

    breakpoint()
    with open("estimated_j1_eigenvalues.json", "w") as f:
        json.dump(j1_eigenvalues, f)
    