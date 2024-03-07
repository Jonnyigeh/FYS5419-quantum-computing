import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from partA import init_basis, hadamard_gate
from partB import H_pauli
from tqdm import tqdm


def pauli_x() -> np.ndarray:
    """
    Returns the Pauli-X matrix.
    """
    return np.array([[0, 1], [1, 0]])

def pauli_y() -> np.ndarray:
    """
    Returns the Pauli-Y matrix.
    """
    return np.array([[0, -1j], [1j, 0]])

def pauli_z() -> np.ndarray:
    """
    Returns the Pauli-Z matrix.
    """
    return np.array([[1, 0], [0, -1]])

def R_x(theta: float) -> np.ndarray:
    """
    Returns the rotation matrix around the x-axis, expressed in Pauli basis.
    """
    return np.cos(theta/2) * np.eye(2) - 1j * np.sin(theta/2) * pauli_x()
    # return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])

def grad_R_x(theta: float) -> np.ndarray:
    """
    Returns the gradient of the rotation matrix around the x-axis, expressed in Pauli basis.
    """
    return -1/2 * theta * pauli_x() * R_x(theta)

def R_y(theta: float) -> np.ndarray:
    """
    Returns the rotation matrix around the y-axis, expressed in Pauli basis.
    """
    return np.cos(theta/2) * np.eye(2) - 1j * np.sin(theta/2) * pauli_y()
    # return np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])

def grad_R_y(theta: float) -> np.ndarray:
    """
    Returns the gradient of the rotation matrix around the y-axis, expressed in Pauli basis.
    """
    return -1/2 * theta * pauli_y() * R_y(theta)
                    
def R_z(theta: float) -> np.ndarray:
    """
    Returns the rotation matrix around the z-axis, expressed in Pauli basis.
    """
    return np.cos(theta/2) * np.eye(2) - 1j * np.sin(theta/2) * pauli_z()
    # return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])

def grad_R_z(theta: float) -> np.ndarray:
    """
    Returns the gradient of the rotation matrix around the z-axis, expressed in Pauli basis.
    """
    return -1/2 * theta * pauli_z() * R_z(theta)

def measure_qubit(qubit: np.ndarray, n_shots: int = 1000) -> np.array:
    """
    Measures the qubit in whichever basis it is currently expressed in.
    """
    prob = np.abs(qubit) ** 2
    measurements = np.zeros(n_shots)
    for i in range(n_shots):
        measurements[i] = np.random.choice([0, 1], p=prob)
    
    return measurements

class VQE():
    def __init__(self, 
                 theta: float = None, 
                 phi: float = None, 
                 lmbda: float = 0,
                 n_iter: int = 100,
                 learning_rate: float = 0.1):
        """Initialize the VQE algorithm.

        ----------
        Parameters:
        phi, theta : float
            The angles for the rotation matrix R_y and R_x respectively.
        lmbda : float
            The parameter for the Hamiltonian matrix.
        n_iter : int
            The number of iterations for the optimization scheme.
        learning_rate : float
            The learning rate for the optimization scheme.
        """
        # Initialize the variational parameters randomly, unless otherwise specified
        if theta is None:
            self.theta = 2 * np.pi * np.random.rand()
        else: 
            self.theta = theta

        if phi is None:
            self.phi = 2 * np.pi * np.random.rand()
        else: 
            self.phi = phi
        
        self.lmbda = lmbda
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.q0, self.q1 = init_basis()         # Initialize the basis |0> and |1> in the Z-basis
        self.H = H_pauli(lmbda)                 # Initialize the Hamiltonian matrix (for use with the exact solution)
        
    def initialize_algorithm(self):
        """
        Starts the VQE algorithm.
        """
        self.initialize_ansatz()                # Initialize the ansatz wavefunction
        self.optimization_scheme()              # Run the optimization scheme

    def initialize_ansatz(self):
        """
        Initializes the ansatz wavefunction for the VQE algorithm.
        """
        self.ansatz = R_y(self.phi) @ R_x(self.theta) @ self.q0
    
    def energy_exact(self, ansatz: np.ndarray=None) -> float:
        """
        Returns the expectation value of the energy for the ansatz wavefunction.
        CHEATING??? Should maybe find the energy using measurements hehe
        """
        if ansatz is None:
            return self.ansatz.conj().T @ self.H @ self.ansatz
        
        return ansatz.conj().T @ self.H @ ansatz
    
    def energy(self, ansatz: np.ndarray=None, n_shots: int = 1000):
        """
        Calculates the expectation value of the energy for the ansatz wavefunction by measurements in the 
        Pauli Z and Pauli X basis, and using the Hamiltonian matrix. See project text for Pauli expansion of the Hamiltonian matrix.
        """
        # Making the constants needed to construct the Hamiltonian matrix
        E1 = 0; E2 = 4; V11 = 3; V12 = 0.2; V21 = 0.2; V22 = - V11
        Eps = (E1 + E2) / 2; Omega = (E1 - E2) / 2; c = (V11 + V22) / 2
        omega_z = (V11 - V22) / 2; omega_x = V21
        if ansatz is None:
            ansatz = self.ansatz
        z_measurements = measure_qubit(ansatz, n_shots)
        x_measurements = measure_qubit(hadamard_gate(ansatz), n_shots)

        # Calculate the energy expectation value
        z_expectation = (n_shots - 2 * np.sum(z_measurements)) / n_shots
        x_expectation = (n_shots - 2 * np.sum(x_measurements)) / n_shots
        I_term = Eps + c * self.lmbda
        Z_term = (Omega + omega_z * self.lmbda) * z_expectation
        X_term = omega_x * self.lmbda * x_expectation
        
        return I_term + Z_term + X_term
    
    
    def temp_ansatz(self, theta: float, phi: float):
        """
        Updates the ansatz wavefunction with new parameters theta and phi.
        """
        return R_y(phi) @ R_x(theta) @ self.q0
    
    
    def optimization_scheme(self, plot: bool=False):
        """
        Optimizes, and minimizes, the energy expectation value using gradient descent 
        to find the optimal parameters for the ansatz wavefunction.

        The algorithmic structure is as follows:
        1. Find an initial estimate for the ground state energy
        2. Calculate the gradients of the variational parameters theta and phi
        3. Update the variational parameters using the gradients (gradient descent)
        4. Repeat steps 2-3 until convergence (or for a set number of iterations)
        """
        
        tol = 1e-13                                # Convergence threshold
        tmp_energy = np.zeros(self.n_iter)
        tmp_energy[0] = self.energy()
        print("Optimization scheme starting...")
        for i in tqdm(range(1, self.n_iter),
                      colour="green",
                      desc="Optimization progress"):
            
            # Calculates the gradients for the parameters theta and phi
            grad_theta = 0.5 * (self.energy(self.temp_ansatz(self.theta + np.pi/2, self.phi))
                                - self.energy(self.temp_ansatz(self.theta - np.pi/2, self.phi)))
            grad_phi = 0.5 * (self.energy(self.temp_ansatz(self.theta, self.phi + np.pi/2)) 
                                - self.energy(self.temp_ansatz(self.theta, self.phi - np.pi/2)))
            
            # Make the update to parameters
            self.theta -= self.learning_rate * grad_theta
            self.phi -= self.learning_rate * grad_phi
            self.ansatz = self.temp_ansatz(self.theta, self.phi)
            
            # Find the energy difference, and update the ground state energy estimate
            tmp_energy[i] = np.real(self.energy())                          # Store the energy estimate, uses np.real to avoid warning.
            delta_E = np.abs(tmp_energy[i] - tmp_energy[i-1]) ** 2
            self.gs_energy_estimate = tmp_energy[i]

            # Check for convergence
            if delta_E < tol:
                print("Convergence reached.")
                break
        
        
        print(f"Optimization finished after {i+1} iterations.")
        print(f"Ground state energy estimate: {self.gs_energy_estimate}")
        
        if plot:
            # The following will visualize the gradient descent, and steady minimization of the energy estimate.
            sns.set_theme(style="darkgrid")
            fig, ax = plt.subplots()
            ax.scatter(np.arange(0, i)[::3], tmp_energy[0:i:3], marker="x", color="tomato")
            ax.plot(np.arange(0, i), tmp_energy[0:i], color="cornflowerblue")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Energy estimate")
            ax.set_title("Energy estimate as a function of iterations")
            fig.savefig('../doc/figs/energy_est_gradient_descent.pdf')
            plt.show()




if __name__ == "__main__":
    # Find the energy estimates for the Hamiltonian for various lambda values.
    lmba_vals = np.linspace(0, 1, 11)
    energies = np.zeros(len(lmba_vals))
    for i, l in enumerate(lmba_vals):
        print("\n", "Starting VQE algorithm for lambda = ", l)
        vqe = VQE(lmbda=l)
        vqe.initialize_algorithm()
        energies[i] = vqe.gs_energy_estimate

    # Visualization of the energy eigenvalues as a function of lambda    
    fig, ax = plt.subplots()
    ax.scatter(lmba_vals, energies, color="darkorange", marker="x")
    ax.plot(lmba_vals, energies, color="cornflowerblue", linestyle="--")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Energy estimate as a function of $\lambda$")
    fig.savefig('../doc/figs/energy_eigenvalues_onequbit_VQE.pdf')
    # plt.show()
    # breakpoint()
    
