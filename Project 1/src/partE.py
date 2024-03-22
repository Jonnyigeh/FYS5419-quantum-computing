import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from partD import H, init_2bit_basis, density_matrix
from partC import VQE, R_x, R_y, R_z
from tqdm import tqdm


class VQE_2qubit(VQE):
    def __init__(self,
                 theta: np.array = None,
                 phi: np.array = None,
                 lmbda: float = 0,
                 n_iter: int = 100,
                 learning_rate: float = 0.1,
                 scipy=False):
        """Initialize the VQE algorithm for a 2 qubit system
        
        ---------
        Parameters:
        phi, theta : np.array
            The angles for the rotation matrix for each individual qubit.
        lmbda : float
            Interaction strenght parameter
        n_iter : int
            Number of iterations in the optimization scheme
        learning_rate : float
            Learning rate for the optimization scheme
        """
        # Initialize the variational paramters randomly, unless otherwise specified
        super().__init__(lmbda=lmbda, n_iter=n_iter, learning_rate=learning_rate)
        if theta is None:
            self.theta = 2 * np.pi * np.random.rand(2)
        else: 
            self.theta = theta
        
        if phi is None:
            self.phi = 2 * np.pi * np.random.rand(2)
        else:
            self.phi = phi
        
        self.q00, self.q01, self.q10, self.q11 = init_2bit_basis()  # Initialize the basis |00>, |01>, |10>, |11> in the Z-basis
        self.H = H(lmbda)                                           # Initialize the Hamiltonian matrix (for use with the exact solution)
        if scipy:
            self.optimization_scheme = self.scipy_optimization
    
    def hadamard_2qubit(self, qubit: np.ndarray) -> np.ndarray:
        """
        Apply the Hadamard gate to both qubits.
        """
        H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
        H = np.kron(H, H)
    
        return H @ qubit
    
    def CNOT(self)-> np.ndarray:
        """
        Gives the 2qubit CNOT gate in matrix form
        """
        return np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])
    
    def measure_qubits(self, ansatz: np.ndarray, n_shots: int = 1000, epsilon_vals: list=[0.0, 2.5, 6.5, 7.0]) -> np.ndarray:
        """
        Measures the wavefunction in the computational basis.
        """
        prob_z = np.abs(ansatz) ** 2
        prob_x = np.abs(self.hadamard_2qubit(ansatz)) ** 2
        z_measurements = np.zeros(n_shots)
        x_measurements = np.zeros(n_shots)
        H0_measurements = np.zeros(n_shots)
        
        for i in range(n_shots):
            z_measurements[i] = np.random.choice([0, 1, 1, 0], p=prob_z)
            x_measurements[i] = np.random.choice([0, 1, 1, 0], p=prob_x)
            H0_measurements[i] = np.random.choice(epsilon_vals, p=prob_z)
        
        return z_measurements, x_measurements, H0_measurements
    
    def initialize_ansatz(self):
        """
        Initializes the ansatz wavefunction for the 2 qubit system
        """
        self.ansatz = self.CNOT() @ np.kron(R_y(self.theta[0]), R_y(self.theta[1])) @ np.kron(R_x(self.phi[0]), R_x(self.phi[1])) @ self.q00    

    def temp_ansatz(self, theta: np.ndarray, phi: np.ndarray):
        """
        Updates the ansatz wavefunction with new parameters theta and phi
        """
        return self.CNOT() @  np.kron(R_y(theta[0]), R_y(theta[1])) @ np.kron(R_x(phi[0]), R_x(phi[1])) @ self.q00
    
    def exact_energy(self, ansatz: np.ndarray=None)->float:
        if ansatz is None:
            return self.ansatz.conj().T @ self.H @ self.ansatz
        return ansatz.conj().T @ self.H @ ansatz

    def energy(self, ansatz: np.ndarray = None, n_shots: int = 1000)-> float:
        """
        Calculate the energy of the ansatz wavefunction.
        """
        Hx = 2.0; Hz = 3.0; epsilon_vals = [0.0, 2.5, 6.5, 7.0]
        if ansatz is None:
            ansatz = self.ansatz
        H0 = np.diag(epsilon_vals)      # Unperturbed Hamiltonian
        z_measurements, x_measurements, H0_measurements = self.measure_qubits(ansatz, n_shots, epsilon_vals)
        # Calculates the expectation value given the measurements
        z_exp = (n_shots - 2 * np.sum(z_measurements)) / n_shots
        x_exp = (n_shots - 2 * np.sum(x_measurements)) / n_shots
        H0_exp = np.mean(H0_measurements)
        
        non_int_term = H0_exp
        int_term = Hx * x_exp + Hz * z_exp
        
        energy = non_int_term + self.lmbda * int_term
        return energy
    
    def _energy(self, angles, n_shots = 1000):
        Hx = 2.0; Hz = 3.0; epsilon_vals = [0.0, 2.5, 6.5, 7.0]
        theta, phi = angles.reshape(2, 2)
        self.ansatz = self.temp_ansatz(theta, phi)
        z_measurements, x_measurements, H0_measurements = self.measure_qubits(self.ansatz, n_shots, epsilon_vals)

        z_exp = (n_shots - 2 * np.sum(z_measurements)) / n_shots
        x_exp = (n_shots - 2 * np.sum(x_measurements)) / n_shots
        H0_exp = np.mean(H0_measurements)
        
        non_int_term = H0_exp
        int_term = Hx * x_exp + Hz * z_exp
        
        energy = non_int_term + self.lmbda * int_term
        return energy
    
    def optimization_scheme(self, plot: bool = False):
        """
        Run the optimization scheme for the VQE algorithm.
        """
        tol = 1e-10
        tmp_energy = np.zeros(self.n_iter)
        tmp_energy[0] = self.energy()
        print("Optimization scheme starting...")
        for i in tqdm(range(1, self.n_iter),
                      colour="green",
                      desc="Optimization progress"):
            
            # Calculate the gradients
            grad_theta0 = 0.5 * (self.energy(ansatz=self.temp_ansatz(theta=np.array([self.theta[0] + np.pi/2, self.theta[1]]), phi=self.phi))
                                - self.energy(ansatz=self.temp_ansatz(theta=np.array([self.theta[0] - np.pi/2, self.theta[1]]), phi=self.phi)))
            
            grad_theta1 = 0.5 * (self.energy(ansatz=self.temp_ansatz(theta=np.array([self.theta[0], self.theta[1] + np.pi/2]), phi=self.phi))
                                - self.energy(ansatz=self.temp_ansatz(theta=np.array([self.theta[0], self.theta[1] - np.pi/2]), phi=self.phi)))
            
            grad_phi0 = 0.5 * (self.energy(ansatz=self.temp_ansatz(theta=self.theta, phi=np.array([self.phi[0] + np.pi/2, self.phi[1]]))) 
                                - self.energy(ansatz=self.temp_ansatz(theta=self.theta, phi=np.array([self.phi[0] - np.pi/2, self.phi[1]]))))

            grad_phi1 = 0.5 * (self.energy(ansatz=self.temp_ansatz(theta=self.theta, phi=np.array([self.phi[0], self.phi[1] + np.pi/2]))) 
                                - self.energy(ansatz=self.temp_ansatz(theta=self.theta, phi=np.array([self.phi[0], self.phi[1] - np.pi/2]))))
            
            # Update the parameters
            self.theta[0] -= self.learning_rate * grad_theta0
            self.theta[1] -= self.learning_rate * grad_theta1
            self.phi[0] -= self.learning_rate * grad_phi0
            self.phi[1] -= self.learning_rate * grad_phi1
            self.ansatz = self.temp_ansatz(self.theta, self.phi)

            # Calculate the energy
            tmp_energy[i] = np.real(self.energy())
            delta_E = np.abs(tmp_energy[i] - tmp_energy[i-1]) ** 2
            self.gs_energy_estimate = tmp_energy[i]

            if delta_E < tol:
                print(f"Convergence reached after {i} iterations.")
                break
        
        print(f"Optimization finished after {i+1} iterations.")
        print(f"Ground state energy estimate: {self.gs_energy_estimate:.3f}")
        if plot:
            sns.set_theme(style="darkgrid")
            fig, ax = plt.subplots()
            ax.scatter(np.arange(0, i)[::3], tmp_energy[0:i:3], marker="x", color="tomato")
            ax.plot(np.arange(0, i), tmp_energy[0:i], color="cornflowerblue")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Energy estimate")
            ax.set_title("Energy estimate as a function of iterations")
            # fig.savefig('../doc/figs/energy_est_gradient_descent.pdf')
            plt.show()

    def scipy_optimization(self):
        """Optimize using SciPy library
        """
        from scipy.optimize import minimize
        angles = np.array([self.theta, self.phi])
        res = minimize(self.energy, angles.ravel(), method="Powell", options={"disp": True, "maxiter": 1000}, tol=1e-12)
        self.gs_energy_estimate = res.fun
        self.theta, self.phi = res.x.reshape(2, 2)

if __name__ == "__main__":    
    # vqe = VQE_2qubit(lmbda=0.5, scipy=True)
    # vqe.initialize_algorithm()
    
    # Load the eigenvalues found in part D
    with open("../doc/eigenvalues.json", "r") as f:  
        exact_eigvals = json.load(f)

    estimated_eigvals = {}
    # Initialize the VQE algorithm
    lmbda = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    for i, l in enumerate(lmbda):
        vqe = VQE_2qubit(lmbda=l,scipy=False)
        vqe.initialize_algorithm()
        # estimated_eigvals[str(l)] = vqe.gs_energy_estimate
        print(f"Exact energy eigenvalue for lambda = {l:.1f}: {exact_eigvals[str(l)]:.3f}")
        print("\n")
    
    breakpoint()
    exit()
    with open("../doc/estimated_eigenvalues.json", "w") as f:
        json.dump(estimated_eigvals, f)
    
    breakpoint()



            