import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from partC import R_x, R_y
from partD import init_2bit_basis
from partE import VQE_2qubit
from partF import j2_hamiltonian_pauli as H_j2




class VQE_lipkin(VQE_2qubit):
    def __init__(self,
                theta: np.array = None,
                phi: np.array = None,
                V: float = 1.0,
                epsilon: float = 1.0,
                n_iter: int = 100,
                learning_rate: float = 0.1):
        """Initialize the VQE algorithm for the 2-qubit (reduced) Lipkin model for J = 2
        
        -----------
        Parameters:
        phi, theta : np.array
            The angles for the rotation matrix for each individual qubit.
        V : float
            Interaction strenght parameter
        n_iter : int
            Number of iterations in the optimization scheme
        learning_rate : float
            Learning rate for the optimization scheme
        """
        # Inherit the parameters from the VQE_2qubit class, and initialize the basis
        super().__init__(theta=theta, phi=phi, n_iter=n_iter, learning_rate=learning_rate)
        self.q00, self.q01, q10, self.q11 = init_2bit_basis()  # |00>, |01>, |10>, |11> in the Z-basis - we only use |00>, |01> and |11>
        self.V = V
        self.epsilon = epsilon
        self.H = H_j2(epsilon=epsilon, V=V)
    
    


    def initialize_algorithm(self):
        """Starts the VQE algorithm
        """
        print(f"VQE algorithm initialized for V = {self.V:.1f}")
        
        self.initialize_ansatz()
        self.optimization_scheme(scheme=scheme)

    def initialize_ansatz(self):
        return self.temp_ansatz(self.phi) 
        
    def CNOT01(self,
               ansatz: np.ndarray):
        """
        Applies the CNOT_01 gate, where second qubit is the target, first qubit the control bit.
        """
        CNOT01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

        return CNOT01 @ ansatz
    
    def CNOT10(self,
               ansatz: np.ndarray):
        """
        Applies the CNOT_01 gate, where first qubit is the target, second qubit the control bit.
        """
        CNOT10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

        return CNOT10 @ ansatz
    
    def hadamard0(self,
                  ansatz: np.ndarray):
        """
        Apply the Hadamard gate to the first qubit
        """
        H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
        H = np.kron(H, np.eye(2))

        return H @ ansatz

    def hadamard1(self,
                  ansatz: np.ndarray):
        """
        Apply the Hadamard gate to the second qubit
        """
        H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
        H = np.kron(np.eye(2), H)

        return H @ ansatz
    
    def SWAP(self,
             ansatz: np.ndarray):
        """
        Applies the SWAP gate
        """
        swap = np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]])

        return swap @ ansatz

    def CRYgate(self,
                phi: float):
        """
        Returns the controlled-Ry gate for an angle phi, where the first qubit is the control bit.
        """
        CRY = np.array([[1,0,0,0], [0, np.cos(phi/2),0, -np.sin(phi/2)], [0, 0, 1, 0], [0, np.sin(phi/2), 0, np.cos(phi/2)]])

        return CRY
    
    def Sdag0(self,
                   ansatz: np.ndarray):
        """
        Apply the inverse Phase gate to the first qubit
        """
        Sdag = np.array([[1, 0], [0, -1j]])

        return np.kron(Sdag, np.eye(2)) @ ansatz

    def Sdag1(self,
                   ansatz: np.ndarray):
        """
        Apply the inverse Phase gate to the second qubit
        """
        Sdag = np.array([[1, 0], [0, -1j]])

        return np.kron(np.eye(2), Sdag) @ ansatz
    
    def temp_ansatz(self, phi: np.ndarray):
        """
        Updates the ansatz wavefunction with new parameters phi1 and phi2
        """
        
        new_state = self.CRYgate(2 * phi[1]) @ np.kron(R_y(2 * phi[0]), np.eye(2)) @ self.q00

        return new_state
    

    def energy(self, angles: float = 0,
                n_shots: int = 10000,
                ansatz = None) -> float:
        """
        Returns the energy of the system, given the angles for wavefunction.
        """
        if ansatz is None:
            phi = angles
            ansatz = self.temp_ansatz(phi)
        
        # Make measurements, and calculate expectation values
        z0_m, z1_m, x0x1_m, y0y1_m = self.j1_measure_qubits(ansatz, n_shots=n_shots)
        z0_exp = (n_shots - 2 * np.sum(z0_m)) / n_shots
        z1_exp = (n_shots - 2 * np.sum(z1_m)) / n_shots
        x0x1_exp = (n_shots - 2 * np.sum(x0x1_m)) / n_shots
        y0y1_exp = (n_shots - 2 * np.sum(y0y1_m)) / n_shots

        first_term = 0.5 * (z0_exp + z1_exp) 
        second_term = - self.V / 2 * (x0x1_exp - y0y1_exp)        
        self.tot_energy = first_term + second_term
        
        return self.tot_energy


    def measure_qubits(self, 
                       ansatz: np.ndarray, 
                       n_shots: int = 1000) -> np.ndarray:
        """
        Measures the qubits in the computational basis, for the operators in the J=2 scheme
        """
        # Make the proper transformations according to Hundt, Quantum computing for programmers, pg. 252
        prob_z0 = np.abs(ansatz) ** 2
        prob_z1 = np.abs(self.SWAP(ansatz)) ** 2
        prob_x0 = np.abs(self.hadamard0(ansatz)) ** 2
        prob_x1 = np.abs(self.hadamard0(self.SWAP(ansatz))) ** 2
        prob_z1x0 = np.abs(self.CNOT10(self.hadamard0(ansatz))) ** 2
        prob_x1z0 = np.abs(self.CNOT10(self.hadamard1(ansatz))) ** 2

        z0_measurements = np.zeros(n_shots)
        z1_measurements = np.zeros(n_shots)
        x0_measurements = np.zeros(n_shots)
        x1_measurements = np.zeros(n_shots)
        z1x0_measurements = np.zeros(n_shots)
        x1z0_measurements = np.zeros(n_shots)

        for i in range(n_shots):
            # I think there is an issue on how these expectation values are computed
            z0_measurements[i] = np.random.choice([0, 0, 1, 1], p=prob_z0)
            # Think maybe all of them should follow the z0 measurement style?
            z1_measurements[i] = np.random.choice([0, 0, 1, 1], p=prob_z1)
            x0_measurements[i] = np.random.choice([0, 0, 1, 1], p=prob_x0)
            x1_measurements[i] = np.random.choice([0, 0, 1, 1], p=prob_x1)
            z1x0_measurements[i] = np.random.choice([0, 0, 1, 1], p=prob_z1x0)
            x1z0_measurements[i] = np.random.choice([0, 0, 1, 1], p=prob_x1z0)

        return z0_measurements, z1_measurements, x0_measurements, x1_measurements, z1x0_measurements, x1z0_measurements
    
    
    def energy(self, angles: np.ndarray = np.array((0,0)),
                n_shots: int = 10000,
                ansatz = None) -> float:
        """
        Returns the energy of the system, given the angles for wavefunction.
        """
        if ansatz is None:
            phi = angles
            ansatz = self.temp_ansatz(phi)
        

        # Make measurements, and calculate expectation values
        z0_m, z1_m, x0_m, x1_m, z1x0_m, x1z0_m = self.j2_measure_qubits(ansatz, n_shots=n_shots)
        z0_exp = (n_shots - 2 * np.sum(z0_m)) / n_shots
        z1_exp = (n_shots - 2 * np.sum(z1_m)) / n_shots
        x0_exp = (n_shots - 2 * np.sum(x0_m)) / n_shots
        x1_exp = (n_shots - 2 * np.sum(x1_m)) / n_shots
        z1x0_exp = (n_shots - 2 * np.sum(z1x0_m)) / n_shots
        x1z0_exp = (n_shots - 2 * np.sum(x1z0_m)) / n_shots

        # Calculate the energy
        self.non_interacting_energy = -self.epsilon * (z0_exp + z1_exp)
        self.interacting_energy = self.V * np.sqrt(6) / 2 * (x0_exp + x1_exp + z1x0_exp - x1z0_exp)
        
        self.tot_energy = self.non_interacting_energy + self.interacting_energy        
        # breakpoint()
        return self.tot_energy
    
    def optimization_scheme(self, plot: bool = False):
        """Starts the optimization scheme using scipy.minimize
            for either scheme="j1" or scheme="j2" for the two Hamiltonians in the Lipkin model. (2 or 4 particles)
        """        
        from scipy.optimize import minimize
        angles = self.phi
        res = minimize(self.energy, angles, method="Powell", options={"disp": True, "maxiter": self.n_iter}, tol=1e-12)
        self.gs_energy_estimate = res.fun
        self.phi = res.x.reshape(self.phi.shape)
        if plot:
            pass


    

if __name__ == "__main__":
    import json
    with open("../doc/j2_eigenvalues.json", "r") as file:
        j2_eigenvalues = json.load(file)
    
    V_list = np.round(np.linspace(0, 2, 11), 1)
    est_j2_eigenvalues = {}
    # N = 4 case
    for v in V_list:
        vqe = VQE_lipkin(V=v)
        vqe.initialize_algorithm()
        est_j2_eigenvalues[str(v)] = vqe.gs_energy_estimate
        print(f"Exact eigenvalue through diagonalization: {j2_eigenvalues[str(v)]:.4f}")
    
    breakpoint()

    with open("../doc/estimated_j2_eigenvalues.json", "w") as f:
        json.dump(j2_eigenvalues, f)