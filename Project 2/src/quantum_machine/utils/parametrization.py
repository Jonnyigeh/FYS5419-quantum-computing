import numpy as np
import matplotlib.pyplot as plt
import qiskit as qk
from qiskit_aer import AerSimulator
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class ParametrizedCircuit:
    def __init__(self,
                 n_qubits,
                 n_clbits,
                 params,
                 model="3"):
        """
        args:
        num_qubits: int
            Number of qubits in the circuit
        num_clbits: int
            Number of classical bits in the circuit
        params: np.array
            The learnable parameters of the circuit.
        """
        self.n_params = len(params)
        self.theta = params
        self.n_qubits = n_qubits
        self.n_clbits = n_clbits
        if (self.n_params-1) % self.n_qubits != 0:
            raise ValueError("The number of parameters must be divisible by the number of qubits.")
        self.n_params_per_qubit = (self.n_params-1) // self.n_qubits
        self.circuit = qk.QuantumCircuit(self.n_qubits, self.n_clbits)
        self.parametrize_circuit(model=model)


    def parametrize_circuit(self, model="1"):
        """
        Set up a parametrized quantum circuit with tunable parameters.
        """
        if model == "1":
            for qubit in range(self.n_qubits):
                self.circuit.ry(self.theta[qubit], qubit)
        
        elif model == "2":
            for qubit in range(self.n_qubits):
                self.circuit.ry(self.theta[qubit], qubit)
            for qubit in range(self.n_qubits-1):
                self.circuit.cx(qubit, qubit+1)


        elif model == "3":
            for sequence in range(self.n_params_per_qubit):
                # Add rotation gates to each qubit
                for qubit in range(self.n_qubits):
                    self.circuit.ry(self.theta[self.n_params_per_qubit * qubit + sequence], qubit)
                # Add CNOT between each pair of qubits
                for control_qubit in range(self.n_qubits):
                    for target_qubit in range(control_qubit, self.n_qubits):
                        if control_qubit != target_qubit:
                            self.circuit.cx(control_qubit, target_qubit)
                # # Add a bias term to the last qubit
                self.circuit.ry(self.theta[-1], self.n_qubits-1)



    @property
    def get_circuit(self):
        """
        Returns the parametrized quantum circuit.
        """
        return self.circuit

if __name__ == "__main__":
    from encode_data import EncodeData
    # Import dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Normalize the dataset using MinMaxScaler - where we fit to the training data.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Initialize the quantum circuit
    encode_data = EncodeData(X_train[0], y_train[0])
    encode_data.set_circuit()
    encode_data.encode_data()
    # Initialize the parametrized circuit
    parametrize_circuit = ParametrizedCircuit(encode_data.circuit.num_qubits, encode_data.circuit.num_clbits, np.random.uniform(0, 2 * np.pi, size=5))
    parametrized_circuit = parametrize_circuit.get_circuit
    breakpoint()
    # # Visualize the circuit
    # parametrized_circuit.draw(output='mpl')
    # plt.show()
