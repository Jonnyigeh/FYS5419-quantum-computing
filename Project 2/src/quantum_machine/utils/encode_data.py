import numpy as np
import matplotlib.pyplot as plt
import qiskit as qk
from qiskit_aer import AerSimulator
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class EncodeData:
    def __init__(self,
                 sample,
                 n_target_bits,
                 model="2"):
        """
        Encodes a single sample and target into a quantum circuit.

        args:
        sample: np.array
            The sample to be encoded.
        n_target_bits: int
            The number of possible targets in the dataset decide how many classical bits are needed.
            If one has 3 possible targets - only 2 classical bits are needed. (00, 01, 10). Has to be
            calculated before calling the class.
        """
        self.sample = sample
        self.n_targets = n_target_bits
        self.n_features = len(sample)
        self.set_circuit()
        self.encode_data(which_model=model)

    def set_circuit(self):
        """
        Initializes the quantum circuit given the number of features and targets.
        """
        # Set the qubits and ancillas
        self.data_register = qk.QuantumRegister(self.n_features, 'q')
        self.classical_register = qk.ClassicalRegister(self.n_targets, 'c')
        # Create the quantum circuit
        self.circuit = qk.QuantumCircuit(self.data_register, self.classical_register)

    def encode_data(self, which_model="2"):
        """
        Encode the data into the quantum circuit. Assumes the data has been normalized, so we can scale it with 2 pi.
        """
        if which_model == "1":
             # Encode the data
            for feature_idx in range(self.n_features):
                # Apply the Hadamard gate to the feature qubit to create a superposition
                self.circuit.h(self.data_register[feature_idx])
                # Apply the rotation gate to the feature qubit to encode the data
                self.circuit.rz(4 * np.pi * self.sample[feature_idx], self.data_register[feature_idx])
           
        elif which_model == "2":   
            # Encode the data
            for feature_idx in range(self.n_features):
                # Apply the Hadamard gate to the feature qubit to create a superposition
                self.circuit.h(self.data_register[feature_idx])
                # Apply the rotation gate to the feature qubit to encode the data
                self.circuit.rz(4 * np.pi * self.sample[feature_idx], self.data_register[feature_idx])
            # Apply the CNOT gate to the feature qubits to create entanglement
            for qubit in range(self.n_features - 1):
                self.circuit.cx(self.data_register[qubit], self.data_register[qubit + 1])
                
            
          
        


    @property
    def get_circuit(self):
        """
        Returns the encoded sample as a quantum circuit object.
        """
        return self.circuit



if __name__ == "__main__":
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
    # Visualize the circuit
    encode_data.circuit.draw(output='mpl')
    breakpoint()