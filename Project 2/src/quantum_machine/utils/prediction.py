import numpy as np
import matplotlib.pyplot as plt
import qiskit as qk
from qiskit_aer import AerSimulator, StatevectorSimulator
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class MakePrediction:
    def __init__(self,
                 circuit,
                 n_shots,
                 backend="StatevectorSimulator"):
        """
        Make a prediction using a parametrized quantum circuit, based on the superposition of the last qubit.
        |00> -> 0
        |01> -> 1
        |10> -> miss
        |11> -> 2

        args:
        circuit: qiskit.QuantumCircuit
            The parametrized circuit to be used for prediction.
        n_shots: int
            The number of shots to be used in the simulation.
        """
        self.circuit = circuit
        self.n_qubits = circuit.num_qubits
        self.n_clbits = circuit.num_clbits
        self.data_register = self.circuit.qregs[0]
        self.classical_register = self.circuit.cregs[0]
        self.n_shots = n_shots
        backends = {"StatevectorSimulator": StatevectorSimulator,
                    "AerSimulator": AerSimulator}
        self.backend = AerSimulator(method="statevector")
        
    
    def run_circuit(self):
        """
        Add measurement to final qubit
        """
        self.circuit.measure([self.data_register[-1]], self.classical_register[-1])
        # Compile the circuit
        compiled_circuit = qk.transpile(self.circuit, self.backend)
        # Run the circuit
        job = self.backend.run(compiled_circuit, shots=self.n_shots)
        result = job.result()
        counts = result.get_counts(self.circuit)
        
        return counts

    # def run_circuit(self):
    #     """
    #     Adds measurements to the appropriate qubits
    #     """
    #     # Add measurements to circuit
    #     for i in range(self.n_clbits):
    #         self.circuit.measure([self.data_register[self.n_qubits - 1 - i]], self.classical_register[i])
    #     # self.circuit.measure([self.data_register[-self.n_clbits:]], self.classical_register)
    #     # Compile the circuit
    #     compiled_circuit = qk.transpile(self.circuit, self.backend)
    #     # Run the circuit
    #     job = self.backend.run(compiled_circuit, shots=self.n_shots)
    #     result = job.result()
    #     counts = result.get_counts(self.circuit)
        
    #     return counts
    
    def make_prediction(self):
        """
        Make predictions according to:

        |0> -> 0
        |1> -> 1

        returns:
        predictions: float
            floating point value of the prediction. Should be rounded to the nearest integer, as the prediction is a discrete value.
        """
        counts = self.run_circuit()
        temp_prediction = 0
        for key in counts.keys():
            if key == '0':
                pass
            elif key == '1':
                temp_prediction += 1 * counts[key]
        
        prediction = temp_prediction / self.n_shots
        
        return prediction

    # def make_prediction(self):
    #     """
    #     Make predicitions according to:

    #     |00> -> 0
    #     |01> -> 1
    #     |10> -> miss
    #     |11> -> 2

    #     returns:
    #     predictions: float
    #         floating point value of the prediction. Should be rounded to the nearest integer, as the prediction is a discrete value.
    #     """
    #     counts = self.run_circuit()
    #     # temp_prediction = 0
    #     # for key in counts.keys():
    #     #     if key == '00':
    #     #         pass
    #     #     elif key == '01':
    #     #         # predictions['1'] = counts[key]
    #     #         temp_prediction += 1 * counts[key]
    #     #     elif key == '11':
    #     #         # predictions['2'] = counts[key]
    #     #         temp_prediction += 2 * counts[key]
    #     #     else:
    #     #         pass
        
    #     # prediction = temp_prediction / self.n_shots
    #     measurement = max(counts, key=counts.get)
    #     if measurement == '00':
    #         prediction = [1,0,0]
    #     elif measurement == '01':
    #         prediction = [0,1,0]
    #     elif measurement == '11':
    #         prediction = [0,0,1]
    #     else:
    #         prediction = [0,0,0]     # If the measurement is not one of the expected values, return infinity.
        
    #     return prediction 

        

if __name__ == "__main__":
    from encode_data import EncodeData
    from parametrization import ParametrizedCircuit
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
    n_features = X_train.shape[1]
    n_targets = 2       # 3 possible targets -> 2 classical bits needed
    # Encode the data
    encoded_data = EncodeData(X_train[0], n_targets)
    circuit = encoded_data.get_circuit
    # Parametrize the circuit
    parametrized_circuit = ParametrizedCircuit(circuit.num_qubits, circuit.num_clbits, np.random.uniform(0, 2 * np.pi, size=12))
    parameter_circuit = parametrized_circuit.get_circuit
    # Join the circuits
    circuit.compose(parameter_circuit, inplace=True)
    # Make a prediction
    predictor = MakePrediction(circuit, 1000)
    predictions = predictor.make_prediction()
    print(predictions)
        