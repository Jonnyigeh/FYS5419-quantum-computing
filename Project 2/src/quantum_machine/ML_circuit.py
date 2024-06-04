import matplotlib.pyplot as plt
import numpy as np
import qiskit as qk
from scipy.optimize import minimize
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from tqdm import tqdm
from qiskit_aer import AerSimulator


# Project imports
from utils.encode_data import EncodeData
from utils.prediction import MakePrediction
from utils.parametrization import ParametrizedCircuit


class ML_circuit:
    def __init__(self,
                 feature_matrix,
                 target_matrix,
                 n_tunable_params,
                 n_target_bits = 1,
                 training_cycles = 10,
                 ):
        """
        Class for a quantum machine learning circuit. 
        """
        self.feature_matrix = feature_matrix
        self._N = len(feature_matrix)
        self.target_matrix = target_matrix
        self.n_tunable_params = n_tunable_params
        self.params = np.random.uniform(0, 4 * np.pi, size=n_tunable_params)
        self.initial_params = self.params.copy()
        self.n_target_bits = n_target_bits
        self.n_cycles = training_cycles
        self.num_qubits = len(feature_matrix[0])
        # # Convert into one-hot encoding
        # self._lb = LabelBinarizer()
        # self.target_matrix = self._lb.fit_transform(self.target_matrix)
        # Initialize the circuit
    
    def run_algorithm(self, para_model="2", encode_model="2"):
        print("Initializing the circuit.. for model: ", encode_model)
        self.encode_data(model=encode_model)
        self.parametrize_circuits(model=para_model) 
        self.make_predictions()
        print("Initialization COMPLETE.")
        # From here we can make training.
        # print("Starting optimization..")
        # CE, acc = self.train_circuit()

        return self.predictions, self.get_accuracy(), self.cost_function(self.predictions, self.target_matrix)
        

    def get_accuracy(self):
        """
        Calculate the accuracy of the model.
        """
        try: 
            predictions = np.argmax(self.predictions, axis=1)
            targets = np.argmax(self.target_matrix, axis=1)
            accuracy = np.sum(predictions == targets) / len(predictions)
        except:
            predictions = np.round(self.predictions)
            accuracy = np.sum(predictions == self.target_matrix) / len(predictions)
            
        return accuracy
    
    def encode_data(self, model="2"):
        """
        Produce a list of the encoded data circuits.
        """
        self.encoded_data = []
        for sample in self.feature_matrix:
            encoded_sample = EncodeData(sample, n_target_bits=self.n_target_bits, model=model)
            self.encoded_data.append(encoded_sample.get_circuit)

    def parametrize_circuits(self, 
                            params=None,
                            model="2"):
        """
        Parametrize the encoded data circuits.
        """
        new_circuits = []
        if params is None:
            params = self.params
        parametrized_circuit = ParametrizedCircuit(n_qubits=self.num_qubits, n_clbits=self.n_target_bits, params=params, model=model)
        for circuit in self.encoded_data:
            new_circ = circuit.compose(parametrized_circuit.get_circuit) # Join the decoded data circuit with the parametrized one.
            new_circuits.append(new_circ)

        self.circuits = new_circuits    # Overwrites the old circuits with the new ones.

    def make_predictions(self):
        """
        Make predictions using the parametrized circuits.
        """
        predictions = []
        for circuit in self.circuits:
            predictor = MakePrediction(circuit, n_shots=10000)
            prediction = predictor.make_prediction()
            predictions.append(prediction)
            
        # Overwrite the old predictions with the new ones.
        self.predictions = np.array(predictions)

    def cost_function(self,
                      predictions,
                      targets):
        """
        Different implementation of the log-loss function.
        """
        L = 0
        for prediction, target in zip(predictions, targets):
            L += np.sum(-target * np.log(prediction + 1e-14) - (1 - target) * np.log(1 - prediction + 1e-14))    # Add a small number to avoid log(0).
        
        return L / self._N

    def grad_cost_function(self,
                           predictions,
                           targets,
                           grad_predictions,
                           ):
        """
        Gradient of the different implemented log-loss function.
        """
        dL = 0
        for prediction, target, grad_prediction in zip(predictions, targets, grad_predictions):
            dL += -(target / (prediction + 1e-14) - (1 - target) / (1 - prediction + 1e-14)) * grad_prediction    # Add a small number to avoid division by zero.
        
        return dL / self._N

    def _cost_function(self,
                      predictions,
                      targets):
        """
        Calculate the Cross-Entropy cost function.
        """
        L = 0
        for prediction, target in zip(predictions, targets):
            L += np.sum(target * np.log(prediction + 1e-14))    # Add a small number to avoid log(0).
        
        return -L / self._N

    def _grad_cost_function(self,
                           predictions,
                           targets,
                           grad_predictions,
                           ):
        """
        Calculate the gradient of the Cross entropy cost function w.r.t a given
        parameter theta_k. 
        """
        # Convert from one-hot encoding to [0,1,2] format
        predictions = np.argmax(predictions, axis=1)
        targets = np.argmax(targets, axis=1)
        dLk = 0
        for prediction, target, grad_prediction in zip(predictions, targets, grad_predictions):
            dLk += (prediction - target) / ((prediction) * (1 - prediction) + 1e-12) * grad_prediction # Add a small number to avoid division by zero.
        
        return dLk / self._N
    
    def grad_prediction(self,
                        ):
        """
        Calculates the gradient of the predictions w.r.t the variational parameters by parameter shift rule.

        returns:
        grad_predictions: list
            List of the gradients of the predictions w.r.t the variational parameters. 
            Each element in the list is the corresponding gradient w.r.t parameter theta_k, and the list is of length n_samples.

        """
        grad_predictions = []
        for i in range(self.n_tunable_params):
            # Shift the parameter theta_i by pi/2
            self.params[i] += np.pi / 2
            self.parametrize_circuits()
            self.make_predictions()
            # Convert the predictions back into [0,1,2] format
            prediction_plus = np.array(self.predictions) 
            # Shift the parameter theta_i by -pi/2
            self.params[i] -= np.pi
            self.parametrize_circuits()
            self.make_predictions()
            prediction_minus = np.array(self.predictions) 
            # Calculate the gradient
            grad_predictions.append((prediction_plus - prediction_minus) / 2)
            # Reset the parameter theta_i
            self.params[i] += np.pi / 2


        return grad_predictions

    def _loss_function(self,
                      params):
        """
        Loss function for the COBYLA optimizer.
        """
        self.parametrize_circuits(params=params)
        self.make_predictions()
        L = self.cost_function(self.predictions, self.target_matrix)

        return L

    def _train_circuit(self):
        """
        Optimize the circuit using COBYLA in scipy.
        """
        import time
        self.losses = []
        self._i = 0
        self._t = time.time()
        def _callback(int_res):
            """
            Callback function for the COBYLA optimizer.
            """
            print(f"Iteration: {self._i}/{self.n_cycles}. Time: {time.time() - self._t:.2f}s", end='\r')
            self._i += 1
        
        res = minimize(self._loss_function, self.params, method='COBYLA', options={'maxiter': self.n_cycles,"disp":True},tol=1e-18, callback=_callback)
        breakpoint()

        return 0

    def train_circuit(self,
                      kill_switch = 1e-8,
                      learning_rate = 1.0,
                      ):
        """
        Optimize the circuit using gradient descent.
        """
        # Calculate initial loss
        cross_entropy = []
        acc_scores = []
        L = self.cost_function(self.predictions, self.target_matrix)
        cross_entropy.append(L)
        print("Starting optimization with learning rate: ", learning_rate, "..")
        with tqdm(total=self.n_cycles,
                  desc=rf"[Optimization progress, L = {L:.4f}]",
                  position=0,
                  colour="green",
                  leave=True) as pbar:
            for n in range(self.n_cycles):
                # Calculate the gradient of the predictions
                grad_predictions = self.grad_prediction()
                # Calculate the gradient of the cost function
                dL = []
                for k in range(self.n_tunable_params):
                    # dL.append(self.grad_cost_function(self.predictions, self.target_matrix) * grad_predictions[k])
                    dL.append(self.grad_cost_function(self.predictions, self.target_matrix, grad_predictions[k]))
                # Update the parameters, first convert list to array for elementwise operations.
                self.params -= learning_rate * np.array(dL).reshape(self.params.shape) # self.learning_rate * dL
                # Update the circuits and predictions
                self.parametrize_circuits()
                self.make_predictions()
                # Calculate the new loss and append to list for plotting purposes
                accuracy_score = self.get_accuracy()
                L = self.cost_function(self.predictions, self.target_matrix)
                # Update the progress bar
                pbar.set_description(
                    rf"[Optimization progress, L = {L:.4f}, Accuracy = {accuracy_score:.4f}]"
                )
                pbar.update(1)
                cross_entropy.append(L)
                acc_scores.append(accuracy_score)
                if np.abs(L) < kill_switch or np.abs(cross_entropy[n+1] - cross_entropy[n]) < kill_switch/10:
                    print("Optimization complete..")
                    break
                elif np.abs(L) > 1e6:
                    print("Optimization failed..")
                    break

        return cross_entropy, acc_scores
            
    
    
if __name__ == "__main__":
    import pickle
    seed = 0
    np.random.seed(seed=seed)
    # Import dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # Filter out the last class
    y_idx = np.where(y < 2)
    X = X[y_idx]
    y = y[y_idx]
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=seed)
    # Normalize the dataset using MinMaxScaler - where we fit to the training data.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Initialize the ML circuit
    ml_circuit = ML_circuit(X_train, y_train, n_tunable_params=5)
    # _, _, _ = ml_circuit.run_algorithm()
    # # CE, acc = ml_circuit.train_circuit(learning_rate=0.1)  
    # ml_circuit.run_algorithm()
    # ml_circuit.train_circuit()
    CE = {}
    acc = {}
    models = ["1", "2"]
    for model in models:
        ml_circuit.run_algorithm(encode_model=model)
        # print(ml_circuit.circuits[0])
        ce, ac = ml_circuit.train_circuit()
        CE["encoding = " + model] = ce
        acc["encoding = " + model] = ac

        # Reset the parameters
        ml_circuit.params = ml_circuit.initial_params.copy()
       
    breakpoint()
    with open("CE_enc.pkl", "wb") as f:
        pickle.dump(CE, f)
    with open("acc_enc.pkl", "wb") as f:
        pickle.dump(acc, f)
    
    if False:
        l_rate = [0.25, 0.5, 1.0, 1.5]
        anstzes = ["1", "2", "3"]
        CE_ans = {}
        acc_ans = {}
        params_ans = {}
        for ans in anstzes:
            ml_circuit.run_algorithm(model = ans)
            ce, ac = ml_circuit.train_circuit()
            CE_ans["model = " + ans] = ce
            acc_ans["model = " + ans] = ac
            params_ans["model = " + ans] = ml_circuit.params
            # Reset the parameters
            ml_circuit.params = ml_circuit.initial_params.copy()
        
        
        with open("CE_ans.pkl", "wb") as f:
            pickle.dump(CE_ans, f)
        with open("acc_ans.pkl", "wb") as f:
            pickle.dump(acc_ans, f)
        with open("params_ans.pkl", "wb") as f:
            pickle.dump(params_ans, f)
    
    breakpoint()