import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.encode_data import EncodeData
from utils.parametrization import ParametrizedCircuit
from utils.prediction import MakePrediction
from ML_circuit import ML_circuit
from sklearn.preprocessing import MinMaxScaler

def make_test_run(X_test,y_test,params):
    mlc = ML_circuit(X_test,y_test,13, n_target_bits=1)
    mlc.params = params # Give the optimal parameters
    preds, acc, loss = mlc.run_algorithm()

    return preds, acc, loss

def plotstuff(x, y, ylabel, title, save=False, filepath=None):
    keys = y.keys()
    sns.set_theme()
    fig, ax = plt.subplots()
    for key in keys:
        ax.plot(x, y[key], label=key)
    
    ax.set(xlabel='Iterations', ylabel=ylabel,
           title=title)
    ax.legend()
    if save:
        plt.savefig(filepath)
    plt.show()


with open("accuracy_score_SGD_shitty.pkl", "rb") as f:
    shitty_accuracy = pickle.load(f)
with open("CE_enc.pkl", "rb") as f:
    CE = pickle.load(f)
with open("acc_enc.pkl", "rb") as f:
    acc = pickle.load(f)
# with open("params_ans.pkl", "rb") as f:
#     params = pickle.load(f)

print("keys: ", CE.keys())
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
bad_run = np.load("barren_run_gradient_descent.npy")
x = np.arange(len(CE["encoding = 1"]))

breakpoint()