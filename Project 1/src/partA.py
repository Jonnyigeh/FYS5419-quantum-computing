import numpy as np
import matplotlib.pyplot as plt

def init_basis() -> np.array:
    """
    Set the basis for the qubits, using the two dimensional
    computational basis |0> = [1,0] and |1> = [0,1], i.e the Z-basis.
    """
    q0 = np.array([1, 0])       # |0>
    q1 = np.array([0, 1])       # |1>    
    
    return q0, q1

def hadamard_gate(qubit: np.array) -> np.array:
    """
    Apply the Hadamard gate to the qubit.
    """
    H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
    qubit = H @ qubit
    
    return qubit

def phase_gate(qubit: np.array) -> np.array:
    """
    Apply the phase gate to the qubit.
    """
    S = np.array([[1, 0], [0, 1j]])
    qubit = S @ qubit
    
    return qubit

def cnot_gate(control_qubit: np.array, target_qubit: np.array) -> (np.array, np.array):
    """
    Apply the 2 qubit CNOT gate to the target and control qubit
    """
    if np.all(control_qubit == np.array([0,1])):        # Checks if control qubit is infact |1>, then applies the pauli x gate
        target_qubit = pauli_x(target_qubit)            # If not, returns the original qubits
    

    return control_qubit, target_qubit

def pauli_x(qubit: np.array) -> np.array:
    """
    Apply the Pauli-X gate to the qubit.
    """
    X = np.array([[0, 1], [1, 0]])
    qubit = X @ qubit
    
    return qubit

def pauli_y(qubit: np.array) -> np.array:
    """
    Apply the Pauli-Y gate to the qubit.
    """
    Y = np.array([[0, -1j], [1j, 0]]) 
    qubit = Y @ qubit
    
    return qubit

def pauli_z(qubit: np.array) -> np.array:
    """
    Apply the Pauli-Z gate to the qubit.
    """
    Z = np.array([[1, 0], [0, -1]])    
    qubit = Z @ qubit
    
    return qubit

def bell_states() -> (np.array, np.array, np.array, np.array):
    """
    Sets up the Bell states, using the 2D computational basis.
    """
    q0, q1 = init_basis()
    phi_plus = 1 / np.sqrt(2) * (np.kron(q0, q0) + np.kron(q1, q1))
    phi_minus = 1 / np.sqrt(2) * (np.kron(q0, q0) - np.kron(q1, q1))

    psi_plus = 1 / np.sqrt(2) * (np.kron(q0, q1) + np.kron(q1, q0))
    psi_minus = 1 / np.sqrt(2) * (np.kron(q0, q1) - np.kron(q1, q0))

    return phi_plus, phi_minus, psi_plus, psi_minus

def hadamard_gate_bell(bellstate: np.array) -> np.array:
    """
    Apply the Hadamard gate to the first qubit in a Bell state
    """
    H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
    new_state = np.kron(H, np.eye(2)) @ bellstate
    
    return new_state

def cnot_gate_bell(bellstate: np.array) -> np.array:
    """
    Apply the CNOT gate to a Bell state using the convention 
    that qubit 1 is the control qubit and qubit 2 is the target qubit.
    """
    X = np.array([[0, 1], [1, 0]])
    if (bellstate[2] or bellstate[3]) == 1:        # Checks if control qubit is infact |1>, then applies the pauli x gate
        new_state = np.kron(np.eye(2), X) @ bellstate

        return new_state
    return bellstate

def bell_measurements(bellstate: np.array, n_measurements:  int) -> np.array:
    """
    Perform measurements on a Bell state, which have the following
    First index = |00>
    Second index = |01>
    Third index = |10>
    Fourth index = |11>
    """
    measurements = np.zeros(n_measurements)
    prob = np.abs(bellstate)**2
    for i in range(n_measurements):
        measurements[i] = np.random.choice([0, 1, 2, 3], p=prob)    # Picks a random index according to the probability from the Bell state

    return measurements

def plot_bell_measurements(measurements: np.array) -> None:
    """
    Plot the measurements of a Bell state.
    """
    fig, ax = plt.subplots()
    ax.hist(measurements, bins=[0, 1, 2, 3, 4], align="left")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([r'$|00\rangle$', r'$|01\rangle$', r'$|10\rangle$', r'$|11\rangle$'])
    ax.set_xlabel('Basis')
    ax.set_ylabel('Counts')
    plt.show()



if __name__ == "__main__":
    # Setting up the one qubit basis
    q0, q1 = init_basis()
    # Applying the Pauli gates to the qubits
    q0x = pauli_x(q0)
    q0y = pauli_y(q0)
    q0z = pauli_z(q0)
    q1x = pauli_x(q1)
    q1y = pauli_y(q1)
    q1z = pauli_z(q1)
    # Applying the Hadamard and Phase gates to the qubits
    q0h = hadamard_gate(q0)
    q1h = hadamard_gate(q1)
    q0s = phase_gate(q0)
    q1s = phase_gate(q1)
    # Defining the Bell states
    phi_plus, phi_minus, psi_plus, psi_minus = bell_states()
    # Applying the Hadamard and CNOT gates to the Phi+ Bell state
    H_phi_plus = hadamard_gate_bell(phi_plus)
    cnot_phiplus = cnot_gate_bell(phi_plus)
    # Performing measurements on the Phi+ Bell state
    measurements = bell_measurements(phi_plus, 1000)
    # Plotting the measurements
    breakpoint()
    plot_bell_measurements(measurements)    
    

