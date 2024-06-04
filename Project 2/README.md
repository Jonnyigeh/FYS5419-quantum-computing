## Repository for the 2nd project in the FYS5419 Course: Quantum Machine Learning

### Description
In this project we've implemented a quantum circuit for classification of the Iris dataset, a well-known dataset used for building, and testing, classification algorithms. The project dives into the importance of a good encoding of the feature values, as well as how one chooses the variational ansatz.
We have been succesful in reaching good accuracy rates on the training data, however, our simple model fails at predicting on unseen data. Further work would include looking at more complex circuits that would generalise better.
### Source code
All the code that have been used in the project is found in src/quantum_machine, and the data from the various runs is located and ready for use in src/quantum_machine/data. The main program is ML_circuit.py, which calls the helper-functions located inside the utils/ folder.

### Report
The report can be found in full in doc/main.pdf, with the individual sections and figures located in doc/setions and /figs respectively.
