### GitHub Repository folder for the first project in the spring course FYS5419: Quantum Computing and Machine Learning

The projects task can be found at in doc/Project1.pdf. 

In this project we've successfully implemented the most basic concecpts in quantum computing, the computational basis and various single- and two-qubit gates. We've used this basis, and these gates - together with clever Pauli encoding, to solve for the ground state energies in various Hamiltonian systems using the VQE algorithm. We've followed closely an article from Physics Review C volume 106, \url{https://link.aps.org/doi/10.1103/PhysRevC.106.024319} - where they study the Lipkin model, and provides both the Pauli encoding, as well as the proper ansatz to use in their encoding schemes. 

## Source code
The structure of our code is built taskwise, and we've made use of inheritance (to some degree) to build the VQE classes for the various subtasks. 
Solution, and the production of the plots for task A is found in partA.py - for task B in partB.py etc. All files related to the numerical implementation is 
found in the 'src' folder.

For the final VQE tasks, we've made a run of our code once and saved the output to json files, which can be found also in doc/ - these are then called in the partXplot.py files. 
The reason for this is the slow computation time for optimization of the VQE algorithm. 

# Report
The report in full can be found in doc/main.pdf, with all the individual sections found in /doc/sections. All the figures used will be located in doc/figs/