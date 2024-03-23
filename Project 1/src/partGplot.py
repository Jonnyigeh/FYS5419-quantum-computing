import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

if __name__ == "__main__":
    # Load the eigenvalues, both exact and VQE
    with open("../doc/j2_eigenvalues.json", "r") as file:
        eigenvalues = json.load(file)
    with open("../doc/estimated_j2_eigenvalues.json", "r") as file:
        estimated_eigenvalues = json.load(file)
    
    # Plot the estimated eigenvalues against the exact eigenvalues
    sns.set_theme()
    fig, ax = plt.subplots()
    x_axis = eigenvalues.keys()
    ax.scatter(x_axis, eigenvalues.values(), label="Exact eigenvalues", color="tomato", marker="x")
    ax.plot(x_axis, estimated_eigenvalues.values(), label="VQE estimated eigenvalues", color="cornflowerblue", linestyle="--")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Energy eigenvalue")
    ax.set_title(r"Exact eigenvalues vs. VQE estimated eigenvalues")
    ax.legend()
    # fig.savefig("../doc/figs/energy_eigenvalues_Lipkin_VQE.pdf")
    plt.show()