---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# When should I use VD?

## Advantages

- VD can exponentially reduce the contributions of non-dominant eigenvectors in the density matrix, bringing the effective state closer to the ideal noise-free state as the number of copies M increases.

- VD can be used complementarily with other error mitigation and calibration techniques, allowing for higher levels of errors reduction.




## Disadvantages

 - Virtual Distillation requires multiple copies of a single state, which will increase the number of required qubits by a factor of M. 

- Real errors and noise on the hardware will lead to population in states that are not orthogonal to the target state, which results in a drift of the dominant eigenvector. This results in limiting the potential error mitigation of VD:  instead of a quadratic suppression in errors there will be a constant factor improvement instead {cite}`Huggins_2021`.

- Currently the implementation of VD only allows for measurement on the Z observable and will not function as desired if non-Z observables are passed to it.

- The VD implementation occasional results in a divide by zero error. This occurs only when the number of iterations is even and typically when that number is fairly low (under 100). In order to circumvent this known error, only run the function with an odd number of K-iterations.