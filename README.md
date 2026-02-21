# Welcome to experiments with Bi-level regularization via iterative mesh refinement for aeroacoustics!

This python code performs the recovery of source in Helmholtz equation presented in the scientific article

[*Bi-level regularization via iterative mesh refinement for aeroacoustics*](https://link.springer.com/chapter/10.1007/978-3-031-87213-6_19)

Authors: ** Christian Aarset and Tram Thi Ngoc Nguyen**

To recreate the numerical results presented therein, install the FEM software [*NGSolve](https://ngsolve.org/index.html)
then run the Jupyter notebook

**Adaptive-FEM-refinement-Aeroacoustic.ipynb**

This code is expected to work on the majority of architectures.

Please contact [tram.nguyen.aca@gmail.com](mailto:tram.nguyen.aca@gmail.com) and up-to-date contact info on [*my homepage*](https://sites.google.com/view/tramtnnguyen/home) for any questions.

![Image of PDE residual as a function of iteration time, comparing Landwteration with single, fine fixed mesh (dotted black line) vs.~bi-level adaptively refined mesh (red line); the shaded blue background is used to visualize the number of times the bi-level algorithm has carried out adaptive mesh refinement.](Graphics/Res001.png)


