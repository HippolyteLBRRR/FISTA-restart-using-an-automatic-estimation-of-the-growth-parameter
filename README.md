# FISTA-restart-using-an-automatic-estimation-of-the-growth-parameter

This repository contains the codes producing the numerical experiments of the article "FISTA restart using an automatic estimation of the growth parameter" by J-F. Aujol, C. Dossal, H. Labarrière and A. Rondepierre (which can be found at https://hal.archives-ouvertes.fr/hal-03153525).

The repository is organized as follows:

* algorithms.py is the file containing all the methods. We insist on the fact that each method is implemented for any given function $F$ and consequently, the codes may not be optimal. Note that this affects the computational cost of each iteration but comparisons of the error w.r.t the number of iterations are still valid.
* Numerical_experiments_LS.ipynb is a notebook where each method is applied to a LASSO problem.
* Numerical_experiments_Inpainting.ipynb is a notebook where each method is applied to an inpainting problem.
* visualizer.py is a file helping visualizing the performance of each method.

If you have any questions or comments, please contact me at labarrie@insa-toulouse.fr.
