# PINNlab_MLandDL25_AGH
This repo contains a simple PINN model with various tests. Completed as a project for the Machine Learning and Deep Learning Class in the summer 2025 semester at AGH.

Contents:

* `pinn_lib.py` - source code with a PINN model. Designed to solve ODEs with 1 initial condition; however, still capable of solving 2nd and 3rd order ODEs, with just 1 initial condition.
  I. e. PINN model will not break and still produce a solution, but the accuracy may be worse. The model can be modified to include more boundary conditions in the future.
  In this version, the simplest architecture is adopted in order to focus on the efficiency testing of the building components (activation functions, optimizers, number of neurons per layer, etc).
  Components:
  1) `PINN` class - Physics Informed Neural Network, requires the list with the number of neurons in each layer (`[1, 25, 25, 1]` will create a neural network with 2 hidden layers, 25 neurons each).
  The user can also input an activation function of their choice (default is `tanh`). Native support for `tanh`, `sin`, `sigmoid`, `relu`, `softplus`. 
  Other activation functions should be entered with exactly the same name as they are listed in `torch.nn` class (including letter capitalization).

  2) `TrainParams` class - a simple container class to separately store the input parameters for the PINN model before the training process. Separation is chosen to ensure the uniqueness of each subsequent training,
  as well as store the parameters that only depend on the specific test (initial conditions).
  This class can also save the training results, such as loss, number of epochs, and timing (with `write_results*` methods).

  3) `train` method - performs training of the PINN model. Requires as input `TrainParams` like object, `ode_residual` function (defined for the ODE equation).
     Performs training for: 2500 epochs (`flash`), 10000 epochs (`standard`), 30000 epochs (`long`).
     Outputs the timing of the training and can save results if requested by the user.

* `start.ipynb` - Jupyter notebook with some initial source code similar to or the same as in `pinn_lib.py`.
  Should be used as a testing ground for the PINN architecture modifications or any additions to the source code before deploying.

* `test_space.ipynb` - Jupyter notebook with tests and experiments on the PINN model from the valid source code.
  This notebook should be used for the experiments on the PINN accuracy or activation function/optimizer choice using the architecture previously tested in the `start.ipynb` notebook.
  For this project, a series of tests has been completed. 5 ODEs of the 1st, 2nd, and 3rd order were tested for accuracy and timing.
  Additional examples include: 3rd order ODE with higher computational complexity, 1st order ODE without analytical solution, 1st and 2nd order ODEs with intentionally wrong initial conditions.
  Also, three tests have been completed to look into the effects of the architecture elements. Specifically:
  1) Test 1 - Training time and loss depending on the number of neurons per layer, and the activation function.
  2) Test 2 - Training time and loss depending on the combination of activation functions and optimizers with a fixed number of neurons. 
  3) Test 3 - Accuracy comparison for the combination of activation functions and optimizers with a fixed number of neurons.
  For more detailed descriptions, please see the notebook `test_space.ipynb`. 
