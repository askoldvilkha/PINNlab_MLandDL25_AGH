import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# definition of the PINN neural network
class PINN(nn.Module):
    def __init__(self, layers: list, activation: str = 'tanh'):
        super(PINN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                if activation.lower() in ['tanh', 'sigmoid', 'relu', 'softplus', 'sin']:
                    if activation.lower() == 'tanh':
                        self.net.add_module(f"activation_{i}", nn.Tanh())
                    elif activation.lower() == 'sigmoid':
                        self.net.add_module(f"activation_{i}", nn.Sigmoid())
                    elif activation.lower() == 'relu':
                        self.net.add_module(f"activation_{i}", nn.ReLU())
                    elif activation.lower() == 'softplus':
                        self.net.add_module(f"activation_{i}", nn.Softplus())
                    elif activation.lower() == 'sin':
                        class Sin(nn.Module):
                            def forward(self, x):
                                return torch.sin(x)
                        self.net.add_module(f"activation_{i}", Sin())
                else:
                    try:
                        self.net.add_module(f"activation_{i}", nn.__dict__[activation]())
                    except KeyError:
                        raise ValueError(f"Activation function '{activation}' is not recognized.")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TrainParams:
    def __init__(self, optimizer: torch.optim.Optimizer, activation: str, ic: list, layers: list):
        self.optimizer = optimizer
        self.activation = activation
        self.x_ic = ic[0]
        self.y_ic = ic[1]
        self.layers = layers
    def write_results(self, epochs: np.ndarray, loss: np.ndarray, time: float):
        self.epochs = epochs
        self.loss = loss
        self.time = time
    def write_results_long(self, epochs_arr: np.ndarray, time_arr: np.ndarray):
        self.epochs_long = epochs_arr
        self.time_long = time_arr

# training loop function
def train(params: TrainParams, ode_residual: callable, length: str = 'flash', save_results: bool = False, output_model: bool = False):
    # hyperparameters
    model = PINN(params.layers, params.activation)
    optimizer = params.optimizer(model.parameters(), lr=1e-3)
    
    # define number of epochs based on length
    if length == 'flash':
        epochs = 2500
    elif length == 'standard':
        epochs = 10000
    elif length == 'long':
        epochs = 30000
    else:
        raise ValueError("Length must be 'flash', 'standard', or 'long'.")

    # initial condition
    x_ic = torch.tensor([[params.x_ic]])
    y_ic = torch.tensor([[params.y_ic]]) 

    # training data (collocation points)
    x_colloc = torch.linspace(0, 1, 100).view(-1, 1)
    x_colloc.requires_grad = True

    if save_results:
        epochs_record = np.zeros(epochs)
        loss_record = np.zeros(epochs)
        if length == 'long':
            epochs_arr = np.zeros(epochs // 100)
            time_arr = np.zeros(epochs // 100)

    time_start = time.time()
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        y_pred = model(x_colloc)
        res = ode_residual(x_colloc, y_pred)
        loss_ode = torch.mean(res**2)

        # Initial condition loss
        y_ic_pred = model(x_ic)
        loss_ic = torch.mean((y_ic_pred - y_ic)**2)

        loss = loss_ode + loss_ic
        loss.backward()
        optimizer.step()

        if save_results:
            epochs_record[epoch] = epoch
            loss_record[epoch] = loss.item()
        
        if length == 'long':
            if epoch % 100 == 0:
                epochs_arr[epoch // 100] = epoch
                time_arr[epoch // 100] = time.time() - time_start
                
        # if epoch % 100 == 0:
        #    print(f"Epoch {epoch}, Loss: {loss.item()}, Time elapsed: {time.time() - time_start:.2f}s")
    time_end = time.time()
    time_elapsed = time_end - time_start
    print(f"Training completed in {time_elapsed:.2f}s")

    if save_results:
        params.write_results(epochs_record, loss_record, time_elapsed)
        if length == 'long':
            params.write_results_long(epochs_arr, time_arr)

    if output_model:
        return model

def traindx(params: TrainParams, 
        ode_residual: callable, 
        ic_dx: list,
        ic_dx2: list = None,
        length: str = 'flash', 
        save_results: bool = False):
    '''Same as `train` but allows to input 1st and 2nd derivative boundary conditions'''
    # hyperparameters
    model = PINN(params.layers, params.activation)
    optimizer = params.optimizer(model.parameters(), lr=1e-3)
    
    # define number of epochs based on length
    if length == 'flash':
        epochs = 2500
    elif length == 'standard':
        epochs = 10000
    elif length == 'long':
        epochs = 30000
    else:
        raise ValueError("Length must be 'flash', 'standard', or 'long'.")

    # initial condition
    x_ic = torch.tensor([[params.x_ic]])
    y_ic = torch.tensor([[params.y_ic]])

    # derivative initial conditions
    x_ic_dx = torch.tensor([[ic_dx[0]]])
    x_ic_dx.requires_grad = True
    y_ic_dx = torch.tensor([[ic_dx[1]]])

    if ic_dx2 is not None:
        x_ic_dx2 = torch.tensor([[ic_dx2[0]]])
        x_ic_dx2.requires_grad = True
        y_ic_dx2 = torch.tensor([[ic_dx2[1]]]) 

    # training data (collocation points)
    x_colloc = torch.linspace(0, 1, 100).view(-1, 1)
    x_colloc.requires_grad = True

    if save_results:
        epochs_record = np.zeros(epochs)
        loss_record = np.zeros(epochs)
        if length == 'long':
            epochs_arr = np.zeros(epochs // 100)
            time_arr = np.zeros(epochs // 100)

    time_start = time.time()
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        y_pred = model(x_colloc)
        res = ode_residual(x_colloc, y_pred)
        loss_ode = torch.mean(res**2)

        # Initial condition loss
        y_ic_pred = model(x_ic)
        loss_ic = torch.mean((y_ic_pred - y_ic)**2)

        # 1st derivative initial condition loss
        y_ic_dx_pred = torch.autograd.grad(model(x_ic_dx), x_ic_dx, 
            grad_outputs=torch.ones_like(y_ic_dx), create_graph=True)[0]
        loss_ic_dx = torch.mean((y_ic_dx_pred - y_ic_dx)**2)
        loss_ic += loss_ic_dx

        # 2nd derivative initial condition loss (if provided)
        if ic_dx2 is not None:
            y_ic_dx_pred = torch.autograd.grad(model(x_ic_dx2), x_ic_dx2, 
                grad_outputs=torch.ones_like(y_ic_dx2), create_graph=True)[0]
            y_ic_dx2_pred = torch.autograd.grad(y_ic_dx_pred, x_ic_dx2, 
                grad_outputs=torch.ones_like(y_ic_dx_pred), create_graph=True)[0]
            loss_ic_dx2 = torch.mean((y_ic_dx2_pred - y_ic_dx2)**2)
            loss_ic += loss_ic_dx2

        loss = loss_ode + loss_ic
        loss.backward()
        optimizer.step()

        if save_results:
            epochs_record[epoch] = epoch
            loss_record[epoch] = loss.item()
        
        if length == 'long':
            if epoch % 100 == 0:
                epochs_arr[epoch // 100] = epoch
                time_arr[epoch // 100] = time.time() - time_start
                
        # if epoch % 100 == 0:
        #    print(f"Epoch {epoch}, Loss: {loss.item()}, Time elapsed: {time.time() - time_start:.2f}s")
    time_end = time.time()
    time_elapsed = time_end - time_start
    print(f"Training completed in {time_elapsed:.2f}s")

    if save_results:
        params.write_results(epochs_record, loss_record, time_elapsed)
        if length == 'long':
            params.write_results_long(epochs_arr, time_arr)

    return model
