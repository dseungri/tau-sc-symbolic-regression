# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tau_sr.schemas import DEFAULT_N_ROIS

# define variables

# load variables as torch.tensor, also need to input train_test_split on variables later

W = torch.zeros((DEFAULT_N_ROIS, DEFAULT_N_ROIS), dtype=torch.float32)
region_indices = torch.arange(DEFAULT_N_ROIS - 1)
W[region_indices, region_indices + 1] = 1.0
W[region_indices + 1, region_indices] = 1.0
D = torch.diag(W.sum(dim=1))

L = D-W

# define reaction diffusion ODE using NN

class ReactionDiffusion(nn.Module):
    def __init__(self, L):
        super().__init__()

        # input non-learnable parameter for the laplacian
        self.register_buffer('L', L)

        # set the learnable parameter roe for the tranport rate
        self.roe = nn.Parameter(torch.tensor(0.1))

        # build MLP and layers using sequential NN
        self.reaction_eq = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, y):
        # y = [subjects, regions] = inital tau SUVR
            
        transport = -self.roe * torch.matmul(y, self.L.T) # transport term in ODE
        reaction = self.reaction_eq(
            y.unsqueeze(-1)
        ).squeeze(-1) 

        # add initial tau SUVR to reaction

        return transport + reaction
        
    def forward_EulIntegration(self, y0, dt, steps=10):
        # y0 = baseline tau
        # dt = time interval
        # steps = variable to improve integration accuracy

        h = dt/steps
        y = y0

        for _ in range(steps):
            dydt = self.forward(y)
            y = y+h*dydt
        return y # final predicted tau
        
def train(model, y0, dt, y_target, epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # start nn training
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # forward prop
        y_pred = model.forward_EulIntegration(
            y0,
            dt,
        )
        loss = criterion(y_pred, y_target)

        loss.backward()
        optimizer.step()
    
    return model

# symbolic regression model

def sr(model):
    from pysr import PySRRegressor

    parameter = next(model.parameters())
    y_samples = torch.linspace(
        0,
        5,
        500,
        device=parameter.device,
        dtype=parameter.dtype,
    ).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        r_pred = (
            model.reaction_eq(y_samples)
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )

    sr_model = PySRRegressor(
        niterations=30,
        binary_operators=["+", "*", "-"],
        unary_operators=[],
        model_selection="best"
    )

    sr_model.fit(y_samples.detach().cpu().numpy(), r_pred)

    print('Symbolic Regression Model for r_m(y):')
    print(sr_model)
