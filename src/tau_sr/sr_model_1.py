# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# define reaction diffusion ODE using NN

class ReactionDiffusion(nn.Module):
    def __init__(self):
        super().__init__()

        # set the unconstrained transport-rate parameter
        self.raw_rho = nn.Parameter(torch.tensor(-2.0))

        # build MLP and layers using sequential NN
        self.reaction_eq = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, y: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        # y = [subjects, regions] = inital tau SUVR

        if laplacian.ndim == 2:
            transport = -self.rho * torch.matmul(
                y,
                laplacian.T,
            )
        elif laplacian.ndim == 3:
            transport = -self.rho * torch.einsum(
                "bij,bj->bi",
                laplacian,
                y,
            )
        else:
            raise ValueError(
                "laplacian must have shape [R, R] "
                "or [B, R, R]."
            )
        reaction = self.reaction_eq(
            y.unsqueeze(-1)
        ).squeeze(-1) 

        # add initial tau SUVR to reaction

        return transport + reaction

    @property
    def rho(self):
        return F.softplus(self.raw_rho)
        
    def integrate_euler(self, y0, laplacian, dt, steps=10):
        # y0 = baseline tau
        # laplacian = subject-specific graph laplacian
        # dt = time interval
        # steps = variable to improve integration accuracy

        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < 1
        ):
            raise ValueError(
                "steps must be a positive integer."
            )

        h = dt/steps
        y = y0

        for _ in range(steps):
            dydt = self.forward(
                y,
                laplacian,
            )
            y = y+h*dydt
        return y # final predicted tau
        
def train(model, y0, laplacian, dt, y_target, epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # start nn training
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # forward prop
        y_pred = model.integrate_euler(
            y0,
            laplacian,
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
    return sr_model
