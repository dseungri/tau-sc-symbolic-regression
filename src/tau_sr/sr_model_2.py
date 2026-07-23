# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from schemas import DEFAULT_N_ROIS
from sklearn.model_selection import train_test_split


# define variables

# load variables as torch.tensor, also need to input train_test_split on variables later

num_subjects = 32

# generate synthetic tau SUVR data
y0 = torch.rand(num_subjects, 148) * 2.0 + 0.5
dt = torch.rand(num_subjects, 1) * 2.0 + 1.0
y_target = y0 + (0.08 * y0 * dt) + torch.randn_like(y0) * 0.02

indices = list(range(len(y0)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

# Slice tensors by subject indices
train_y0, test_y0 = y0[train_idx], y0[test_idx]
train_dt, test_dt = dt[train_idx], dt[test_idx]
train_y_target, test_y_target = y_target[train_idx], y_target[test_idx]


W = torch.zeros((DEFAULT_N_ROIS, DEFAULT_N_ROIS), dtype=torch.float32)
region_indices = torch.arange(DEFAULT_N_ROIS - 1)
W[region_indices, region_indices + 1] = 1.0
W[region_indices + 1, region_indices] = 1.0
D = torch.diag(W.sum(dim=1))

L = D-W

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

def evaluate_model(model, test_y0, test_dt, test_y_target):
    # test how good the model is
    model.eval()
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        test_pred = model(test_y0, L)
        test_loss = criterion(test_pred, test_y_target)
        
    print("\n--- Test Set Evaluation ---")
    print(f"Test MSE Loss (Holdout Subjects): {test_loss.item():.6f}")
    return test_loss.item()

# symbolic regression model

def sr(model):
    from pysr import PySRRegressor

    reaction_parameter = next(model.reaction_eq.parameters())
    y_samples = torch.linspace(
        0,
        5,
        500,
        device=reaction_parameter.device,
        dtype=reaction_parameter.dtype,
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
        binary_operators=["+", "*", "-", '/'],
        unary_operators=['exp'],
        model_selection="best"
    )

    sr_model.fit(y_samples.detach().cpu().numpy(), r_pred)

    print('Symbolic Regression Model for r_m(y):')
    print(sr_model)
    return sr_model

model = ReactionDiffusion()
trained_model = train(model, train_y0, L, train_dt, train_y_target)
evaluate = evaluate_model(model, test_y0, test_dt, test_y_target)
sr_model = sr(model)