# import libraries
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from schemas import DEFAULT_N_ROIS
from sklearn.model_selection import train_test_split
import numpy as np

from synthetic import (
    SyntheticCohortConfig,
    build_connectome_representations,
    generate_count_connectome,
    generate_edge_attributes,
    generate_synthetic_roi_labels,
)

# Set global random seeds for reproducibility
TORCH_SEED = 42
torch.manual_seed(TORCH_SEED)
np.random.seed(TORCH_SEED)

def load_synthetic_dataset(config: SyntheticCohortConfig):
    """Generates structural graph representations and synthetic longitudinal Tau data."""
    rng = np.random.default_rng(config.seed)

    # Generate graph connectivity structure
    count_matrix = generate_count_connectome(
        config.n_regions, rng, edge_probability=config.edge_probability
    )
    fa_matrix, length_matrix = generate_edge_attributes(count_matrix, rng)
    connectomes = build_connectome_representations(
        count_matrix, fa_matrix, length_matrix
    )

    # Build Graph Laplacian (L = D - A) using length-adjusted network
    A = connectomes["length_adjusted"]
    D = np.diag(A.sum(axis=1))
    L_np = D - A
    L_tensor = torch.from_numpy(L_np).float()

    # Generate baseline Tau SUVR [N_subjects, N_regions]
    y0_np = rng.uniform(
        config.baseline_tau_min,
        config.baseline_tau_max,
        size=(config.n_subjects, config.n_regions),
    )
    y0 = torch.from_numpy(y0_np).float()

    # Time intervals between visits (e.g., 1 to 3 years)
    dt = torch.from_numpy(
        rng.uniform(1.0, 3.0, size=(config.n_subjects, 1))
    ).float()

    # Create Ground Truth Target Data using a known biological reaction law:
    # r_true(y) = 0.15 * y * (1 - y / 3.5)
    # Target is generated via simple forward Euler steps
    with torch.no_grad():
        y_sim = y0.clone()
        n_steps = 20
        for i in range(config.n_subjects):
            step_size = dt[i].item() / n_steps
            for _ in range(n_steps):
                # dy/dt = -0.05 * (y @ L^T) + 0.15 * y * (1 - y / 3.5)
                diffusion = -0.05 * torch.matmul(y_sim[i : i + 1], L_tensor.T)
                reaction = 0.15 * y_sim[i : i + 1] * (1 - y_sim[i : i + 1] / 3.5)
                dydt = diffusion + reaction
                y_sim[i : i + 1] += step_size * dydt

        if config.add_observation_noise:
            y_sim += torch.randn_like(y_sim) * config.observation_noise_std

    y_target = y_sim

    # Fetch ROI labels for regional indexing
    roi_labels = generate_synthetic_roi_labels(config)

    return L_tensor, y0, dt, y_target, roi_labels

config = SyntheticCohortConfig(
        n_subjects=24,
        n_regions=148,
        visits_per_subject=2,
        seed=42,
        baseline_tau_min=0.2,
        baseline_tau_max=3.0,
    )

L, y0, dt, y_target, roi_labels = load_synthetic_dataset(config)

indices = list(range(len(y0)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=TORCH_SEED)

# Slice tensors by subject indices
train_y0, test_y0 = y0[train_idx], y0[test_idx]
train_dt, test_dt = dt[train_idx], dt[test_idx]
train_y_target, test_y_target = y_target[train_idx], y_target[test_idx]

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
