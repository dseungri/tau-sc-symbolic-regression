from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pysr import PySRRegressor

# set random seeds for data generation
TORCH_SEED = 42
torch.manual_seed(TORCH_SEED)
np.random.seed(TORCH_SEED)


# load variables from excel file
def load_tau_pet_excel(file_path: str):
    df = pd.read_excel(file_path)

    # Parse dates and sort chronologically per subject
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"].astype(str))
    df = df.sort_values(by=["PTID", "EXAMDATE"]).reset_index(drop=True)

    # Identify node columns (Node 1 through Node 160)
    node_cols = [c for c in df.columns if c.startswith("Node")]

    y0_list, dt_list, y_target_list, metadata = [], [], [], []

    for ptid, group in df.groupby("PTID"):
        scans = group.to_dict("records")
        for i in range(len(scans) - 1):
            b_scan = scans[i]  # Baseline/earlier scan
            f_scan = scans[i + 1]  # Follow-up scan

            days_diff = (f_scan["EXAMDATE"] - b_scan["EXAMDATE"]).days
            years_diff = days_diff / 365.25

            if years_diff <= 0:
                continue

            y0_vals = [b_scan[node] for node in node_cols]
            y_target_vals = [f_scan[node] for node in node_cols]

            y0_list.append(y0_vals)
            dt_list.append([years_diff])
            y_target_list.append(y_target_vals)

            metadata.append(
                {
                    "PTID": ptid,
                    "baseline_date": b_scan["EXAMDATE"].strftime("%Y-%m-%d"),
                    "followup_date": f_scan["EXAMDATE"].strftime("%Y-%m-%d"),
                    "dt_years": round(years_diff, 3),
                }
            )

    y0 = torch.tensor(y0_list, dtype=torch.float32)
    dt = torch.tensor(dt_list, dtype=torch.float32)
    y_target = torch.tensor(y_target_list, dtype=torch.float32)
    meta_df = pd.DataFrame(metadata)

    return y0, dt, y_target, meta_df, node_cols


# generate laplacian (since we don't have MRI data)
def generate_synthetic_laplacian(
    n_regions: int, seed: int = 42, edge_probability: float = 0.08
) -> torch.Tensor:
    """Generates a synthetic length-adjusted Graph Laplacian L [N, N]."""
    rng = np.random.default_rng(seed)

    # Adjacency edge mask
    edge_mask = rng.random((n_regions, n_regions)) < edge_probability
    edge_mask = np.triu(edge_mask, k=1)

    chain_idx = np.arange(n_regions - 1)
    edge_mask[chain_idx, chain_idx + 1] = True

    # Fiber count and tract lengths
    weights = rng.uniform(0.1, 1.0, size=(n_regions, n_regions))
    count_matrix = np.triu(weights * edge_mask, k=1)
    count_matrix = count_matrix + count_matrix.T

    length_matrix = np.zeros_like(count_matrix)
    support = count_matrix > 0
    upper_support = np.triu(support, k=1)
    rows, cols = np.nonzero(upper_support)

    length_vals = rng.uniform(20.0, 150.0, size=len(rows))
    length_matrix[rows, cols] = length_vals
    length_matrix[cols, rows] = length_vals

    # Length adjustment: count / scaled length
    pos_lengths = length_matrix[upper_support]
    med_length = np.median(pos_lengths)
    scaled_length = np.zeros_like(length_matrix)
    scaled_length[support] = length_matrix[support] / med_length

    length_adjusted = np.zeros_like(count_matrix)
    length_adjusted[support] = count_matrix[support] / (
        scaled_length[support] + 1e-8
    )
    np.fill_diagonal(length_adjusted, 0.0)

    # Compute Graph Laplacian L = D - A
    D = np.diag(length_adjusted.sum(axis=1))
    L = D - length_adjusted

    return torch.from_numpy(L).float()


# neural network for ODE and different solvers
class ReactionDiffusionODE(nn.Module):
    def __init__(self, L_matrix: torch.Tensor):
        super().__init__()
        self.register_buffer("L", L_matrix)

        # Global diffusion coefficient
        self.rho_raw = nn.Parameter(torch.tensor(0.05))

        # Local kinetic network r_m(y)
        self.r_m = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    @property
    def rho(self) -> torch.Tensor:
        return nn.functional.softplus(self.rho_raw)

    def forward_dynamics(self, y: torch.Tensor) -> torch.Tensor:
        """dy/dt = -rho * (L @ y^T)^T + r_m(y)"""
        diffusion = -self.rho * torch.matmul(y, self.L.T)

        B, N = y.shape
        y_flat = y.view(-1, 1)
        reaction_flat = self.r_m(y_flat)
        reaction = reaction_flat.view(B, N)

        return diffusion + reaction

    def integrate_euler(
        self, y0: torch.Tensor, dt: torch.Tensor, steps: int = 10
    ) -> torch.Tensor:
        """Euler Integration Solver."""
        y_outputs = []
        B = y0.shape[0]

        for i in range(B):
            h = dt[i] / steps
            y_i = y0[i : i + 1]
            for _ in range(steps):
                y_i = y_i + h * self.forward_dynamics(y_i)
            y_outputs.append(y_i)

        return torch.cat(y_outputs, dim=0)

    def integrate_rk4(
        self, y0: torch.Tensor, dt: torch.Tensor, steps: int = 10
    ) -> torch.Tensor:
        """4th-order Runge-Kutta Solver."""
        y_outputs = []
        B = y0.shape[0]

        for i in range(B):
            h = dt[i] / steps
            y_i = y0[i : i + 1]
            for _ in range(steps):
                k1 = self.forward_dynamics(y_i)
                k2 = self.forward_dynamics(y_i + 0.5 * h * k1)
                k3 = self.forward_dynamics(y_i + 0.5 * h * k2)
                k4 = self.forward_dynamics(y_i + h * k3)
                y_i = y_i + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            y_outputs.append(y_i)

        return torch.cat(y_outputs, dim=0)


# train model
def train_model(
    model: nn.Module,
    y0: torch.Tensor,
    dt: torch.Tensor,
    y_target: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-2,
    solver: str = "euler",
    steps: int = 10,
) -> nn.Module:
    """Trains the ReactionDiffusionODE model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    print(
        f"\nStarting training ({epochs} epochs, solver='{solver}', lr={lr})..."
    )

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        if solver.lower() == "euler":
            y_pred = model.integrate_euler(y0, dt, steps=steps)
        else:
            y_pred = model.integrate_rk4(y0, dt, steps=steps)

        loss = criterion(y_pred, y_target)
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0 or epoch == epochs:
            print(
                f"   Epoch {epoch:03d}/{epochs} | MSE Loss: {loss.item():.6f} | Learned Rho: {model.rho.item():.4f}"
            )

    return model


# symbolic regression function
def sr(
    model: nn.Module,
    y_min: float = 0.5,
    y_max: float = 2.5,
    n_samples: int = 300,
    niterations: int = 30,
) -> PySRRegressor:
    """Probes the learned local reaction MLP and fits symbolic expressions via PySR."""
    print("\nProbing learned kinetic network r_m(y)...")
    model.eval()

    with torch.no_grad():
        y_probe = torch.linspace(y_min, y_max, n_samples).unsqueeze(-1)
        r_pred = model.r_m(y_probe).numpy().squeeze()
        y_samples = y_probe.numpy().squeeze()

    print("Fitting symbolic equations with PySR...")
    pysr_model = PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["exp"],
        variable_names=["y"],
        model_selection="best",
        progress=False,
        temp_equation_file=True,
        delete_tempfiles=True,
    )

    pysr_model.fit(y_samples.reshape(-1, 1), r_pred)
    return pysr_model


# run entire sr and NN
def run_pipeline(excel_path: str = "pilot_tau_workbook.xlsx", solver: str = "euler"):
    # Step 1: Load Data
    y0, dt, y_target, metadata, node_names = load_tau_pet_excel(excel_path)
    n_nodes = y0.shape[1]

    print(
        f"1. Data loaded successfully ({y0.shape[0]} pairs, {n_nodes} nodes)."
    )

    # Step 2: Build Laplacian
    L_matrix = generate_synthetic_laplacian(n_regions=n_nodes, seed=42)
    print(f"2. Constructed synthetic {n_nodes}x{n_nodes} Graph Laplacian.")

    # Step 3: Instantiate & Train
    model = ReactionDiffusionODE(L_matrix)
    model = train_model(
        model,
        y0,
        dt,
        y_target,
        epochs=200,
        lr=1e-2,
        solver=solver,
        steps=10,
    )

    # Step 4: Symbolic Regression
    pysr_model = sr(
        model,
        y_min=y0.min().item(),
        y_max=y0.max().item(),
        niterations=30,
    )

    # Step 5: Print Results
    print("\n" + "=" * 60)
    print("DISTILLED LOCAL REACTION EQUATION:")
    print("=" * 60)
    best_eq = pysr_model.get_best()
    print(f"r_m(y) = {best_eq['equation']}")
    print("=" * 60)

    return model, pysr_model

model, pysr_model = run_pipeline(r"pilot_tau_workbook.xlsx", solver="rk4")