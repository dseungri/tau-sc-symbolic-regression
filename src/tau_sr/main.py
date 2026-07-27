from __future__ import annotations
import argparse

import pandas as pd

from SR_Project.graphs import build_graph_laplacian
from SR_Project.training import fit_noise_free_recovery_model
from SR_Project.symbolic import fit_symbolic_reaction

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

def main():
    parser = argparse.ArgumentParser(
        description="Graph Neural ODE Tau-PET Pipeline"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="Tau_PET.xlsx",
        help="Path to Excel dataset",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="euler",
        choices=["euler", "rk4"],
        help="ODE solver method",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Learning rate"
    )
    args = parser.parse_args()

    # 1. Load Data
    print(f"1. Loading PET data from: {args.data_path}")
    y0, dt, y_target, metadata, node_names = load_tau_pet_excel(args.data_path)
    n_pairs, n_nodes = y0.shape
    print(f"   Loaded {n_pairs} scan pairs across {n_nodes} regions.")

    # 2. Build Graph Laplacian Matrix
    print(f"2. Building {n_nodes}x{n_nodes} Graph Laplacian matrix...")
    L_matrix = build_graph_laplacian(n_regions=n_nodes, seed=42)

    # 3. Fit Neural ODE Model
    print(
        f"3. Fitting recovery model using '{args.solver}' solver for {args.epochs} epochs..."
    )
    model = fit_noise_free_recovery_model(
        L_matrix=L_matrix,
        y0=y0,
        dt=dt,
        y_target=y_target,
        epochs=args.epochs,
        lr=args.lr,
        solver=args.solver,
    )

    # 4. Perform Symbolic Regression
    print("4. Distilling reaction kinetics using PySR...")
    pysr_model = fit_symbolic_reaction(
        model=model,
        y_min=y0.min().item(),
        y_max=y0.max().item(),
        niterations=30,
    )

    # 5. Display Best Equation
    print("\n" + "=" * 60)
    print("DISTILLED LOCAL REACTION EQUATION:")
    print("=" * 60)
    best_eq = pysr_model.get_best()
    print(f"r_m(y) = {best_eq['equation']}")
    print("=" * 60)