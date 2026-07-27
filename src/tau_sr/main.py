from __future__ import annotations
import argparse

import pandas as pd
import numpy as np
import torch

from tau_sr.graphs import build_graph_laplacian
from tau_sr.sr_model_3 import train_model
from tau_sr.symbolic import (
    SymbolicRegressionConfig,
    fit_symbolic_reaction,
    sample_neural_reaction,
)
from tau_sr.synthetic import SyntheticCohortConfig, build_connectome_representations, generate_count_connectome, generate_edge_attributes
from tau_sr.neural import NeuralReactionDiffusion

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

def create_synthetic_laplacian_from_config(
    config: SyntheticCohortConfig | None = None,
    network_type: str = "length_adjusted",
    kind: str = "unnormalized",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generates synthetic connectomes and converts a selected network representation

    into a Graph Laplacian.

    Parameters
    ----------
    config : SyntheticCohortConfig, optional
        Configuration holding n_regions, seed, edge_probability, etc.
    network_type : str
        Which network representation to build the Laplacian from:
        'count', 'fa_weighted', or 'length_adjusted'.
    kind : str
        Laplacian type passed to build_graph_laplacian ('unnormalized' or 'normalized').

    Returns
    -------
    L : np.ndarray
        The computed (N, N) Graph Laplacian matrix.
    networks : dict[str, np.ndarray]
        Dictionary containing all generated adjacency representations.
    """
    if config is None:
        config = SyntheticCohortConfig()

    rng = np.random.default_rng(config.seed)

    # 1. Generate base count matrix
    count_matrix = generate_count_connectome(
        n_regions=config.n_regions,
        rng=rng,
        edge_probability=config.edge_probability,
    )

    # 2. Generate FA and tract lengths matching count support
    fa_matrix, length_matrix = generate_edge_attributes(
        count_matrix=count_matrix, rng=rng
    )

    
    networks = build_connectome_representations(
        count_matrix=count_matrix,
        fa_matrix=fa_matrix,
        length_matrix=length_matrix,
    )

    if network_type not in networks:
        raise ValueError(
            f"Invalid network_type '{network_type}'. Choose from {list(networks.keys())}."
        )

    adjacency = networks[network_type]
    L = build_graph_laplacian(adjacency, kind=kind)
    L = torch.from_numpy(L)
    return L

L = create_synthetic_laplacian_from_config()

def main(L):
    parser = argparse.ArgumentParser(
        description="Graph Neural ODE Tau-PET Pipeline"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=r"CHANGE TO FILEPATH FOR EXCEL FILE",
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

    NNmodel = NeuralReactionDiffusion(hidden_width=16, initial_rho=0.1)

    # 3. Fit Neural ODE Model
    print(
        f"3. Fitting recovery model using '{args.solver}' solver for {args.epochs} epochs..."
    )
    model = train_model(
        model=NNmodel,
        laplacian=L,
        y0=y0,
        dt=dt,
        y_target=y_target,
        epochs=args.epochs,
        lr=args.lr,
        solver=args.solver,
    )

    # configure variables for symbolic regression
    ref_param = next(model.parameters())

    samples = sample_neural_reaction(
    model=model,
    domain_min=float(y_target.min()),
    domain_max=float(y_target.max()),
    n_samples=500,
    device=ref_param.device,
    dtype=ref_param.dtype,
    )

    # Step B: Configure symbolic regression parameters
    config = SymbolicRegressionConfig(
        niterations=100,
        populations=15,
        population_size=33,
        maxsize=15,
        seed=42,
        model_selection="best",
    )

    # 4. Perform Symbolic Regression
    print("4. Distilling reaction kinetics using PySR...")
    pysr_model = fit_symbolic_reaction(samples=samples, config=config)

    # 5. Display Best Equation
    print("\n" + "=" * 60)
    print("DISTILLED LOCAL REACTION EQUATION:")
    print("=" * 60)
    best_eq = pysr_model.equation_string
    print(f"Best Symbolic Equation: {best_eq}")
    print("=" * 60)

main(L=L)