# Synthetic Robustness Sweep

## Objective

This synthetic-only robustness sweep evaluates whether the controlled tau reaction-diffusion recovery pipeline remains stable across pipeline seeds, observation noise, graph representation, and Laplacian definition.

## Prespecified Design

The study uses 23 unique one-factor-at-a-time runs: five seed runs, nine additional observation-noise runs, three binary-graph runs, and six alternative-Laplacian runs. Higher-order interactions were not tested.

Pairing is by master seed. Within a seed, the weighted topology, clean initial states, subject split, true reaction law, true rho, and base standard-normal noise draw are reused; only the declared factor changes.

## Fixed Configuration

- Subjects: 24
- Regions: 148
- Times: [0.0, 0.5, 1.0, 1.5, 2.0]
- True rho: 0.1
- True reaction: 0.5*y - 0.5*y**2
- Neural hidden width: 16
- Max epochs/patience: 1000/100
- PySR samples/maxsize: 500/15

## Canonical Reproduction Check

Passed: True. The check compares the seed-42 weighted symmetric-normalized zero-noise run against the previously reported approximate canonical values with documented deterministic tolerances.

## Run Status

Completed runs: 23. Failed runs: 0.

## Recovery Rates

- rho_within_10_percent: 20 / 23
- linear_within_10_percent: 22 / 23
- quadratic_within_10_percent: 21 / 23
- strict_symbolic_structure: 13 / 23
- correct_sign: 23 / 23
- joint_coefficient_within_10_percent: 21 / 23

## Seed Results

Seed robustness: see JSON summary for per-seed points, paired differences, and rates.

## Noise Results

Noise robustness: see JSON summary for per-seed points, paired differences, and rates.

## Graph Representation Results

Graph representation sensitivity: see JSON summary for per-seed points, paired differences, and rates.

## Laplacian Results

Laplacian sensitivity: see JSON summary for per-seed points, paired differences, and rates.

## Symbolic Structure And Coefficients

The selected symbolic equations and coefficient interpretations are listed in the CSV and JSON artifacts. Structurally incorrect or non-polynomial equations are retained as failures for strict recovery.

## Failure Cases

No failed runs were recorded.

## Runtime

Total runtime seconds: 6973.208155749831.

## Limitations

This is a synthetic study generated from the same model family being fitted. It is paired and one-factor-at-a-time, so interactions were not estimated. Controlled synthetic robustness cannot replace longitudinal real-data validation and does not establish clinical validity.

## Strongest Defensible Conclusion

Synthetic robustness was evaluated under a paired one-factor-at-a-time design with 23 completed runs and 0 failed runs. Strict symbolic structure recovery succeeded in 13 of 23 eligible completed runs; joint linear/quadratic coefficient recovery succeeded in 21 of 23 eligible completed runs. These controlled synthetic results support only the tested simulation regime and do not establish clinical validity or higher-order factor interactions.

## Run Inventory

- seed_master_40: complete, factor=seed, level=master_seed_40
- seed_master_41: complete, factor=seed, level=master_seed_41
- seed_master_42: complete, factor=seed, level=master_seed_42
- seed_master_43: complete, factor=seed, level=master_seed_43
- seed_master_44: complete, factor=seed, level=master_seed_44
- noise_0p01_seed_41: complete, factor=noise, level=noise_0.01
- noise_0p05_seed_41: complete, factor=noise, level=noise_0.05
- noise_0p1_seed_41: complete, factor=noise, level=noise_0.1
- noise_0p01_seed_42: complete, factor=noise, level=noise_0.01
- noise_0p05_seed_42: complete, factor=noise, level=noise_0.05
- noise_0p1_seed_42: complete, factor=noise, level=noise_0.1
- noise_0p01_seed_43: complete, factor=noise, level=noise_0.01
- noise_0p05_seed_43: complete, factor=noise, level=noise_0.05
- noise_0p1_seed_43: complete, factor=noise, level=noise_0.1
- graph_binary_seed_41: complete, factor=graph_representation, level=binary
- graph_binary_seed_42: complete, factor=graph_representation, level=binary
- graph_binary_seed_43: complete, factor=graph_representation, level=binary
- laplacian_random_walk_seed_41: complete, factor=laplacian, level=random_walk
- laplacian_combinatorial_seed_41: complete, factor=laplacian, level=combinatorial
- laplacian_random_walk_seed_42: complete, factor=laplacian, level=random_walk
- laplacian_combinatorial_seed_42: complete, factor=laplacian, level=combinatorial
- laplacian_random_walk_seed_43: complete, factor=laplacian, level=random_walk
- laplacian_combinatorial_seed_43: complete, factor=laplacian, level=combinatorial
