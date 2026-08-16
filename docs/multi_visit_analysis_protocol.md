# Multi-Visit Longitudinal Tau Analysis Protocol

Status: frozen pre-data software protocol. No real-data run occurred during this phase.

This protocol defines the infrastructure for a future limited-cohort analysis
with three or more tau PET observations per participant. It does not revise the
existing frozen two-timepoint cohort analysis and must not be used to make
scientific claims before the future private data are received and quality
controlled.

## Scientific Question

Does the availability of three or more longitudinal tau PET observations provide
evidence that nonlinear local reaction dynamics improve final-visit prediction
over linear reaction dynamics?

## Eligibility

- At least three valid tau PET observations per included participant.
- Exactly one matching 160-by-160 structural-connectivity matrix.
- A 160-region tau vector at every included visit.
- Strictly increasing elapsed times after sorting visits by elapsed year.
- Consistent atlas, region order, SUVR normalization, and reference region.
- Finite structural-connectivity, tau, and elapsed-time values.
- No zero-degree graph nodes after preprocessing.

## Data Handling

- Use actual elapsed years, not visit numbers.
- Treat structural connectivity as static across visits.
- Symmetrize structural connectivity using `(W + W.T) / 2`.
- Set the structural-connectivity diagonal to zero.
- Preserve normalized positive connections without thresholding.
- Use the symmetric-normalized graph Laplacian as the primary representation.
- Sort visits by elapsed time.
- Use neutral internal participant labels in private and aggregate artifacts.
- Do not store original filenames, source identifiers, exact dates, private
  paths, raw vectors, or subject-level records in tracked artifacts.

## Prediction Task

Global model parameters are trained using eligible training participants only.
For an evaluated participant, only observations preceding the final visit are
permissible. The final visit is forecast from the penultimate observed state
over the actual elapsed interval to the final visit.

The evaluated final visit must never be used for training, early stopping, seed
selection, feature construction, symbolic fitting, normalization, or PySR
domain construction.

## Models

1. Prior-visit persistence.
2. Linear extrapolation using the two latest permissible earlier visits.
3. Diffusion-only reaction-diffusion, `r(y) = 0`.
4. Linear reaction-diffusion, `r(y) = a*y`.
5. Linear-quadratic reaction-diffusion, `r(y) = a*y + b*y^2`.
6. Zero-anchored neural reaction-diffusion.
7. Fold-specific PySR distillation of the selected neural reaction.

## Comparisons

Primary comparison: Quadratic minus linear.

Secondary comparisons:

- Linear minus persistence.
- Quadratic minus persistence.
- Neural minus linear.
- Neural minus quadratic.
- Symbolic minus neural.

Negative model-A-minus-model-B differences favor model A.

## Metrics

- Final-visit aggregate RMSE and MAE.
- Participant-level RMSE and MAE.
- Paired participant-bootstrap RMSE and MAE differences with 95% confidence
  intervals.
- Numerical-instability counts.
- Fold-level rho and reaction coefficients.
- Neural-seed sensitivity.
- Symbolic equation, complexity, loss, coefficients, and structure by fold.
- Symbolic-versus-neural reaction RMSE and MAE.

## Interpretation

A lower point estimate alone does not establish improvement. Improvement
requires compatible paired uncertainty and stability across folds. Equation
structure or coefficients must not be presented as a population law if they are
unstable across folds.

Results from fewer than 15 eligible participants are feasibility-only. Results
from 15-29 participants are exploratory. The software must not automatically
convert those categories into stronger claims.

## Prespecified Configuration

- Outer folds: 5, participant-level, deterministic seed 42.
- Inner validation: participant-level and drawn only from outer-training
  participants.
- Bootstrap: participant-level paired resampling, 10,000 samples, seed 42.
- Parametric optimizer: Adam.
- Parametric learning rate: 0.01.
- Parametric maximum epochs: 1000.
- Parametric patience: 100.
- RK4 maximum step: 0.05 years.
- Neural width: 16.
- Neural initial rho: 0.05.
- Neural learning rate: 1e-3.
- Neural seeds: 41, 42, and 43.
- Neural precision/device: float64 CPU.
- Neural seed selection: inner-validation RMSE only.
- PySR binary operators: +, -, *.
- No unary operators.
- PySR maximum expression size: 15.
- PySR seed: 42.
- PySR deterministic serial execution.
- Reaction samples: 500.
- PySR fitting domain derived from outer-training trajectories only.

## Leakage Firewall

For each outer fold:

1. Hold out every visit from the outer-test participants during fitting.
2. Divide outer-training participants into fitting and inner-validation groups
   deterministically.
3. Train and select models using fitting and inner-validation participants only.
4. Select neural seed using inner-validation RMSE only.
5. Construct the PySR fitting domain and samples using outer-training data only.
6. Freeze the fold model.
7. Forecast the final visit for outer-test participants using only preceding
   observations, elapsed times, and static structural connectivity.
8. Save private predictions and residuals for reproducibility.
9. Do not expose fold outcomes until every prespecified model in every fold has
   completed.

Training participants may contribute all consecutive intervals. Outer-test
participants may contribute no outcome values to fitting or model selection.

## Analysis-Freeze Artifact

The private analysis-freeze artifact must include:

- Protocol hash.
- Git commit.
- Model settings.
- Split assignments in private neutral form.
- Fold-generation seed.
- Metric definitions.
- Bootstrap settings.
- Software versions.

The runner must refuse to start if the protocol and JSON disagree, the protocol
hash is missing or changed, input QC fails, private output permissions are
unsafe, the output directory is inside the repository, completed results would
be overwritten, or required participant/time mappings are ambiguous.

## Privacy

Future manifests, raw files, predictions, residuals, checkpoints, and private
freeze artifacts are sensitive medical/research data. They must remain outside
the repository with restrictive permissions. Console output and tracked
documentation must use aggregate counts and neutral labels only.

## Phase Boundary

This phase creates and tests pre-data infrastructure using synthetic fixtures
only. No real-data run occurred during this phase, no old frozen cohort
outcomes were inspected for model tuning, and no SPIE submission files were
revised.
