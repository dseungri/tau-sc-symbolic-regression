# Data Access and Security Notice

This repository is designed for restricted research data, but it must not contain real subject-level data.

Authorized users must store real data outside Git-controlled paths and point tooling to that location with `TAU_SR_DATA_ROOT`. Incoming files are immutable. Scripts in this repository are non-destructive by default and must never modify source data in place.

Never commit credentials, direct identifiers, protected health information, imaging archives, NIfTI files, DICOM files, matrices, Parquet files, generated subject outputs, model checkpoints, or local environment files.
