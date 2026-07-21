"""Project-wide constants."""

EXPECTED_ROI_COUNT = 148
DATA_ROOT_ENV = "TAU_SR_DATA_ROOT"
UNSET = "UNSET"
SYNTHETIC_SUBJECT_PREFIX = "sub-syn"

NETWORK_REPRESENTATIONS = ("count", "fa", "length")
LAPLACIAN_TYPES = ("unnormalized", "symmetric_normalized")

FORBIDDEN_DATA_EXTENSIONS = {
    ".nii",
    ".nii.gz",
    ".dcm",
    ".dicom",
    ".npy",
    ".npz",
    ".parquet",
    ".pt",
    ".pth",
    ".ckpt",
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".7z",
    ".rar",
}
