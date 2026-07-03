#####################################################################################################
#  Compute standardisation statistics (mean/std) + min/max over the FULL Omega1..4 dataset.
#
#  Fixes vs the single-folder version:
#    1. Iterates the SAME file list the training script builds (all Omega, all subdomain prefixes),
#       so the stats match exactly what the model sees.
#    2. Correct slicing for u/v/w min/max (the y-axis was dropped in the old min/max code).
#    3. float64 accumulation. std = sqrt(E[x^2] - mean^2) subtracts two large numbers for pressure
#       (mean~1975, E[P^2]~3.9e6 -> result ~19); in float32 this cancels catastrophically. Casting
#       every per-file reduction to float64 keeps it accurate.
#
#  Slicing matches the dataset exactly:
#      X channel 0:  X[0, 1:-1, 1:-1]            (drop ghost cells)
#      Y channel c:  Y[c, 1:-1, :, 1:-1]
#####################################################################################################
import os
from pathlib import Path

import h5py
import numpy as np

# ---- Paths (keep in sync with the training script) ----
Case         = "clean"
samples_root = Path(f"...")
file_indices = list(range(280000, 43290001, 10000))
minsmax_file = Path("minsmaxs_bottom.h5")

# ---- Build the SAME file list as the trainer (all Omega, all subdomain prefixes) ----
file_paths = []
for omega in range(1, 5):
    omega_dir = samples_root / f"Omega{omega}"
    for i in file_indices:
        matches = sorted(omega_dir.glob(f"datapreproc_bottom_3d_turbulent_channel_flow_{i}-DATA.h5"))
        file_paths.extend(matches)
file_paths = [str(p) for p in sorted(file_paths)]
print(f"Number of samples for stats: {len(file_paths)}")

# ---- Accumulators (Python floats => float64) ----
# For each field: running sum, sum of squares, element count, min, max.
fields = ["P", "u", "v", "w"]
acc = {f: {"sum": 0.0, "sumsq": 0.0, "count": 0,
           "min": np.inf, "max": -np.inf} for f in fields}


def update(name, arr):
    a = np.asarray(arr, dtype=np.float64)          # force float64 BEFORE reducing
    acc[name]["sum"]   += a.sum()
    acc[name]["sumsq"] += np.square(a).sum()
    acc[name]["count"] += a.size
    amin = a.min(); amax = a.max()
    if amin < acc[name]["min"]:
        acc[name]["min"] = amin
    if amax > acc[name]["max"]:
        acc[name]["max"] = amax


for n_done, fp in enumerate(file_paths, 1):
    with h5py.File(fp, "r", swmr=True) as f:
        X = f["X_features"][...]        # (Cx, H+2, W+2)
        Y = f["Y_features"][...]        # (3,  H+2, Ny, W+2)

    update("P", X[0, 1:-1, 1:-1])
    update("u", Y[0, 1:-1, :, 1:-1])
    update("v", Y[1, 1:-1, :, 1:-1])
    update("w", Y[2, 1:-1, :, 1:-1])

    if n_done % 1000 == 0:
        print(f"  processed {n_done}/{len(file_paths)} files")


def mean_std(name):
    c = max(acc[name]["count"], 1)
    mean = acc[name]["sum"] / c
    var  = acc[name]["sumsq"] / c - mean ** 2
    var  = max(var, 0.0)                            # guard tiny negative from rounding
    return mean, np.sqrt(var)


stats = {}
for f in fields:
    m, s = mean_std(f)
    stats[f"{f}_mean"] = m
    stats[f"{f}_std"]  = s
    stats[f"{f}_min"]  = float(acc[f]["min"])
    stats[f"{f}_max"]  = float(acc[f]["max"])

print("\n--- Statistics over the full dataset ---")
for f in fields:
    print(f"{f}: mean={stats[f'{f}_mean']:.6g}  std={stats[f'{f}_std']:.6g}  "
          f"min={stats[f'{f}_min']:.6g}  max={stats[f'{f}_max']:.6g}")

# ---- Save with the same key names the dataset class expects ----
with h5py.File(minsmax_file, "w") as f:
    grp = f.create_group("minsmaxs")
    for k, v in stats.items():
        grp.create_dataset(k, data=float(v))
print(f"\nSaved: {minsmax_file}")
