"""Make a reduced copy of the Top Quark Tagging dataset (https://zenodo.org/records/2603256).

Keeps the first FRACTION of each split (full sizes: 1.2M train, 400k val, 400k test jets)
with the same HDF5 layout, so pd.read_hdf(path, key="table") works exactly as with the
original files. The jets are already shuffled, so the first rows are a fair sample, and a
smaller fraction is always a subset of a larger one.

Usage (e.g. in a Colab session that already has the full dataset):
    python make_reduced_dataset.py --fraction 0.5  --dst /content/drive/MyDrive/top_tagging_half
    python make_reduced_dataset.py --fraction 0.25 --dst /content/drive/MyDrive/top_tagging_quarter
"""
import argparse
import os
import shutil
import tempfile

import pandas as pd

CHUNK = 100_000


def n_rows(path):
    with pd.HDFStore(path, "r") as store:
        return store.get_storer("table").nrows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="Top_Quark_Tagging_dataset", help="folder with the full train/val/test.h5")
    parser.add_argument("--dst", required=True, help="output folder (e.g. on a mounted Google Drive)")
    parser.add_argument("--fraction", type=float, default=0.5, help="fraction of each split to keep (default: 0.5)")
    args = parser.parse_args()
    assert 0 < args.fraction <= 1, "--fraction must be in (0, 1]"
    os.makedirs(args.dst, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:  # write locally first: HDF5 appends over a Drive mount are slow
        for name in ["train", "val", "test"]:
            src = os.path.join(args.src, f"{name}.h5")
            n = int(round(n_rows(src) * args.fraction))
            out = os.path.join(tmp, f"{name}.h5")
            for start in range(0, n, CHUNK):
                df = pd.read_hdf(src, key="table", start=start, stop=min(start + CHUNK, n))
                df.to_hdf(out, key="table", format="table", append=True, complib="blosc:zstd", complevel=5)
            assert n_rows(out) == n, (name, n_rows(out), n)
            shutil.copy(out, os.path.join(args.dst, f"{name}.h5"))
            print(f"{name}.h5: {n} jets, {os.path.getsize(out) / 1e6:.0f} MB -> {args.dst}")


if __name__ == "__main__":
    main()
