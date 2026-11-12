"""Make the half-size copy of the Top Quark Tagging dataset used by the tutorial.

Keeps the first half of each split of https://zenodo.org/records/2603256
(600k train, 200k val, 200k test jets) with the same HDF5 layout, so
pd.read_hdf(path, key="table") works exactly as with the original files.

Usage (e.g. in a Colab session that already has the full dataset):
    python make_half_dataset.py --src Top_Quark_Tagging_dataset --dst /content/drive/MyDrive/top_tagging_half
"""
import argparse
import os
import shutil
import tempfile

import pandas as pd

N_JETS = {"train": 600_000, "val": 200_000, "test": 200_000}
CHUNK = 100_000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="Top_Quark_Tagging_dataset", help="folder with the full train/val/test.h5")
    parser.add_argument("--dst", required=True, help="output folder (e.g. on a mounted Google Drive)")
    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:  # write locally first: HDF5 appends over a Drive mount are slow
        for name, n in N_JETS.items():
            out = os.path.join(tmp, f"{name}.h5")
            for start in range(0, n, CHUNK):
                df = pd.read_hdf(os.path.join(args.src, f"{name}.h5"), key="table", start=start, stop=min(start + CHUNK, n))
                df.to_hdf(out, key="table", format="table", append=True, complib="blosc:zstd", complevel=5)
            with pd.HDFStore(out, "r") as store:
                nrows = store.get_storer("table").nrows
            assert nrows == n, (name, nrows, n)
            shutil.copy(out, os.path.join(args.dst, f"{name}.h5"))
            print(f"{name}.h5: {nrows} jets, {os.path.getsize(out) / 1e6:.0f} MB -> {args.dst}")


if __name__ == "__main__":
    main()
