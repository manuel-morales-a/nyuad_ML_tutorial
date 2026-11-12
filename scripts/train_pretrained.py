"""Train the network that the tutorial loads by default (models/pfn_pretrained.pth).

Runs example_top_dataset.ipynb with TRAIN_FROM_SCRATCH = True, so the preprocessing and
training are exactly those of the notebook, then copies the trained weights and their
loss history into models/. Run it from the repository root, with the dataset in
Top_Quark_Tagging_dataset/ (the notebook downloads it if missing):

    python scripts/train_pretrained.py

Needs jupyter (nbconvert) in the environment. A GPU (CUDA or Apple Silicon) is recommended.
"""
import json
import os
import shutil
import subprocess
import sys

NOTEBOOK = "example_top_dataset.ipynb"
EXECUTED = "train_pretrained_executed.ipynb"

nb = json.load(open(NOTEBOOK))
flag = "TRAIN_FROM_SCRATCH = False"
cells = [c for c in nb["cells"] if flag in "".join(c["source"])]
assert len(cells) == 1, f"expected exactly one cell with '{flag}'"
cells[0]["source"] = "".join(cells[0]["source"]).replace(flag, "TRAIN_FROM_SCRATCH = True")
with open(EXECUTED, "w") as f:
    json.dump(nb, f)

subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                "--ExecutePreprocessor.timeout=-1", EXECUTED], check=True)

import torch  # noqa: E402

os.makedirs("models", exist_ok=True)
# store CPU tensors, so the weights load on any device (CUDA, Apple Silicon or CPU)
torch.save(torch.load("pfn_trained.pth", map_location="cpu"), "models/pfn_pretrained.pth")
shutil.copy("pfn_trained_history.json", "models/pfn_history.json")
history = json.load(open("models/pfn_history.json"))
print(f"Trained for {len(history['val_loss'])} epochs, best validation loss {min(history['val_loss']):.4f}")
print(f"Executed notebook with all outputs: {EXECUTED}")
