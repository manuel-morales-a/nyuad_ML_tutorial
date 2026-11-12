# Machine Learning for High Energy Physics — a hands-on tutorial

A practical introduction to machine learning for science, using jet tagging at the LHC as the running example.

The starting point is `example_top_dataset.ipynb`, adapted from [SimpleTopTagger](https://github.com/manuel-morales-a/SimpleTopTagger) (prepared for the MCNET24 summer school). It walks through:

1. **Data exploration**: the public [Top Quark Tagging Reference Dataset](https://zenodo.org/records/2603256), jet constituents, and physics-motivated observables (jet mass, multiplicity, ⟨p_T⟩).
2. **A baseline classifier**: logistic regression on two high-level features, with accuracy, ROC/AUC, confusion matrices and decision boundaries.
3. **A Deep Sets / Particle Flow Network** in PyTorch that acts directly on the jet constituents, plus a first look at its latent space.

## Running locally

```bash
# 1. Environment (any Python >= 3.10)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Launch; the notebook downloads the (quarter-size) dataset by itself
jupyter lab example_top_dataset.ipynb
```

To work with the full dataset from Zenodo (~1.7 GB) instead, run `./download_data.sh` before opening the notebook.

The notebook loads 300k training jets by default, which needs roughly 4 GB of RAM. Lower `N_TRAIN` / `N_VAL` in the notebook if you have less.

## Running on Google Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manuel-morales-a/nyuad_ML_tutorial/blob/main/example_top_dataset.ipynb)

Click the badge, then select a GPU under *Runtime → Change runtime type* and run the cells from top to bottom. The notebook installs what Colab is missing, downloads the dataset and fetches the pre-trained weights by itself, so a full run takes only a few minutes.

By default it downloads a quarter-size copy of the dataset (300k train / 100k val / 100k test jets, ~380 MB) from Google Drive, because Zenodo can be very slow. The copy is made with `scripts/make_reduced_dataset.py --fraction 0.25`; if the Drive download fails, the notebook falls back to the full dataset on Zenodo.

## Pre-trained network

Training the Particle Flow Network takes a while, so by default the notebook loads weights trained beforehand (`models/pfn_pretrained.pth`) together with their training history (`models/pfn_history.json`), which it uses to plot the learning curves. Set `TRAIN_FROM_SCRATCH = True` in the notebook to train the network yourself.

The pre-trained weights are produced by running the notebook itself with `TRAIN_FROM_SCRATCH = True`:

```bash
python scripts/train_pretrained.py
```

## Repository layout

| File | Purpose |
|---|---|
| `example_top_dataset.ipynb` | Main tutorial notebook |
| `models/` | Pre-trained Particle Flow Network weights and training history |
| `requirements.txt` | Python dependencies (for local runs) |
| `download_data.sh` | Resumable download of the full dataset from Zenodo (for local runs) |
| `scripts/make_reduced_dataset.py` | Builds the reduced (half / quarter) dataset copies hosted on Google Drive |
| `scripts/train_pretrained.py` | Re-trains the pre-trained network by running the notebook |

## License

MIT, see [LICENSE](LICENSE).
