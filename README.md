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

# 2. Dataset (~1.7 GB in total) from Zenodo; the notebook also downloads it if missing
./download_data.sh

# 3. Launch
jupyter lab example_top_dataset.ipynb
```

`saved_model.pth` holds pre-trained weights for the Particle Flow Network. The notebook loads them if present, so training resumes from them instead of starting from scratch.

The notebook loads 500k training jets, which needs roughly 6-7 GB of RAM. Lower `N_TRAIN` / `N_VAL` in the notebook if you have less.

## Running on Google Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manuel-morales-a/nyuad_ML_tutorial/blob/main/example_top_dataset.ipynb)

Click the badge, then select a GPU under *Runtime → Change runtime type* and run the cells from top to bottom. The notebook installs what Colab is missing, downloads the dataset from Zenodo and fetches the pre-trained weights by itself.

## Repository layout

| File | Purpose |
|---|---|
| `example_top_dataset.ipynb` | Main tutorial notebook |
| `saved_model.pth` | Pre-trained Particle Flow Network weights |
| `requirements.txt` | Python dependencies (for local runs) |
| `download_data.sh` | Resumable dataset download (for local runs) |

## License

MIT, see [LICENSE](LICENSE).
