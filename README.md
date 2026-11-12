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

# 2. Dataset (~1.7 GB in total) from Zenodo
mkdir -p Top_Quark_Tagging_dataset
for f in train val test; do
  curl -L -o Top_Quark_Tagging_dataset/$f.h5 https://zenodo.org/records/2603256/files/$f.h5
done

# 3. Launch
jupyter lab example_top_dataset.ipynb
```

`saved_model.pth` holds pre-trained weights for the Particle Flow Network. The notebook loads them if present, so training resumes from them instead of starting from scratch.

The notebook loads 500k training jets into memory (about 1.5 GB as a DataFrame), so a machine with 8 GB of RAM or more is recommended.

## Running on Google Colab

Coming soon.

## Repository layout

| File | Purpose |
|---|---|
| `example_top_dataset.ipynb` | Main tutorial notebook |
| `saved_model.pth` | Pre-trained Particle Flow Network weights |
| `requirements.txt` | Python dependencies |

## License

MIT, see [LICENSE](LICENSE).
