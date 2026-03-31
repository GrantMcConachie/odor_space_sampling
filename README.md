# odor_space_sampling

A toolkit for sampling lists of odorants that span odor space. Given a dataset of molecules (as SMILES strings), the package converts them into numerical descriptors, and provides several strategies for selecting a diverse, representative subset. It also includes tools for visualizing and evaluating how well each strategy covers the space.

The included dataset combines three sources: Goodscents, Leffingwell, and a human perceptual dataset.

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/GrantMcConachie/odor_space_sampling.git
cd odor_space_sampling
```

**2. Create and activate a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

**3. Install the package and its dependencies**
```bash
pip install -e .
```

---

## Quick start

The typical workflow is three lines:

```python
from odor_space_sampling import load_and_prepare, sample_with_all_methods, plot_all_sampling_method_points

# 1. load your data
data = load_and_prepare("data/gslf_and_human_data.csv")

# 2. sample odors using all methods
results = sample_with_all_methods(data, n_samples=20)

# 3. visualize where each method placed its samples
plot_all_sampling_method_points(data, results)
```

For a full walkthrough, open `noteboooks/example_usage.ipynb`.

---

## What's in the package

### Data loading — `data.py`

| Function | What it does |
|---|---|
| `load_and_prepare(filepath)` | Loads a CSV, computes RDKit descriptors, removes bad features, z-scores, and reduces with PCA. Returns an `OdorData` object. |
| `add_cid_to_data(filepath)` | Queries PubChem to add CID and IUPAC name columns to a CSV that only has SMILES. |

Your CSV must have a column named `smiles`. Everything else is optional.

The `OdorData` object returned by `load_and_prepare` has two fields:
- `data.df` — the original dataframe (smiles, label, cid, IUPAC, ...)
- `data.x` — the processed numeric matrix used for sampling; row `i` corresponds to `data.df.iloc[i]`

---

### Sampling — `sampling.py`

`sample_with_all_methods` runs all six strategies at once and returns a dictionary:

```python
results = sample_with_all_methods(data, n_samples=20, seed=12345)
# results["gmm"]["indices"]   → row indices into data.df
# results["gmm"]["samples"]   → the sampled points in feature space
# results["gmm"]["distances"] → distance from each sample to nearest real odor
```

The six strategies are:

| Method | Description |
|---|---|
| `uniform` | Samples uniformly at random within the bounding box of the data |
| `LHS` | Latin Hypercube Sampling — divides the space into a grid and samples one point per cell |
| `gaussian` | Samples from a multivariate Gaussian fit to the data |
| `min_max` | Greedy max-min diversity — each new point maximises its distance from all previously selected points |
| `kmeans` | Fits K-Means clusters and returns the data point nearest each cluster centre |
| `gmm` | Fits a Gaussian Mixture Model and returns the data points nearest its samples |

You can also run any method individually. Each returns `(samples, indices, distances)`:

```python
from odor_space_sampling.sampling import gmm_sample

result = gmm_sample(data.x, n_samples=20, seed=42)
samples, indices, distances = result
selected_odors = data.df.iloc[indices]
```

**GMM-specific tools**

| Function | Description |
|---|---|
| `get_n_closest_points_gmm(data, n_closest_points, ...)` | For each GMM sample, returns the `n` nearest real odors — useful for a ranked shortlist |
| `gmm_resample_varying_seeds(data, seeds, ...)` | Fits a fresh GMM for each seed and saves a separate CSV per seed |
| `aic_and_bic_gmm(data, max_n_clusters, ...)` | Sweeps over cluster counts and plots AIC, BIC, and KS stats to help you choose `n_clusters` |

---

### Plotting and evaluation — `plotting.py`

**Space exploration**

| Function | Description |
|---|---|
| `plot_scree_plot(data)` | Cumulative variance explained vs number of PCA components |
| `plot_feature_covariance(data)` | Histogram of pairwise feature correlations |
| `plot_fun_group_dist(data, ...)` | Normalized functional group frequency bar chart; pass a list of `(OdorData, label)` pairs to compare datasets side-by-side |

**Sample visualization**

| Function | Description |
|---|---|
| `plot_all_sampling_method_points(data, results)` | PCA + UMAP scatter plots for every method in `results` |
| `plot_sampling_projections(data, sample_methods)` | Same, but accepts any mix of sampling results and plain index arrays |

**Sample evaluation**

| Function | Description |
|---|---|
| `plot_all_sampling_methods_coverage(data, results)` | Bar chart of mean nearest-neighbour distance — lower = better coverage |
| `plot_all_sampling_methods_fun_groups(data, results)` | Bar chart of how many distinct functional groups each method captured; pass `save_path` to write missing groups to a text file |
| `plot_all_sampling_methods_data_dist(data, results)` | Grouped bar chart of human / gslf / both counts per method; pass `density=True` for fractions |
| `plot_ks_dist(reference, test_methods)` | KS statistic histograms comparing one or more sampled subsets against a reference dataset |

---

## Notebooks

| Notebook | Description |
|---|---|
| `noteboooks/example_usage.ipynb` | Full walkthrough of every most functions in the package |
---

## Data format

Your input CSV must have at minimum a `smiles` column containing valid SMILES strings. Optional columns that the package recognises:

| Column | Description |
|---|---|
| `smiles` | SMILES string **(required)** |
| `label` | Dataset origin, e.g. `['gslf']`, `['human']`, or `['human', 'gslf']` |
| `cid` | PubChem compound ID |
| `IUPAC` | IUPAC name |

---

## Project structure

```
odor_space_sampling/
├── data/                        example data files
├── noteboooks/
│   └── example_usage.ipynb      full function walkthrough
├── src/odor_space_sampling/
│   ├── __init__.py              public API
│   ├── data.py                  data loading and preparation
│   ├── sampling.py              sampling methods
│   ├── plotting.py              visualization and evaluation
│   └── utils.py                 internal helpers
├── requirements.txt
└── setup.py
```
