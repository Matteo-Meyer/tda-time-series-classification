# Topological Features for Time-Series Analysis and Learning

An independent research project investigating when persistent-homology features of delay-coordinate embeddings provide useful information for time-series analysis and classification.

The central question is whether these representations add information beyond conventional statistical and Fourier-based features, and how their usefulness depends on noise, embedding parameters, and signal structure.

## Current progress

The project has progressed from exploratory persistent-homology experiments to the construction of an initial dataset for supervised learning.

Work completed so far includes:

- Persistent-homology experiments on synthetic point clouds.
- Delay-coordinate embeddings of periodic and noisy signals, with investigations of the effects of time delay and embedding dimension on the results.
- Controlled experiments comparing noisy periodic signals with shuffled versions of the same observations, to investigate how H1 features respond to changes in temporal structure.
- Analysis of geometric scale and diameter normalization of persistence lifetimes.
- Autocorrelation and false-nearest-neighbours diagnostics for exploring the delay parameter and embedding dimension, respectively.
- Implementation of a reusable five-feature representation derived from degree-one persistent homology.
- Generation and initial validation of a balanced dataset containing 100 matched pairs: 100 noisy periodic signals and their 100 shuffled counterparts.

The supervised classification benchmark is under development. Classifier performance and comparisons with conventional features have not yet been evaluated.

## Notebooks

The notebooks document the mathematical reasoning, experimental choices, code, and saved outputs.

| Notebook | Contents |
| --- | --- |
| [01 — Synthetic TDA](notebooks/01_synthetic_tda.ipynb) | Connected components and loops in synthetic point clouds; interpretation of persistence diagrams and lifetimes. |
| [02 — Time-series embeddings](notebooks/02_time_series_embeddings.ipynb) | Delay-coordinate geometry, noise and shuffling experiments, higher-dimensional embeddings, normalization, parameter choice, and persistence-feature construction. |
| [03 — Time-series classification](notebooks/03_time_series_classification.ipynb) | Construction and validation of the paired synthetic dataset; ongoing development of exploratory analysis and supervised evaluation. |

**Suggested starting point:** Notebook 02 provides the main investigation connecting time-series geometry with persistent homology.

## Persistence-feature representation

The shared implementation is in [src/tda_features.py](src/tda_features.py):

- `delay_embed`: constructs a delay-coordinate point cloud from a one-dimensional signal.
- `extract_H1_features`: computes degree-one Vietoris–Rips persistent homology and returns five summary features.

Persistence lifetimes are normalized by the diameter of the embedded point cloud. The resulting representation contains:

1. Largest normalized lifetime.
2. Second-largest normalized lifetime.
3. Total normalized persistence.
4. Number of normalized H1 lifetimes exceeding a chosen threshold.
5. Persistence entropy, computed from relative lifetime weights.

Diameter normalization removes dependence on uniform geometric scaling. Dependence on embedding shape, sampling, noise, and parameter choices remains part of the investigation.

## Initial classification experiment

Each independently generated noisy periodic signal is paired with a shuffled version of its observations. Shuffling preserves the empirical distribution of sample values while changing their temporal ordering.

The resulting feature table contains 200 rows, five persistence features, a class label, and a pair identifier.

Training and evaluation splits will keep both members of each pair together to avoid leakage between related signals.

## Next research steps

- Complete exploratory analysis of the synthetic paired dataset.
- Establish supervised evaluation with splits that respect pair membership.
- Compare persistence features with statistical, autocorrelation, and Fourier-based features.
- Evaluate whether combining topological and conventional features improves prediction.
- Investigate sensitivity to noise, embedding choices, and changes in the signal-generating process.
- Extend the study to more varied synthetic signals and, subsequently, suitable real-world datasets.

## Environment and reproducibility

The notebooks were developed using Python 3.12 and Jupyter. The repository includes a recorded dependency list in `requirements.txt`.

That dependency list reflects the Windows development environment.

Experimental parameters and saved outputs are documented in the notebooks. Shared embedding and feature-extraction functions are maintained in `src/tda_features.py`.