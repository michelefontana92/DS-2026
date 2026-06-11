# On the Fairness of Feature Attribution Explanations

This repository contains the code needed to reproduce the core experiments for
the paper: **On the Fairness of Feature Attribution Explanations.** It includes
FairLab black-box training, FastSHAP/LIME/Integrated Gradients/KernelSHAP
explanations, and explanation-quality metrics with sensitive-group gaps.

The public protocol uses a fixed sensitive-group-stratified **test subset**. The
original FairLab data loader may still expect files named `*_val.csv`; in this
repository that split is treated and reported as the test split.

## Repository Structure

```text
.
├── conimg/
│   └── icde_experiments.yaml
├── img/
│   ├── Compas_test_exploratory_summary_grid.png
│   ├── MEPS_test_exploratory_summary_grid.png
│   ├── Income_test_exploratory_summary_grid.png
│   ├── Income_3_test_exploratory_summary_grid.png
│   └── Education_test_exploratory_summary_grid.png
├── src/
│   ├── main.py                         # FairLab training CLI
│   ├── export_black_box_predictions.py # test logits/probabilities cache
│   ├── train_fastshap.py               # FastSHAP surrogate + explainer training
│   ├── create_test_subsets.py          # stratified fixed test subsets
│   ├── compute_explanations.py         # FastSHAP/LIME/IntGrad/KernelSHAP
│   ├── compute_xai_metrics.py          # quality means and gaps
│   ├── architectures/, dataloaders/, metrics/, runs/, wrappers/
│   └── fastshap/                       # tabular FastSHAP implementation
├── pyproject.toml
└── README.md
```

Plotting scripts, logs, generated results, checkpoints, and exploratory
mitigation code are intentionally excluded.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

FairLab uses Weights & Biases logging. For local offline runs:

```bash
export WANDB_MODE=offline
```

## Expected Data Layout

Place processed CSV files under a data root such as:

```text
data/
└── <experiment_name>/
    └── node_1/
        ├── <dataset>_train.csv
        └── <dataset>_val.csv   # used as the paper test split
```

You can also pass explicit `--train-set` and `--test-set` paths to the export
script.

## 1. Train FairLab Black Boxes

Example for Group-DP on Income:

```bash
fairlab-train \
  --run income_fairlab \
  --project_name Group-DP-Income \
  --root_dir data \
  --start_index 1 \
  -ml demographic_parity \
  -gl Race \
  -tl 0.10 \
  --num_classes 2
```

For equalized odds, use:

```bash
-ml equalized_odds
```

For multiclass datasets (`income_3_fairlab`, `education_fairlab`), pass
`--num_classes 3`.

## 2. Export Black-Box Test Predictions

Each trained black box must be exported into the experiment directory expected
by the explanation scripts:

```bash
fairxai-export-bb \
  --run income_fairlab \
  --checkpoint checkpoints/path/to/model.h5 \
  --output-dir results/Tau_010/Income/Race \
  --sensitive-attribute Race \
  --root-dir data \
  --experiment-name group_fairness \
  --start-index 1
```

This writes:

```text
black_box.h5
black_box_logits_and_data.npz
```

The cache includes `X_train`, test `X`, black-box logits/probabilities,
black-box predicted labels, true labels, sensitive-group IDs, feature names, and
categorical-feature masks.

## 3. Train FastSHAP

FastSHAP is the only explainer here that requires training.

```bash
fairxai-train-fastshap \
  --experiment-dir results/Tau_010/Income/Race \
  --device cpu
```

Default FastSHAP settings:

```text
surrogate: mlp2hidden_surrogate, hidden=(512, 128), epochs=200, lr=1e-4
explainer: mlp2hidden_explainer, hidden=(256, 128), epochs=300, lr=1e-4
coalition samples per explainer batch: 128
```

The script writes:

```text
surrogate_model.h5
explainer_model.h5
fastshap_explanations.npz
```

## 4. Create Fixed Test Subsets

The main experimental protocol uses a fixed test subset, stratified by the
sensitive groups of each dataset/attribute pair:

```bash
fairxai-create-test-subsets \
  --root results/New_Experiments \
  --n-records 2000 \
  --min-per-group 20 \
  --seed 42
```

Subsets are saved under:

```text
results/New_Experiments/test_subsets/
```

## 5. Compute Explainer Attributions

```bash
fairxai-compute-explanations \
  --root results/New_Experiments \
  --n-records 2000 \
  --min-per-group 20 \
  --seed 42 \
  --explainers fastshap intgrad lime kernelshap
```

Default explainer settings:

```text
FastSHAP: uses saved fastshap_explanations.npz and subsets it.
IntGrad: shap.GradientExplainer, background size 256, batch size 64.
LIME: LimeTabularExplainer, categorical features from cat_cols_mask,
      num_samples=1000, discretize_continuous=True.
KernelSHAP: shap.KernelExplainer, kmeans background size 100,
            nsamples=200, batch size 16.
```

Attribution files are written beside each experiment:

```text
test_subset_n2000_min20_seed42_<explainer>_explanations.npz
```

## 6. Compute Explanation Quality and Gaps

```bash
fairxai-compute-metrics \
  --root results/New_Experiments \
  --n-records 2000 \
  --min-per-group 20 \
  --seed 42 \
  --explainers fastshap intgrad lime kernelshap
```

The output CSV reports:

```text
quality_mean, quality_std, quality_n_eval, gap
```

The `gap` column is the paper gap:

1. Compute per-record metric scores using a within-group reference.
2. Split each sensitive group by the black-box predicted class.
3. For each predicted class, compute the maximum pairwise group difference.
4. Average these class-wise gaps.

Metric defaults:

```text
Sufficiency: Quantus Sufficiency, threshold=0.6, distance_func=seuclidean,
             normalise=True, abs=False.
Consistency: Quantus Consistency, top_n_sign with n=5, normalise=True.
Stability: rank agreement in cosine-similarity neighborhoods, k=5 and k=10,
           signed feature ranking, standardized input features.
```

Outputs are stored under:

```text
results/New_Experiments/xai_metrics/test_subset_n2000_min20_seed42/
```

## Supported Runs

```text
compas_fairlab    -> Compas, binary
mep_fairlab       -> MEPS, binary
income_fairlab    -> Income, binary
income_3_fairlab  -> Income_3, multiclass
education_fairlab -> Education, multiclass
```

## Data Description
Here, we report a descriptive analysis of the datasets used in our experiments. For each dataset, we summarize: (A) the target class distribution; (B) the smallest and largest sensitive groups, highlighting group-size imbalance; (C) the class composition within these extreme groups; (D) the most label-informative features, measured through normalized mutual information with the target variable; (E) the cumulative concentration of label information across ranked features; and (F) the strongest pairwise feature associations, measured through normalized mutual information. Together, these diagnostics provide insights into class imbalance, sensitive-group heterogeneity, feature relevance, and feature dependence structures that may influence both predictive performance and explanation fairness.
### Compas
The dataset contains 6,172 criminal records from Broward County, Florida, with 34 features describing demographics, criminal history, and incarceration details. The goal is to predict whether a defendant will re-offend within two years.  
<p align="center">
  <img src="img/Compas_test_exploratory_summary_grid.png" width="900">
</p>

### MEPS
Derived from the 2015 Medical Expenditure Panel Survey, it includes about 30,000 records and 132 features after preprocessing. The task is to predict whether an individual’s annual medical expenditures exceed the third quartile.  
<p align="center">
  <img src="img/MEPS_test_exploratory_summary_grid.png" width="900">
</p>

### Income
Drawn from the Folktables suite, based on the 2014 U.S. Census, it contains approximately 2.45 million records and 20 features after preprocessing. The task is to predict whether an individual earns more than \$50,000 annually.  
<p align="center">
  <img src="img/Income_test_exploratory_summary_grid.png" width="900">
</p>

### Income_3
This dataset extends \income\ to multiclass classification, using the same 20 features and 2.45 million records but with three income brackets as labels: below \$30,000, between \$30,000 and \$50,000, and above \$50,000. 
<p align="center">
  <img src="img/Income_3_test_exploratory_summary_grid.png" width="900">
</p>

### Education
Derived from Folktables, it contains about 2.45 million records and 25 features after preprocessing. The task is multiclass classification: predicting education level with three classes (less than high school, high school, and college or above).

<p align="center">
  <img src="img/Education_test_exploratory_summary_grid.png" width="900">
</p>

## Black-box Predictive Model

Across all datasets, the predictive model is implemented as a feedforward neural network with two hidden layers containing $300$ and $100$ neurons, respectively, and ReLU activation functions. A Dropout layer with rate $0.2$ is applied after the last hidden layer for regularization, while the output layer uses the Softmax function.
