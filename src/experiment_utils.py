import csv
from pathlib import Path

import numpy as np
import torch

from architectures import ArchitectureFactory


TAU_ORDER = {
    "Tau_010": 0.10,
    "Tau_020": 0.20,
    "Tau_030": 0.30,
    "Tau_040": 0.40,
    "Tau_1": 1.00,
}

TAU_STRING = {
    0.10: "010",
    0.20: "020",
    0.30: "030",
    0.40: "040",
    1.00: "1",
}

DATASET_ORDER = ["Compas", "MEPS", "Income", "Income_3", "Education"]


def normalise_tau_token(token):
    token = str(token)
    if token.startswith("Tau_"):
        return token
    if token in {"010", "0.1", "0.10"}:
        return "Tau_010"
    if token in {"020", "0.2", "0.20"}:
        return "Tau_020"
    if token in {"030", "0.3", "0.30"}:
        return "Tau_030"
    if token in {"040", "0.4", "0.40"}:
        return "Tau_040"
    if token in {"1", "1.0", "1.00"}:
        return "Tau_1"
    raise ValueError(f"Unsupported tau filter: {token}")


def discover_experiments(root, datasets=None, attributes=None, taus=None):
    root = Path(root)
    datasets = set(datasets) if datasets else None
    attributes = set(attributes) if attributes else None
    taus = {normalise_tau_token(t) for t in taus} if taus else None

    experiments = []
    for tau_dir in sorted(root.glob("Tau_*"), key=lambda p: TAU_ORDER.get(p.name, 99)):
        if not tau_dir.is_dir():
            continue
        if taus is not None and tau_dir.name not in taus:
            continue
        for dataset_dir in sorted(tau_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            if datasets is not None and dataset_dir.name not in datasets:
                continue
            for attribute_dir in sorted(dataset_dir.iterdir()):
                if not attribute_dir.is_dir():
                    continue
                if attributes is not None and attribute_dir.name not in attributes:
                    continue
                required = (
                    attribute_dir / "black_box_logits_and_data.npz",
                    attribute_dir / "black_box.h5",
                )
                if all(path.exists() for path in required):
                    experiments.append(
                        {
                            "path": attribute_dir,
                            "tau_dir": tau_dir.name,
                            "tau": TAU_ORDER.get(tau_dir.name, np.nan),
                            "dataset": dataset_dir.name,
                            "attribute": attribute_dir.name,
                        }
                    )
    return experiments


def experiment_key(exp):
    return f"{exp['dataset']}_{exp['attribute']}"


def subset_dir(root):
    return Path(root) / "test_subsets"


def subset_tag(n, min_per_group, seed):
    return f"n{int(n)}_min{int(min_per_group)}_seed{int(seed)}"


def subset_filename(dataset, attribute, n, min_per_group, seed):
    return f"{dataset}_{attribute}_{subset_tag(n, min_per_group, seed)}.npz"


def explanation_filename(explainer, n, min_per_group, seed):
    return f"test_subset_{subset_tag(n, min_per_group, seed)}_{explainer}_explanations.npz"


def subset_path(root, dataset, attribute, n, min_per_group, seed):
    return subset_dir(root) / subset_filename(dataset, attribute, n, min_per_group, seed)


def load_subset_indices(root, dataset, attribute, n, min_per_group, seed):
    path = subset_path(root, dataset, attribute, n, min_per_group, seed)
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["indices"], dtype=np.int64), path


def make_model_from_state_dict(state_dict, device):
    model = ArchitectureFactory.create_architecture(
        "mlp2hidden",
        model_params={
            "input": state_dict["fc1.weight"].shape[1],
            "hidden1": state_dict["fc1.weight"].shape[0],
            "hidden2": state_dict["fc2.weight"].shape[0],
            "dropout": 0.2,
            "output": state_dict["out.weight"].shape[0],
        },
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_black_box(exp_dir, device):
    state_dict = torch.load(Path(exp_dir) / "black_box.h5", map_location=device)
    return make_model_from_state_dict(state_dict, device)


def predict_logits_and_proba(model, X_np, device, batch_size=1024):
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, X_np.shape[0], batch_size):
            xb = torch.as_tensor(X_np[start:start + batch_size], dtype=torch.float32, device=device)
            logits.append(model(xb).detach().cpu())
    logits = torch.cat(logits, dim=0)
    proba = torch.softmax(logits, dim=1).numpy()
    return logits.numpy(), proba


def normalise_shap_output(shap_values, batch_size, n_features, n_classes):
    if isinstance(shap_values, list):
        arr = np.stack([np.asarray(s, dtype=np.float32) for s in shap_values], axis=0)
        arr = np.transpose(arr, (1, 2, 0))
    else:
        arr = np.asarray(shap_values, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        elif arr.ndim != 3:
            raise ValueError(f"Unexpected attribution output shape: {arr.shape}")

    if arr.shape != (batch_size, n_features, n_classes):
        raise ValueError(
            f"Unexpected attribution shape {arr.shape}; expected "
            f"{(batch_size, n_features, n_classes)}"
        )
    return arr.astype(np.float32, copy=False)


def build_target_attributions(shap_values, y_pred):
    shap_values = np.asarray(shap_values, dtype=np.float32)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    if shap_values.ndim == 2:
        return shap_values
    if shap_values.ndim != 3:
        raise ValueError(f"Unsupported shap_values shape: {shap_values.shape}")
    return np.stack(
        [shap_values[i, :, int(y_pred[i])] for i in range(shap_values.shape[0])],
        axis=0,
    ).astype(np.float32)


def finite_summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(np.mean(finite)),
        float(np.std(finite)),
        float(np.median(finite)),
        float(np.percentile(finite, 75) - np.percentile(finite, 25)),
    )


def pairwise_abs_gap(group_rows):
    means = np.asarray([row["mean"] for row in group_rows], dtype=np.float64)
    if means.size < 2:
        return np.nan
    diffs = np.abs(means[:, None] - means[None, :])
    mask = ~np.eye(diffs.shape[0], dtype=bool)
    finite = np.isfinite(diffs) & mask
    if not finite.any():
        return np.nan
    return float(np.max(diffs[finite]))


def summarise_by_group(scores, groups, min_group_size):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    rows = []
    for group in np.unique(groups):
        mask = groups == group
        if int(mask.sum()) < int(min_group_size):
            continue
        group_scores = scores[mask]
        finite_n = int(np.isfinite(group_scores).sum())
        if finite_n == 0:
            continue
        mean, std, median, iqr = finite_summary(group_scores)
        rows.append(
            {
                "group": int(group)
                if np.issubdtype(np.asarray(group).dtype, np.integer)
                else group,
                "n_total": int(mask.sum()),
                "n_eval": finite_n,
                "mean": mean,
                "std": std,
                "median": median,
                "iqr": iqr,
            }
        )
    return rows, pairwise_abs_gap(rows)


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
