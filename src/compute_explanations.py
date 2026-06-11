import argparse
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fairxai_mpl")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import numpy as np
import torch

from experiment_utils import (
    build_target_attributions,
    discover_experiments,
    explanation_filename,
    load_black_box,
    load_subset_indices,
    normalise_shap_output,
    predict_logits_and_proba,
    subset_tag as make_subset_tag,
)


DEFAULT_EXPLAINERS = ["fastshap", "intgrad", "lime", "kernelshap"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute FastSHAP, Integrated Gradients, LIME, and KernelSHAP "
            "explanations on fixed sensitive-group-stratified test subsets."
        )
    )
    parser.add_argument("--root", default="results/New_Experiments")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root where explanation files will be written. Defaults to --root.",
    )
    parser.add_argument(
        "--subset-root",
        default=None,
        help="Root containing test_subsets. Defaults to --root.",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--attributes", nargs="*", default=None)
    parser.add_argument("--taus", nargs="*", default=None)
    parser.add_argument(
        "--explainers",
        nargs="*",
        default=DEFAULT_EXPLAINERS,
        choices=tuple(DEFAULT_EXPLAINERS),
    )
    parser.add_argument("--n-records", type=int, default=2000)
    parser.add_argument("--min-per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true")

    parser.add_argument(
        "--intgrad-background-size",
        type=int,
        default=256,
        help="Background size for shap.GradientExplainer.",
    )
    parser.add_argument("--intgrad-batch-size", type=int, default=64)

    parser.add_argument(
        "--kernelshap-background-size",
        type=int,
        default=100,
        help="Background size for shap.KernelExplainer.",
    )
    parser.add_argument(
        "--kernelshap-background-mode",
        choices=("sample", "kmeans"),
        default="kmeans",
        help="Use sampled background rows or shap.kmeans summarisation.",
    )
    parser.add_argument(
        "--kernelshap-nsamples",
        default="200",
        help="nsamples passed to KernelExplainer.shap_values; use an int or 'auto'.",
    )
    parser.add_argument("--kernelshap-batch-size", type=int, default=16)

    parser.add_argument("--lime-num-samples", type=int, default=1000)
    parser.add_argument("--lime-batch-size", type=int, default=1024)
    parser.add_argument(
        "--lime-discretize-continuous",
        dest="lime_discretize_continuous",
        action="store_true",
        help="Enable LIME continuous-feature discretisation.",
    )
    parser.add_argument(
        "--lime-no-discretize-continuous",
        dest="lime_discretize_continuous",
        action="store_false",
        help="Disable LIME continuous-feature discretisation.",
    )
    parser.add_argument(
        "--lime-no-categorical-features",
        dest="lime_use_categorical_features",
        action="store_false",
        help="Do not pass categorical columns from cat_cols_mask to LIME.",
    )
    parser.set_defaults(
        lime_discretize_continuous=True,
        lime_use_categorical_features=True,
    )
    return parser.parse_args()


def parse_kernelshap_nsamples(value):
    if str(value).lower() == "auto":
        return "auto"
    return int(value)


def subset_tag(args):
    return make_subset_tag(args.n_records, args.min_per_group, args.seed)


def artifact_exp_dir(exp, args):
    if args.output_root:
        return Path(args.output_root) / exp["tau_dir"] / exp["dataset"] / exp["attribute"]
    return Path(exp["path"])


def output_path(exp, explainer, args):
    return artifact_exp_dir(exp, args) / explanation_filename(
        explainer,
        args.n_records,
        args.min_per_group,
        args.seed,
    )


def feature_names_from_files(exp_dir, n_features):
    fastshap_path = Path(exp_dir) / "fastshap_explanations.npz"
    if fastshap_path.exists():
        data = np.load(fastshap_path, allow_pickle=True)
        if "feature_names" in data:
            return np.asarray(data["feature_names"], dtype=object)
    return np.asarray([f"feature_{i}" for i in range(n_features)], dtype=object)


def categorical_features_from_info(info, n_features):
    if "cat_cols_mask" not in info:
        return []
    mask = np.asarray(info["cat_cols_mask"]).reshape(-1).astype(bool)
    if mask.size != n_features:
        raise ValueError(f"cat_cols_mask has length {mask.size}, expected {n_features}.")
    return np.flatnonzero(mask).astype(int).tolist()


def build_lime_categorical_maps(X_train, X_test, categorical_features):
    maps = {}
    for feature in categorical_features:
        values = np.unique(
            np.concatenate(
                [
                    np.asarray(X_train[:, feature], dtype=np.float32),
                    np.asarray(X_test[:, feature], dtype=np.float32),
                ]
            )
        ).astype(np.float32)
        maps[int(feature)] = values
    return maps


def encode_lime_categoricals(X_np, categorical_maps):
    encoded = np.asarray(X_np, dtype=np.float32).copy()
    for feature, values in categorical_maps.items():
        codes = np.searchsorted(values, encoded[:, feature])
        valid = (
            (codes >= 0)
            & (codes < values.size)
            & np.isclose(values[np.clip(codes, 0, values.size - 1)], encoded[:, feature])
        )
        if not np.all(valid):
            nearest = np.argmin(
                np.abs(encoded[:, [feature]] - values.reshape(1, -1)),
                axis=1,
            )
            codes = np.where(valid, codes, nearest)
        encoded[:, feature] = codes.astype(np.float32)
    return encoded


def decode_lime_categoricals(X_np, categorical_maps):
    decoded = np.asarray(X_np, dtype=np.float32).copy()
    if decoded.ndim == 1:
        decoded_2d = decoded.reshape(1, -1)
        squeeze = True
    else:
        decoded_2d = decoded
        squeeze = False
    for feature, values in categorical_maps.items():
        codes = np.rint(decoded_2d[:, feature]).astype(np.int64)
        codes = np.clip(codes, 0, values.size - 1)
        decoded_2d[:, feature] = values[codes]
    return decoded_2d.reshape(-1) if squeeze else decoded_2d


def save_explanations(
    *,
    out_path,
    shap_values,
    indices,
    y_pred,
    y_true,
    groups,
    feature_names,
    explainer,
    method,
    meta,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        shap_values=np.asarray(shap_values, dtype=np.float32),
        target_attributions=build_target_attributions(shap_values, y_pred),
        indices=np.asarray(indices, dtype=np.int64),
        y=np.asarray(y_pred, dtype=np.int64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
        y_true=np.asarray(y_true) if y_true is not None else np.asarray([], dtype=np.int64),
        groups=np.asarray(groups),
        feature_names=np.asarray(feature_names, dtype=object),
        explainer=str(explainer),
        method=str(method),
        meta=np.asarray([meta], dtype=object),
    )


def compute_fastshap(exp, indices, args):
    exp_dir = Path(exp["path"])
    out = output_path(exp, "fastshap", args)
    if out.exists() and not args.overwrite:
        return "skipped", out

    info = np.load(exp_dir / "black_box_logits_and_data.npz", allow_pickle=True)
    saved = np.load(exp_dir / "fastshap_explanations.npz", allow_pickle=True)
    shap_values = np.asarray(saved["shap_values"], dtype=np.float32)[indices]
    y_pred = np.asarray(info["y_pred"], dtype=np.int64).reshape(-1)[indices]
    y_true = np.asarray(info["y_true"])[indices] if "y_true" in info else None
    groups = np.asarray(info["groups"]).reshape(-1)[indices]
    feature_names = feature_names_from_files(exp_dir, shap_values.shape[1])

    save_explanations(
        out_path=out,
        shap_values=shap_values,
        indices=indices,
        y_pred=y_pred,
        y_true=y_true,
        groups=groups,
        feature_names=feature_names,
        explainer="fastshap",
        method="saved_fastshap_test_subset",
        meta={
            "source_file": str(exp_dir / "fastshap_explanations.npz"),
            "subset_tag": subset_tag(args),
            "dataset": exp["dataset"],
            "attribute": exp["attribute"],
            "tau": float(exp["tau"]),
            "tau_dir": exp["tau_dir"],
        },
    )
    return "computed", out


def compute_intgrad(exp, indices, args):
    import shap

    exp_dir = Path(exp["path"])
    out = output_path(exp, "intgrad", args)
    if out.exists() and not args.overwrite:
        return "skipped", out

    rng = np.random.default_rng(args.seed)
    info = np.load(exp_dir / "black_box_logits_and_data.npz", allow_pickle=True)
    X_test = np.asarray(info["X"], dtype=np.float32)
    X_train = np.asarray(info["X_train"], dtype=np.float32)
    y_pred = np.asarray(info["y_pred"], dtype=np.int64).reshape(-1)
    y_true = np.asarray(info["y_true"])[indices] if "y_true" in info else None
    groups = np.asarray(info["groups"]).reshape(-1)

    model = load_black_box(exp_dir, args.device)
    n_bg = min(int(args.intgrad_background_size), X_train.shape[0])
    background_indices = np.sort(
        rng.choice(np.arange(X_train.shape[0]), size=n_bg, replace=False)
    )
    background = torch.as_tensor(
        X_train[background_indices],
        dtype=torch.float32,
        device=args.device,
    )
    explainer = shap.GradientExplainer(model, background)

    X_subset = X_test[indices]
    with torch.no_grad():
        logits0 = model(torch.as_tensor(X_subset[:1], dtype=torch.float32, device=args.device))
    n_classes = int(logits0.shape[1])
    n_features = int(X_subset.shape[1])
    shap_values = np.zeros((X_subset.shape[0], n_features, n_classes), dtype=np.float32)

    t0 = time.time()
    for start in range(0, X_subset.shape[0], int(args.intgrad_batch_size)):
        end = min(start + int(args.intgrad_batch_size), X_subset.shape[0])
        xb = torch.as_tensor(X_subset[start:end], dtype=torch.float32, device=args.device)
        sv = explainer.shap_values(xb)
        shap_values[start:end] = normalise_shap_output(
            sv,
            batch_size=end - start,
            n_features=n_features,
            n_classes=n_classes,
        )
        print(
            f"  intgrad {exp['tau_dir']}/{exp['dataset']}/{exp['attribute']} "
            f"{end}/{X_subset.shape[0]} elapsed={time.time() - t0:.1f}s"
        )

    feature_names = feature_names_from_files(exp_dir, n_features)
    save_explanations(
        out_path=out,
        shap_values=shap_values,
        indices=indices,
        y_pred=y_pred[indices],
        y_true=y_true,
        groups=groups[indices],
        feature_names=feature_names,
        explainer="intgrad",
        method="shap_gradient_explainer_expected_gradients",
        meta={
            "subset_tag": subset_tag(args),
            "dataset": exp["dataset"],
            "attribute": exp["attribute"],
            "tau": float(exp["tau"]),
            "tau_dir": exp["tau_dir"],
            "background_size": int(n_bg),
            "background_indices": background_indices,
            "batch_size": int(args.intgrad_batch_size),
            "device": str(args.device),
            "public_implementation": "shap.GradientExplainer",
        },
    )
    return "computed", out


def compute_kernelshap(exp, indices, args):
    import shap

    exp_dir = Path(exp["path"])
    out = output_path(exp, "kernelshap", args)
    if out.exists() and not args.overwrite:
        return "skipped", out

    rng = np.random.default_rng(args.seed)
    info = np.load(exp_dir / "black_box_logits_and_data.npz", allow_pickle=True)
    X_test = np.asarray(info["X"], dtype=np.float32)
    X_train = np.asarray(info["X_train"], dtype=np.float32)
    y_pred = np.asarray(info["y_pred"], dtype=np.int64).reshape(-1)
    y_true = np.asarray(info["y_true"])[indices] if "y_true" in info else None
    groups = np.asarray(info["groups"]).reshape(-1)
    model = load_black_box(exp_dir, args.device)

    n_bg = min(int(args.kernelshap_background_size), X_train.shape[0])
    if args.kernelshap_background_mode == "kmeans":
        background_indices = np.asarray([], dtype=np.int64)
        background = shap.kmeans(X_train, n_bg)
    else:
        background_indices = np.sort(
            rng.choice(np.arange(X_train.shape[0]), size=n_bg, replace=False)
        )
        background = X_train[background_indices]

    def predict_fn(batch):
        logits, _ = predict_logits_and_proba(
            model=model,
            X_np=np.asarray(batch, dtype=np.float32),
            device=args.device,
            batch_size=max(int(args.kernelshap_batch_size), 1),
        )
        return logits

    X_subset = X_test[indices]
    logits0 = predict_fn(X_subset[:1])
    n_classes = int(logits0.shape[1])
    n_features = int(X_subset.shape[1])
    shap_values = np.zeros((X_subset.shape[0], n_features, n_classes), dtype=np.float32)
    nsamples = parse_kernelshap_nsamples(args.kernelshap_nsamples)
    feature_names = feature_names_from_files(exp_dir, n_features)

    explainer = shap.KernelExplainer(
        predict_fn,
        background,
        feature_names=[str(name) for name in feature_names],
    )

    t0 = time.time()
    for start in range(0, X_subset.shape[0], int(args.kernelshap_batch_size)):
        end = min(start + int(args.kernelshap_batch_size), X_subset.shape[0])
        with warnings.catch_warnings():
            try:
                from sklearn.exceptions import ConvergenceWarning

                warnings.simplefilter("ignore", ConvergenceWarning)
            except Exception:
                pass
            sv = explainer.shap_values(
                X_subset[start:end],
                nsamples=nsamples,
                silent=True,
            )
        shap_values[start:end] = normalise_shap_output(
            sv,
            batch_size=end - start,
            n_features=n_features,
            n_classes=n_classes,
        )
        print(
            f"  kernelshap {exp['tau_dir']}/{exp['dataset']}/{exp['attribute']} "
            f"{end}/{X_subset.shape[0]} elapsed={time.time() - t0:.1f}s"
        )

    save_explanations(
        out_path=out,
        shap_values=shap_values,
        indices=indices,
        y_pred=y_pred[indices],
        y_true=y_true,
        groups=groups[indices],
        feature_names=feature_names,
        explainer="kernelshap",
        method="shap_kernel_explainer",
        meta={
            "subset_tag": subset_tag(args),
            "dataset": exp["dataset"],
            "attribute": exp["attribute"],
            "tau": float(exp["tau"]),
            "tau_dir": exp["tau_dir"],
            "background_size": int(n_bg),
            "background_mode": str(args.kernelshap_background_mode),
            "background_indices": background_indices,
            "batch_size": int(args.kernelshap_batch_size),
            "nsamples": nsamples,
            "device": str(args.device),
            "public_implementation": "shap.KernelExplainer",
            "model_output": "logits",
        },
    )
    return "computed", out


def compute_lime(exp, indices, args):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'lime' package is not installed. Install it before running LIME."
        ) from exc

    exp_dir = Path(exp["path"])
    out = output_path(exp, "lime", args)
    if out.exists() and not args.overwrite:
        return "skipped", out

    info = np.load(exp_dir / "black_box_logits_and_data.npz", allow_pickle=True)
    X_test = np.asarray(info["X"], dtype=np.float32)
    X_train = np.asarray(info["X_train"], dtype=np.float32)
    y_pred = np.asarray(info["y_pred"], dtype=np.int64).reshape(-1)
    y_true = np.asarray(info["y_true"])[indices] if "y_true" in info else None
    groups = np.asarray(info["groups"]).reshape(-1)
    model = load_black_box(exp_dir, args.device)

    logits, _ = predict_logits_and_proba(
        model=model,
        X_np=X_test,
        device=args.device,
        batch_size=args.lime_batch_size,
    )
    n_classes = int(logits.shape[1])
    n_features = int(X_test.shape[1])
    feature_names = feature_names_from_files(exp_dir, n_features)
    categorical_features = (
        categorical_features_from_info(info, n_features)
        if args.lime_use_categorical_features
        else []
    )
    categorical_maps = build_lime_categorical_maps(
        X_train,
        X_test,
        categorical_features,
    )
    X_train_lime = encode_lime_categoricals(X_train, categorical_maps)
    X_lime = encode_lime_categoricals(X_test, categorical_maps)
    categorical_names = {
        feature: [str(value) for value in values]
        for feature, values in categorical_maps.items()
    }

    lime_explainer = LimeTabularExplainer(
        training_data=X_train_lime,
        feature_names=[str(v) for v in feature_names],
        class_names=[str(i) for i in range(n_classes)],
        categorical_features=categorical_features,
        categorical_names=categorical_names,
        mode="classification",
        discretize_continuous=bool(args.lime_discretize_continuous),
        random_state=int(args.seed),
    )
    print(
        "  lime config: "
        f"categorical_features={len(categorical_features)}, "
        f"discretize_continuous={bool(args.lime_discretize_continuous)}, "
        f"num_samples={int(args.lime_num_samples)}"
    )

    def predict_fn(batch):
        decoded_batch = decode_lime_categoricals(batch, categorical_maps)
        _, proba = predict_logits_and_proba(
            model=model,
            X_np=np.asarray(decoded_batch, dtype=np.float32),
            device=args.device,
            batch_size=args.lime_batch_size,
        )
        return proba

    shap_values = np.zeros((indices.size, n_features, n_classes), dtype=np.float32)
    t0 = time.time()
    for pos, idx in enumerate(indices):
        target = int(y_pred[idx])
        explanation = lime_explainer.explain_instance(
            X_lime[idx],
            predict_fn,
            labels=[target],
            num_features=n_features,
            num_samples=int(args.lime_num_samples),
        )
        values = np.zeros(n_features, dtype=np.float32)
        for feature_idx, weight in explanation.as_map().get(target, []):
            values[int(feature_idx)] = float(weight)
        shap_values[pos, :, target] = values
        if (pos + 1) % 25 == 0 or pos + 1 == indices.size:
            print(
                f"  lime {exp['tau_dir']}/{exp['dataset']}/{exp['attribute']} "
                f"{pos + 1}/{indices.size} elapsed={time.time() - t0:.1f}s"
            )

    save_explanations(
        out_path=out,
        shap_values=shap_values,
        indices=indices,
        y_pred=y_pred[indices],
        y_true=y_true,
        groups=groups[indices],
        feature_names=feature_names,
        explainer="lime",
        method="lime_tabular_categorical_mask_encoded",
        meta={
            "subset_tag": subset_tag(args),
            "dataset": exp["dataset"],
            "attribute": exp["attribute"],
            "tau": float(exp["tau"]),
            "tau_dir": exp["tau_dir"],
            "lime_num_samples": int(args.lime_num_samples),
            "lime_discretize_continuous": bool(args.lime_discretize_continuous),
            "lime_use_categorical_features": bool(args.lime_use_categorical_features),
            "lime_categorical_features": np.asarray(categorical_features, dtype=np.int64),
            "lime_categorical_value_counts": {
                int(feature): int(values.size)
                for feature, values in categorical_maps.items()
            },
            "lime_categorical_original_values": categorical_maps,
            "lime_categorical_encoding": "sorted_original_values_to_integer_codes",
            "device": str(args.device),
        },
    )
    return "computed", out


def main():
    args = parse_args()
    subset_root = args.subset_root or args.root
    experiments = discover_experiments(
        root=args.root,
        datasets=args.datasets,
        attributes=args.attributes,
        taus=args.taus,
    )
    print(f"Discovered {len(experiments)} experiments.")
    for exp in experiments:
        try:
            indices, subset_file = load_subset_indices(
                subset_root,
                exp["dataset"],
                exp["attribute"],
                args.n_records,
                args.min_per_group,
                args.seed,
            )
        except FileNotFoundError as exc:
            if args.list_only:
                print(
                    f"{exp['tau_dir']}/{exp['dataset']}/{exp['attribute']}: "
                    f"missing subset ({exc.filename})"
                )
                continue
            raise
        print(
            f"{exp['tau_dir']}/{exp['dataset']}/{exp['attribute']}: "
            f"subset_n={indices.size}, subset={subset_file}"
        )
        if args.list_only:
            continue
        for explainer in args.explainers:
            print(f"Computing {explainer}...")
            if explainer == "fastshap":
                status, path = compute_fastshap(exp, indices, args)
            elif explainer == "intgrad":
                status, path = compute_intgrad(exp, indices, args)
            elif explainer == "lime":
                status, path = compute_lime(exp, indices, args)
            elif explainer == "kernelshap":
                status, path = compute_kernelshap(exp, indices, args)
            else:
                raise ValueError(explainer)
            print(f"  {status}: {path}")


if __name__ == "__main__":
    main()
