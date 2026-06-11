import argparse
import os
from functools import partial
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fairxai_mpl")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import numpy as np
import quantus
from quantus.functions.discretise_func import top_n_sign

from experiment_utils import (
    discover_experiments,
    explanation_filename,
    finite_summary,
    load_black_box,
    subset_tag as make_subset_tag,
    write_csv,
)
from stability_utils import (
    compute_stability_scores,
    prepare_explanations,
    prepare_similarity_features,
)


DEFAULT_EXPLAINERS = ["fastshap", "intgrad", "lime", "kernelshap"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute explanation quality and sensitive-group gaps on fixed test "
            "subsets. The reported gap follows the paper protocol: within-group "
            "reference, local scores by black-box predicted class, and class-average "
            "aggregation."
        )
    )
    parser.add_argument("--root", default="results/New_Experiments")
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Root containing explanation files. Defaults to --root.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for metric outputs. Defaults to <root>/xai_metrics.",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--attributes", nargs="*", default=None)
    parser.add_argument("--taus", nargs="*", default=None)
    parser.add_argument("--explainers", nargs="*", default=DEFAULT_EXPLAINERS)
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["sufficiency", "consistency", "stability"],
        choices=("sufficiency", "consistency", "stability"),
    )
    parser.add_argument("--n-records", type=int, default=2000)
    parser.add_argument("--min-per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-group-size", type=int, default=1)
    parser.add_argument("--min-class-size", type=int, default=2)
    parser.add_argument("--min-group-class-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true")

    parser.add_argument("--sufficiency-threshold", type=float, default=0.6)
    parser.add_argument("--sufficiency-distance-func", default="seuclidean")
    parser.add_argument("--sufficiency-batch-size", type=int, default=256)
    parser.add_argument("--sufficiency-abs", action="store_true")
    parser.add_argument(
        "--sufficiency-no-normalise",
        dest="sufficiency_normalise",
        action="store_false",
    )
    parser.set_defaults(sufficiency_normalise=True)

    parser.add_argument("--consistency-discretise-n", type=int, default=5)
    parser.add_argument("--consistency-abs", action="store_true")

    parser.add_argument("--stability-k-values", nargs="*", type=int, default=[5, 10])
    parser.add_argument(
        "--stability-distance",
        choices=("rank_agreement", "cosine_similarity", "mse", "rmse", "euclidean"),
        default="rank_agreement",
    )
    parser.add_argument(
        "--stability-rank-mode",
        choices=("signed", "absolute"),
        default="signed",
    )
    parser.add_argument(
        "--stability-feature-scaling",
        choices=("standardize", "none"),
        default="standardize",
    )
    parser.add_argument("--stability-normalise-explanations", action="store_true")
    parser.add_argument("--stability-neighbour-batch-size", type=int, default=4096)
    parser.add_argument("--stability-n-jobs", type=int, default=1)
    parser.add_argument("--summary-csv", default=None)
    return parser.parse_args()


def subset_tag(args):
    return make_subset_tag(args.n_records, args.min_per_group, args.seed)


def output_root(args):
    return Path(args.output_root) if args.output_root else Path(args.root) / "xai_metrics"


def output_dir(exp, explainer, args):
    return (
        output_root(args)
        / f"test_subset_{subset_tag(args)}"
        / exp["tau_dir"]
        / exp["dataset"]
        / exp["attribute"]
        / explainer
    )


def output_npz_path(exp, explainer, args):
    return output_dir(exp, explainer, args) / "xai_metrics.npz"


def explanations_path(exp, explainer, args):
    artifact_root = Path(args.artifact_root) if args.artifact_root else Path(args.root)
    return (
        artifact_root
        / exp["tau_dir"]
        / exp["dataset"]
        / exp["attribute"]
        / explanation_filename(
            explainer,
            args.n_records,
            args.min_per_group,
            args.seed,
        )
    )


def load_y_true(info):
    if "y_true" not in info:
        return None
    y_true = np.asarray(info["y_true"])
    if y_true.ndim > 1:
        return np.argmax(y_true, axis=1)
    return y_true.reshape(-1)


def load_bundle(exp, explainer, args):
    exp_dir = Path(exp["path"])
    info = np.load(exp_dir / "black_box_logits_and_data.npz", allow_pickle=True)
    X_all = np.asarray(info["X"], dtype=np.float32)
    y_pred_all = np.asarray(info["y_pred"], dtype=np.int64).reshape(-1)
    groups_all = np.asarray(info["groups"]).reshape(-1)
    y_true = load_y_true(info)

    explanation_path = explanations_path(exp, explainer, args)
    explanations = np.load(explanation_path, allow_pickle=True)
    indices = np.asarray(explanations["indices"], dtype=np.int64)
    y_pred_expl = np.asarray(explanations["y_pred"], dtype=np.int64).reshape(-1)
    a_batch = np.asarray(explanations["target_attributions"], dtype=np.float32)
    method = str(np.asarray(explanations["method"]).item()) if "method" in explanations else explainer

    X = X_all[indices]
    y_pred = y_pred_all[indices]
    groups = groups_all[indices]
    y_true_subset = y_true[indices] if y_true is not None else None
    if not np.array_equal(y_pred, y_pred_expl):
        raise ValueError(f"Predicted labels in {explanation_path} do not match black-box cache.")
    if not (X.shape[0] == a_batch.shape[0] == y_pred.shape[0] == groups.shape[0]):
        raise ValueError(
            f"Shape mismatch for {explanation_path}: X={X.shape}, "
            f"a_batch={a_batch.shape}, y_pred={y_pred.shape}, groups={groups.shape}."
        )

    return {
        "path": explanation_path,
        "indices": indices,
        "X": X,
        "a_batch": a_batch,
        "groups": groups,
        "y_pred": y_pred,
        "y_true": y_true_subset,
        "method": method,
    }


def nonzero_attribution_mask(a_batch, idx):
    values = np.asarray(a_batch[idx], dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    finite = np.all(np.isfinite(values), axis=1)
    nonzero = np.linalg.norm(values.reshape(values.shape[0], -1), axis=1) > 0.0
    return finite & nonzero


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def pairwise_abs_gap(means):
    means = np.asarray(means, dtype=np.float64)
    means = means[np.isfinite(means)]
    if means.size < 2:
        return np.nan
    return float(np.max(means) - np.min(means))


def safe_scalar(value):
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.integer):
        return int(value)
    if np.issubdtype(arr.dtype, np.floating):
        return float(value)
    return value.item() if arr.ndim == 0 else value


def class_average_gap(scores, groups, y_pred, min_class_size, min_group_class_size):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rows = []

    for class_label in np.unique(y_pred):
        class_mask = y_pred == class_label
        if int(class_mask.sum()) < int(min_class_size):
            continue

        group_rows = []
        for group in np.unique(groups[class_mask]):
            mask = class_mask & (groups == group)
            if int(mask.sum()) < int(min_group_class_size):
                continue
            values = scores[mask]
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            group_rows.append(
                {
                    "class_label": safe_scalar(class_label),
                    "group": safe_scalar(group),
                    "n_total": int(mask.sum()),
                    "n_eval": int(finite.size),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                }
            )

        gap = pairwise_abs_gap([row["mean"] for row in group_rows])
        rows.append(
            {
                "class_label": safe_scalar(class_label),
                "n_class": int(class_mask.sum()),
                "n_valid_groups": int(len(group_rows)),
                "class_gap": gap,
                "group_rows": group_rows,
            }
        )

    valid_gaps = [row["class_gap"] for row in rows if np.isfinite(row["class_gap"])]
    return finite_mean(valid_gaps), rows


def make_sufficiency_metric(args):
    return quantus.Sufficiency(
        threshold=float(args.sufficiency_threshold),
        distance_func=str(args.sufficiency_distance_func),
        abs=bool(args.sufficiency_abs),
        normalise=bool(args.sufficiency_normalise),
        return_aggregate=False,
        disable_warnings=True,
        display_progressbar=False,
    )


def sufficiency_scores(metric, model, bundle, idx, args):
    idx = np.asarray(idx, dtype=np.int64)
    out = np.full(idx.size, np.nan, dtype=np.float32)
    valid_mask = nonzero_attribution_mask(bundle["a_batch"], idx)
    valid_idx = idx[valid_mask]
    if valid_idx.size == 0:
        return out
    batch_size = max(1, min(int(args.sufficiency_batch_size), int(valid_idx.size)))
    values = metric(
        model=model,
        x_batch=np.asarray(bundle["X"][valid_idx], dtype=np.float32),
        y_batch=np.asarray(bundle["y_pred"][valid_idx], dtype=np.int64),
        a_batch=np.asarray(bundle["a_batch"][valid_idx], dtype=np.float32),
        device=args.device,
        softmax=False,
        batch_size=batch_size,
    )
    out[valid_mask] = np.asarray(values, dtype=np.float32)
    return out


def consistency_scores(metric, model, bundle, idx, args):
    idx = np.asarray(idx, dtype=np.int64)
    out = np.full(idx.size, np.nan, dtype=np.float32)
    valid_mask = nonzero_attribution_mask(bundle["a_batch"], idx)
    valid_idx = idx[valid_mask]
    if valid_idx.size < 2:
        return out
    values = metric(
        model=model,
        x_batch=np.asarray(bundle["X"][valid_idx], dtype=np.float32),
        y_batch=np.asarray(bundle["y_pred"][valid_idx], dtype=np.int64),
        a_batch=np.asarray(bundle["a_batch"][valid_idx], dtype=np.float32),
        device=args.device,
        softmax=False,
        batch_size=int(valid_idx.size),
    )
    out[valid_mask] = np.asarray(values, dtype=np.float32)
    return out


def group_reference_scores(compute_subset, groups, min_group_size):
    n_total = int(groups.shape[0])
    out = np.full(n_total, np.nan, dtype=np.float32)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        if idx.size < int(min_group_size):
            continue
        out[idx] = compute_subset(idx)
    return out


def evaluate_sufficiency(exp, bundle, args):
    model = load_black_box(exp["path"], args.device)
    metric = make_sufficiency_metric(args)
    all_idx = np.arange(bundle["X"].shape[0], dtype=np.int64)
    compute_subset = lambda idx: sufficiency_scores(metric, model, bundle, idx, args)
    quality_scores = compute_subset(all_idx)
    gap_scores = group_reference_scores(compute_subset, bundle["groups"], args.min_group_size)
    return [("sufficiency", None, "higher", quality_scores, gap_scores)]


def evaluate_consistency(exp, bundle, args):
    model = load_black_box(exp["path"], args.device)
    metric = quantus.Consistency(
        discretise_func=partial(top_n_sign, n=int(args.consistency_discretise_n)),
        abs=bool(args.consistency_abs),
        normalise=True,
        return_aggregate=False,
        disable_warnings=True,
        display_progressbar=False,
    )
    all_idx = np.arange(bundle["X"].shape[0], dtype=np.int64)
    compute_subset = lambda idx: consistency_scores(metric, model, bundle, idx, args)
    quality_scores = compute_subset(all_idx)
    gap_scores = group_reference_scores(compute_subset, bundle["groups"], args.min_group_size)
    return [("consistency", None, "higher", quality_scores, gap_scores)]


def evaluate_stability(bundle, args):
    X_sim = prepare_similarity_features(bundle["X"], args.stability_feature_scaling)
    a_batch = prepare_explanations(bundle["a_batch"], args.stability_normalise_explanations)
    k_values = sorted(set(int(k) for k in args.stability_k_values))
    quality_by_k = compute_stability_scores(
        X_sim=X_sim,
        a_batch=a_batch,
        k_values=k_values,
        distance=args.stability_distance,
        rank_mode=args.stability_rank_mode,
        neighbour_batch_size=args.stability_neighbour_batch_size,
        n_jobs=args.stability_n_jobs,
    )
    gap_by_k = {k: np.full(bundle["X"].shape[0], np.nan, dtype=np.float32) for k in k_values}
    for group in np.unique(bundle["groups"]):
        idx = np.flatnonzero(bundle["groups"] == group)
        if idx.size < int(args.min_group_size):
            continue
        valid_k = [k for k in k_values if idx.size > k]
        if not valid_k:
            continue
        local = compute_stability_scores(
            X_sim=X_sim[idx],
            a_batch=a_batch[idx],
            k_values=valid_k,
            distance=args.stability_distance,
            rank_mode=args.stability_rank_mode,
            neighbour_batch_size=args.stability_neighbour_batch_size,
            n_jobs=args.stability_n_jobs,
        )
        for k, values in local.items():
            gap_by_k[k][idx] = values

    score_direction = "higher" if args.stability_distance in {"rank_agreement", "cosine_similarity"} else "lower"
    return [
        ("stability", k, score_direction, quality_by_k[k], gap_by_k[k])
        for k in k_values
    ]


def summarise_metric(
    exp,
    explainer,
    metric_name,
    k,
    score_direction,
    quality_scores,
    gap_scores,
    bundle,
    out_npz,
    args,
):
    quality_stats = finite_summary(quality_scores)
    gap, class_rows = class_average_gap(
        scores=gap_scores,
        groups=bundle["groups"],
        y_pred=bundle["y_pred"],
        min_class_size=int(args.min_class_size),
        min_group_class_size=int(args.min_group_class_size),
    )
    valid_class_gaps = [row["class_gap"] for row in class_rows if np.isfinite(row["class_gap"])]
    return {
        "status": "computed",
        "dataset": exp["dataset"],
        "attribute": exp["attribute"],
        "tau": exp["tau"],
        "tau_dir": exp["tau_dir"],
        "explainer": explainer,
        "metric": metric_name,
        "k": "" if k is None else int(k),
        "score_direction": score_direction,
        "quality_mean": quality_stats[0],
        "quality_std": quality_stats[1],
        "quality_n_eval": int(np.isfinite(quality_scores).sum()),
        "gap": gap,
        "n_classes_averaged": int(len(valid_class_gaps)),
        "min_class_gap": float(np.min(valid_class_gaps)) if valid_class_gaps else np.nan,
        "max_class_gap": float(np.max(valid_class_gaps)) if valid_class_gaps else np.nan,
        "n_total": int(bundle["X"].shape[0]),
        "output_npz": str(out_npz),
        "explanations_npz": str(bundle["path"]),
    }, class_rows


def flatten_class_rows(base_row, class_rows):
    rows = []
    for class_row in class_rows:
        row = {
            "dataset": base_row["dataset"],
            "attribute": base_row["attribute"],
            "tau": base_row["tau"],
            "tau_dir": base_row["tau_dir"],
            "explainer": base_row["explainer"],
            "metric": base_row["metric"],
            "k": base_row["k"],
            "class_label": class_row["class_label"],
            "n_class": class_row["n_class"],
            "n_valid_groups": class_row["n_valid_groups"],
            "class_gap": class_row["class_gap"],
        }
        rows.append(row)
    return rows


def run_one(exp, explainer, args):
    out_npz = output_npz_path(exp, explainer, args)
    if out_npz.exists() and not args.overwrite:
        data = np.load(out_npz, allow_pickle=True)
        return list(data["summary_rows"]), list(data["class_gap_rows"])

    bundle = load_bundle(exp, explainer, args)
    evaluations = []
    if "sufficiency" in args.metrics:
        evaluations.extend(evaluate_sufficiency(exp, bundle, args))
    if "consistency" in args.metrics:
        evaluations.extend(evaluate_consistency(exp, bundle, args))
    if "stability" in args.metrics:
        evaluations.extend(evaluate_stability(bundle, args))

    summary_rows = []
    class_gap_rows = []
    payload = {
        "indices": bundle["indices"],
        "groups": bundle["groups"],
        "y_pred": bundle["y_pred"],
        "y_true": bundle["y_true"] if bundle["y_true"] is not None else np.asarray([], dtype=np.int64),
        "meta": np.asarray(
            [
                {
                    "test_subset": subset_tag(args),
                    "method": bundle["method"],
                    "gap": "within_group_reference_local_class_average",
                    "class_labels": "black_box_predictions",
                    "quantus_version": str(getattr(quantus, "__version__", "unknown")),
                }
            ],
            dtype=object,
        ),
    }

    for metric_name, k, score_direction, quality_scores, gap_scores in evaluations:
        metric_key = metric_name if k is None else f"{metric_name}_k{int(k)}"
        payload[f"{metric_key}_quality_scores"] = np.asarray(quality_scores, dtype=np.float32)
        payload[f"{metric_key}_gap_reference_scores"] = np.asarray(gap_scores, dtype=np.float32)
        row, class_rows = summarise_metric(
            exp,
            explainer,
            metric_name,
            k,
            score_direction,
            quality_scores,
            gap_scores,
            bundle,
            out_npz,
            args,
        )
        summary_rows.append(row)
        class_gap_rows.extend(flatten_class_rows(row, class_rows))

    payload["summary_rows"] = np.asarray(summary_rows, dtype=object)
    payload["class_gap_rows"] = np.asarray(class_gap_rows, dtype=object)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)
    return summary_rows, class_gap_rows


def summary_csv_path(args):
    if args.summary_csv:
        return Path(args.summary_csv)
    return output_root(args) / f"test_subset_{subset_tag(args)}" / "xai_metrics.csv"


def write_outputs(summary_rows, class_gap_rows, args):
    summary_csv = summary_csv_path(args)
    class_csv = summary_csv.with_name(f"{summary_csv.stem}_class_gaps{summary_csv.suffix}")
    summary_fields = [
        "status",
        "dataset",
        "attribute",
        "tau",
        "tau_dir",
        "explainer",
        "metric",
        "k",
        "score_direction",
        "quality_mean",
        "quality_std",
        "quality_n_eval",
        "gap",
        "n_classes_averaged",
        "min_class_gap",
        "max_class_gap",
        "n_total",
        "output_npz",
        "explanations_npz",
        "error",
    ]
    class_fields = [
        "dataset",
        "attribute",
        "tau",
        "tau_dir",
        "explainer",
        "metric",
        "k",
        "class_label",
        "n_class",
        "n_valid_groups",
        "class_gap",
    ]
    write_csv(summary_csv, summary_rows, summary_fields)
    write_csv(class_csv, class_gap_rows, class_fields)
    return summary_csv, class_csv


def main():
    args = parse_args()
    experiments = discover_experiments(
        root=args.root,
        datasets=args.datasets,
        attributes=args.attributes,
        taus=args.taus,
    )
    print(f"Discovered {len(experiments)} experiments.")

    summary_rows = []
    class_gap_rows = []
    for exp in experiments:
        for explainer in args.explainers:
            label = f"{exp['tau_dir']}/{exp['dataset']}/{exp['attribute']}/{explainer}"
            if args.list_only:
                print(label)
                continue
            try:
                rows, class_rows = run_one(exp, explainer, args)
            except Exception as exc:
                print(f"ERROR {label}: {exc!r}")
                rows = [
                    {
                        "status": "error",
                        "dataset": exp["dataset"],
                        "attribute": exp["attribute"],
                        "tau": exp["tau"],
                        "tau_dir": exp["tau_dir"],
                        "explainer": explainer,
                        "metric": "",
                        "k": "",
                        "error": repr(exc),
                    }
                ]
                class_rows = []
            summary_rows.extend(rows)
            class_gap_rows.extend(class_rows)
            for row in rows:
                if row.get("status") == "computed":
                    print(
                        f"{label}: {row['metric']} {row['k']} "
                        f"quality={row['quality_mean']:.6g} gap={row['gap']:.6g}"
                    )

    if not args.list_only:
        summary_csv, class_csv = write_outputs(summary_rows, class_gap_rows, args)
        print(f"Wrote {summary_csv}")
        print(f"Wrote {class_csv}")


if __name__ == "__main__":
    main()
