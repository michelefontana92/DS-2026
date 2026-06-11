import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment_utils import (
    DATASET_ORDER,
    discover_experiments,
    experiment_key,
    subset_dir,
    subset_path,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create fixed test subsets for explanation experiments. Sampling "
            "is stratified by sensitive group, approximately preserving group "
            "proportions while enforcing a minimum per group when possible."
        )
    )
    parser.add_argument("--root", default="results/New_Experiments")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root where test_subsets will be saved. Defaults to --root.",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--attributes", nargs="*", default=None)
    parser.add_argument("--taus", nargs="*", default=None)
    parser.add_argument("--n-records", type=int, default=1000)
    parser.add_argument("--min-per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing subset files.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list dataset/attribute pairs and test-set alignment.",
    )
    return parser.parse_args()


def group_experiments(experiments):
    grouped = defaultdict(list)
    for exp in experiments:
        grouped[experiment_key(exp)].append(exp)
    return grouped


def grouped_sort_key(item):
    _, exps = item
    dataset = exps[0]["dataset"]
    attribute = exps[0]["attribute"]
    dataset_pos = DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else 99
    return dataset_pos, dataset, attribute


def load_test_signature(exp):
    data = np.load(Path(exp["path"]) / "black_box_logits_and_data.npz", allow_pickle=True)
    return {
        "X": np.asarray(data["X"]),
        "y_true": np.asarray(data["y_true"]) if "y_true" in data else None,
        "groups": np.asarray(data["groups"]).reshape(-1),
    }


def validate_same_test_set(exps):
    ref = load_test_signature(exps[0])
    issues = []
    for exp in exps[1:]:
        cur = load_test_signature(exp)
        if ref["X"].shape != cur["X"].shape or not np.array_equal(ref["X"], cur["X"]):
            issues.append(f"{exp['tau_dir']}: X differs")
        if ref["groups"].shape != cur["groups"].shape or not np.array_equal(ref["groups"], cur["groups"]):
            issues.append(f"{exp['tau_dir']}: groups differ")
        if ref["y_true"] is not None and cur["y_true"] is not None:
            if ref["y_true"].shape != cur["y_true"].shape or not np.array_equal(ref["y_true"], cur["y_true"]):
                issues.append(f"{exp['tau_dir']}: y_true differs")
    return ref, issues


def allocate_counts(group_counts, n_records, min_per_group):
    groups = np.asarray([g for g, _ in group_counts])
    counts = np.asarray([c for _, c in group_counts], dtype=int)
    n_total = int(counts.sum())
    target_n = min(int(n_records), n_total)

    if target_n >= n_total:
        return dict(zip(groups, counts))

    exact = target_n * counts / float(n_total)
    allocation = np.floor(exact).astype(int)
    positive = counts > 0
    allocation[positive] = np.maximum(allocation[positive], 1)

    minimum = np.minimum(counts, int(min_per_group))
    allocation = np.maximum(allocation, minimum)
    allocation = np.minimum(allocation, counts)

    if int(allocation.sum()) > target_n and int(minimum.sum()) > target_n:
        raise ValueError(
            "Cannot satisfy the requested minimum per group within n_records: "
            f"minimum_sum={int(minimum.sum())}, n_records={target_n}."
        )

    while int(allocation.sum()) > target_n:
        reducible = np.flatnonzero(allocation > minimum)
        if reducible.size == 0:
            break
        over = allocation - exact
        i = reducible[np.argmax(over[reducible])]
        allocation[i] -= 1

    fractional = exact - np.floor(exact)
    while int(allocation.sum()) < target_n:
        expandable = np.flatnonzero(allocation < counts)
        if expandable.size == 0:
            break
        i = expandable[np.argmax(fractional[expandable])]
        allocation[i] += 1
        fractional[i] = 0.0

    return dict(zip(groups, allocation.astype(int)))


def stratified_sample(groups, n_records, min_per_group, seed):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups).reshape(-1)
    unique_groups, counts = np.unique(groups, return_counts=True)
    allocation = allocate_counts(list(zip(unique_groups, counts)), n_records, min_per_group)

    selected = []
    for group in unique_groups:
        idx = np.flatnonzero(groups == group)
        k = int(allocation[group])
        if k <= 0:
            continue
        chosen = rng.choice(idx, size=min(k, idx.size), replace=False)
        selected.append(chosen)

    if not selected:
        return np.empty(0, dtype=np.int64), allocation
    indices = np.sort(np.concatenate(selected).astype(np.int64))
    return indices, allocation


def write_subset_report(report_rows, root, n_records, min_per_group, seed):
    out = subset_dir(root) / f"subset_report_n{n_records}_min{min_per_group}_seed{seed}.csv"
    fieldnames = [
        "dataset",
        "attribute",
        "tau_dirs",
        "test_aligned",
        "alignment_issues",
        "n_test",
        "n_requested",
        "n_selected",
        "n_groups",
        "group",
        "group_n_test",
        "group_fraction_test",
        "group_n_selected",
        "group_fraction_selected",
        "subset_path",
    ]
    write_csv(out, report_rows, fieldnames)
    return out


def main():
    args = parse_args()
    root = Path(args.root)
    output_root = Path(args.output_root) if args.output_root else root
    experiments = discover_experiments(
        root=root,
        datasets=args.datasets,
        attributes=args.attributes,
        taus=args.taus,
    )
    grouped = group_experiments(experiments)
    print(f"Discovered {len(experiments)} experiments in {len(grouped)} dataset/attribute pairs.")

    subset_dir(output_root).mkdir(parents=True, exist_ok=True)
    report_rows = []

    for key, exps_for_key in sorted(grouped.items(), key=grouped_sort_key):
        exps = sorted(exps_for_key, key=lambda exp: exp["tau"])
        dataset = exps[0]["dataset"]
        attribute = exps[0]["attribute"]
        ref, issues = validate_same_test_set(exps)
        groups = ref["groups"]
        out_path = subset_path(
            output_root,
            dataset,
            attribute,
            args.n_records,
            args.min_per_group,
            args.seed,
        )
        aligned = len(issues) == 0
        print(
            f"{dataset}/{attribute}: n={groups.shape[0]}, taus={len(exps)}, "
            f"aligned={aligned}, subset={out_path}"
        )
        if issues:
            for issue in issues:
                print(f"  - {issue}")

        indices = np.empty(0, dtype=np.int64)
        allocation = {}
        if not args.list_only:
            if out_path.exists() and not args.overwrite:
                saved = np.load(out_path, allow_pickle=True)
                indices = np.asarray(saved["indices"], dtype=np.int64)
                allocation = {
                    group: int(n)
                    for group, n in zip(saved["group_values"], saved["group_n_selected"])
                }
                print(f"  reusing existing subset with n={indices.size}")
            else:
                indices, allocation = stratified_sample(
                    groups=groups,
                    n_records=args.n_records,
                    min_per_group=args.min_per_group,
                    seed=args.seed,
                )
                unique_groups, counts = np.unique(groups, return_counts=True)
                selected_counts = np.asarray(
                    [int(np.sum(groups[indices] == group)) for group in unique_groups],
                    dtype=np.int64,
                )
                np.savez_compressed(
                    out_path,
                    indices=indices,
                    group_values=unique_groups,
                    group_n_test=counts.astype(np.int64),
                    group_n_selected=selected_counts,
                    dataset=dataset,
                    attribute=attribute,
                    n_records_requested=int(args.n_records),
                    min_per_group=int(args.min_per_group),
                    seed=int(args.seed),
                    tau_dirs=np.asarray([exp["tau_dir"] for exp in exps], dtype=object),
                    test_aligned=bool(aligned),
                    alignment_issues=np.asarray(issues, dtype=object),
                )
                print(f"  saved subset with n={indices.size}")

        unique_groups, counts = np.unique(groups, return_counts=True)
        selected_groups = groups[indices] if indices.size else np.empty(0, dtype=groups.dtype)
        for group, count in zip(unique_groups, counts):
            selected_count = int(np.sum(selected_groups == group)) if indices.size else int(allocation.get(group, 0))
            report_rows.append(
                {
                    "dataset": dataset,
                    "attribute": attribute,
                    "tau_dirs": ";".join(exp["tau_dir"] for exp in exps),
                    "test_aligned": aligned,
                    "alignment_issues": "|".join(issues),
                    "n_test": int(groups.shape[0]),
                    "n_requested": int(args.n_records),
                    "n_selected": int(indices.size) if indices.size else "",
                    "n_groups": int(unique_groups.size),
                    "group": int(group)
                    if np.issubdtype(np.asarray(group).dtype, np.integer)
                    else group,
                    "group_n_test": int(count),
                    "group_fraction_test": float(count / groups.shape[0]),
                    "group_n_selected": selected_count,
                    "group_fraction_selected": float(selected_count / max(indices.size, 1))
                    if indices.size
                    else "",
                    "subset_path": str(out_path),
                }
            )

    if not args.list_only:
        report = write_subset_report(
            report_rows,
            output_root,
            args.n_records,
            args.min_per_group,
            args.seed,
        )
        print(f"Wrote {report}")


if __name__ == "__main__":
    main()
