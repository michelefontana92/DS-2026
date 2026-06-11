import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from architectures import ArchitectureFactory
from dataloaders import DataModule
from runs.CompasCentralized.compas_run import CentralizedCompasRun
from runs.Education.education_run import EducationRun
from runs.FolkTablesBinary.folk_run import FolkTablesBinaryRun
from runs.Income_3.income_3_run import Income3Run
from runs.MEPCentralized.mep_run import CentralizedMEPRun


RUN_CONFIGS = {
    "compas_fairlab": CentralizedCompasRun,
    "education_fairlab": EducationRun,
    "income_fairlab": FolkTablesBinaryRun,
    "income_3_fairlab": Income3Run,
    "mep_fairlab": CentralizedMEPRun,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export black-box logits, probabilities, labels, sensitive groups, "
            "and training features required by the explainer scripts. The "
            "evaluation split is referred to as test in the release, even when "
            "the original FairLab files use *_val.csv names."
        )
    )
    parser.add_argument("--run", required=True, choices=sorted(RUN_CONFIGS))
    parser.add_argument("--checkpoint", required=True, help="FairLab black-box checkpoint.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sensitive-attribute", required=True)
    parser.add_argument("--root-dir", default=None, help="Root directory containing dataset folders.")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--train-set", default=None)
    parser.add_argument("--test-set", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def default_split_path(run_config, args, split):
    if args.experiment_name is None:
        prefix = f"node_{args.start_index}/{run_config.dataset}"
    else:
        prefix = f"{args.experiment_name}/node_{args.start_index}/{run_config.dataset}"
    return f"{prefix}_{split}.csv"


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


def collect_split(loader, sensitive_attribute):
    X, y, groups = [], [], []
    for batch in loader:
        X.append(batch["data"].detach().cpu().numpy())
        y.append(batch["labels"].detach().cpu().numpy())
        groups.append(batch["groups_tensor"][sensitive_attribute].detach().cpu().numpy())
    return (
        np.concatenate(X, axis=0).astype(np.float32),
        np.concatenate(y, axis=0).reshape(-1),
        np.concatenate(groups, axis=0).reshape(-1),
    )


def predict(model, X, device, batch_size):
    logits = []
    with torch.no_grad():
        for start in range(0, X.shape[0], int(batch_size)):
            xb = torch.as_tensor(X[start:start + int(batch_size)], dtype=torch.float32, device=device)
            logits.append(model(xb).detach().cpu())
    logits = torch.cat(logits, dim=0)
    proba = torch.softmax(logits, dim=1).numpy()
    return logits.numpy(), proba, np.argmax(proba, axis=1)


def main():
    args = parse_args()
    run_config = RUN_CONFIGS[args.run](root_dir=args.root_dir)
    train_set = args.train_set or default_split_path(run_config, args, "train")
    test_set = args.test_set or default_split_path(run_config, args, "val")

    sensitive_names = {name for name, _ in run_config.sensitive_attributes}
    if args.sensitive_attribute not in sensitive_names:
        raise ValueError(
            f"Unknown sensitive attribute {args.sensitive_attribute!r}. "
            f"Available attributes: {sorted(sensitive_names)}"
        )

    data_module = DataModule(
        dataset=run_config.dataset,
        root=run_config.data_root,
        train_set=train_set,
        val_set=test_set,
        test_set=test_set,
        batch_size=args.batch_size,
        num_workers=0,
        sensitive_attributes=run_config.sensitive_attributes,
    )

    checkpoint = Path(args.checkpoint)
    state_dict = torch.load(checkpoint, map_location=args.device)
    model = make_model_from_state_dict(state_dict, args.device)

    X_train, y_train, train_groups = collect_split(
        data_module.train_loader_eval(batch_size=args.batch_size),
        args.sensitive_attribute,
    )
    X_test, y_test, groups = collect_split(
        data_module.val_loader(batch_size=args.batch_size),
        args.sensitive_attribute,
    )
    logits, proba, y_pred = predict(model, X_test, args.device, args.batch_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "black_box_logits_and_data.npz"
    np.savez_compressed(
        cache_path,
        X=X_test,
        X_train=X_train,
        y_true=y_test,
        y_train=y_train,
        groups=groups,
        train_groups=train_groups,
        logits=logits.astype(np.float32),
        proba=proba.astype(np.float32),
        y_pred=y_pred.astype(np.int64),
        feature_names=np.asarray(data_module.feature_names(), dtype=object),
        cat_cols_mask=np.asarray(data_module.get_cat_cols_mask(), dtype=bool),
        sensitive_attribute=str(args.sensitive_attribute),
        run=str(args.run),
        train_set=str(train_set),
        test_set=str(test_set),
    )
    shutil.copyfile(checkpoint, output_dir / "black_box.h5")
    print(f"Wrote {cache_path}")
    print(f"Wrote {output_dir / 'black_box.h5'}")


if __name__ == "__main__":
    main()
