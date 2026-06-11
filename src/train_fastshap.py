import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from architectures import ArchitectureFactory
from experiment_utils import (
    build_target_attributions,
    load_black_box,
    make_model_from_state_dict,
    normalise_shap_output,
)
from fastshap import FastSHAP, KLDivLoss, Surrogate


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the FastSHAP surrogate and explainer for one FairLab black-box "
            "experiment, then export explanations on the experiment test set."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Directory containing black_box.h5 and black_box_logits_and_data.npz.",
    )
    parser.add_argument("--black-box", default=None, help="Optional black-box checkpoint path.")
    parser.add_argument("--prediction-cache", default=None, help="Optional black_box_logits_and_data.npz path.")
    parser.add_argument("--output", default=None, help="FastSHAP explanations .npz path.")
    parser.add_argument("--surrogate-output", default=None, help="Surrogate checkpoint path.")
    parser.add_argument("--explainer-output", default=None, help="FastSHAP explainer checkpoint path.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--surrogate-hidden1", type=int, default=512)
    parser.add_argument("--surrogate-hidden2", type=int, default=128)
    parser.add_argument("--surrogate-dropout", type=float, default=0.0)
    parser.add_argument("--surrogate-batch-size", type=int, default=128)
    parser.add_argument("--surrogate-epochs", type=int, default=200)
    parser.add_argument("--surrogate-lr", type=float, default=1e-4)
    parser.add_argument("--surrogate-eval-samples", type=int, default=64)
    parser.add_argument("--surrogate-lookback", type=int, default=5)

    parser.add_argument("--explainer-hidden1", type=int, default=256)
    parser.add_argument("--explainer-hidden2", type=int, default=128)
    parser.add_argument("--explainer-dropout", type=float, default=0.0)
    parser.add_argument("--explainer-batch-size", type=int, default=128)
    parser.add_argument("--explainer-num-samples", type=int, default=128)
    parser.add_argument("--explainer-epochs", type=int, default=300)
    parser.add_argument("--explainer-lr", type=float, default=1e-4)
    parser.add_argument("--explainer-eval-samples", type=int, default=64)
    parser.add_argument("--explainer-lookback", type=int, default=5)
    parser.add_argument("--explanation-batch-size", type=int, default=256)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_dict_to_cpu(module):
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
    }


def load_black_box_from_path(path, device):
    state_dict = torch.load(path, map_location=device)
    return make_model_from_state_dict(state_dict, device)


def make_surrogate_model(n_features, n_classes, args):
    return ArchitectureFactory.create_architecture(
        "mlp2hidden_surrogate",
        model_params={
            "input": int(n_features),
            "hidden1": int(args.surrogate_hidden1),
            "hidden2": int(args.surrogate_hidden2),
            "dropout": float(args.surrogate_dropout),
            "output": int(n_classes),
        },
    )


def make_explainer_model(n_features, n_classes, args):
    return ArchitectureFactory.create_architecture(
        "mlp2hidden_explainer",
        model_params={
            "input": int(n_features),
            "hidden1": int(args.explainer_hidden1),
            "hidden2": int(args.explainer_hidden2),
            "dropout": float(args.explainer_dropout),
            "output": int(n_classes),
        },
    )


def feature_names_from_cache(cache, n_features):
    if "feature_names" in cache:
        return np.asarray(cache["feature_names"], dtype=object)
    return np.asarray([f"feature_{i}" for i in range(n_features)], dtype=object)


def y_true_from_cache(cache):
    if "y_true" not in cache:
        return np.asarray([], dtype=np.int64)
    y_true = np.asarray(cache["y_true"])
    if y_true.ndim > 1:
        return np.argmax(y_true, axis=1)
    return y_true.reshape(-1)


def export_shap_values(fastshap, X_test, y_pred, y_true, groups, feature_names, output, args):
    n_test, n_features = X_test.shape
    with torch.no_grad():
        sample = torch.as_tensor(X_test[:1], dtype=torch.float32, device=args.device)
        raw = fastshap.shap_values(sample)
    raw_np = raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
    if raw_np.ndim == 2:
        n_classes = int(raw_np.shape[1])
    elif raw_np.ndim == 3:
        n_classes = int(raw_np.shape[2])
    else:
        raise ValueError(f"Unexpected FastSHAP output shape: {raw_np.shape}")

    values = np.zeros((n_test, n_features, n_classes), dtype=np.float32)
    for start in range(0, n_test, int(args.explanation_batch_size)):
        end = min(start + int(args.explanation_batch_size), n_test)
        xb = torch.as_tensor(X_test[start:end], dtype=torch.float32, device=args.device)
        sv = fastshap.shap_values(xb)
        if torch.is_tensor(sv):
            sv = sv.detach().cpu().numpy()
        values[start:end] = normalise_shap_output(
            sv,
            batch_size=end - start,
            n_features=n_features,
            n_classes=n_classes,
        )
        print(f"  FastSHAP explanations {end}/{n_test}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        shap_values=values,
        target_attributions=build_target_attributions(values, y_pred),
        y=np.asarray(y_pred, dtype=np.int64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
        y_true=np.asarray(y_true),
        groups=np.asarray(groups),
        feature_names=np.asarray(feature_names, dtype=object),
        explainer="fastshap",
        method="fastshap_surrogate_explainer",
        meta=np.asarray(
            [
                {
                    "surrogate_epochs": int(args.surrogate_epochs),
                    "surrogate_lr": float(args.surrogate_lr),
                    "surrogate_batch_size": int(args.surrogate_batch_size),
                    "explainer_epochs": int(args.explainer_epochs),
                    "explainer_lr": float(args.explainer_lr),
                    "explainer_batch_size": int(args.explainer_batch_size),
                    "explainer_num_samples": int(args.explainer_num_samples),
                    "seed": int(args.seed),
                    "device": str(args.device),
                }
            ],
            dtype=object,
        ),
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    exp_dir = Path(args.experiment_dir)
    cache_path = Path(args.prediction_cache) if args.prediction_cache else exp_dir / "black_box_logits_and_data.npz"
    black_box_path = Path(args.black_box) if args.black_box else exp_dir / "black_box.h5"
    surrogate_output = Path(args.surrogate_output) if args.surrogate_output else exp_dir / "surrogate_model.h5"
    explainer_output = Path(args.explainer_output) if args.explainer_output else exp_dir / "explainer_model.h5"
    output = Path(args.output) if args.output else exp_dir / "fastshap_explanations.npz"

    cache = np.load(cache_path, allow_pickle=True)
    X_train = np.asarray(cache["X_train"], dtype=np.float32)
    X_test = np.asarray(cache["X"], dtype=np.float32)
    y_pred = np.asarray(cache["y_pred"], dtype=np.int64).reshape(-1)
    y_true = y_true_from_cache(cache)
    groups = np.asarray(cache["groups"]).reshape(-1) if "groups" in cache else np.asarray([])
    n_features = int(X_train.shape[1])

    if black_box_path == exp_dir / "black_box.h5":
        black_box = load_black_box(exp_dir, args.device)
    else:
        black_box = load_black_box_from_path(black_box_path, args.device)
    black_box.eval()
    with torch.no_grad():
        logits = black_box(torch.as_tensor(X_test[:1], dtype=torch.float32, device=args.device))
    n_classes = int(logits.shape[1])
    original_model = nn.Sequential(black_box, nn.Softmax(dim=-1))

    surrogate_model = make_surrogate_model(n_features, n_classes, args).to(args.device)
    surrogate = Surrogate(surrogate_model, n_features)

    if surrogate_output.exists() and not args.overwrite:
        print(f"Loading surrogate from {surrogate_output}")
        surrogate.surrogate.load_state_dict(torch.load(surrogate_output, map_location=args.device))
    else:
        print("Training FastSHAP surrogate...")
        surrogate.train_original_model(
            X_train,
            X_test,
            original_model,
            batch_size=int(args.surrogate_batch_size),
            max_epochs=int(args.surrogate_epochs),
            loss_fn=KLDivLoss(),
            validation_samples=int(args.surrogate_eval_samples),
            lr=float(args.surrogate_lr),
            lookback=int(args.surrogate_lookback),
            training_seed=int(args.seed),
            validation_seed=int(args.seed),
            verbose=bool(args.verbose),
            bar=bool(args.verbose),
        )
        surrogate_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict_to_cpu(surrogate.surrogate), surrogate_output)

    explainer_model = make_explainer_model(n_features, n_classes, args).to(args.device)
    fastshap = FastSHAP(
        explainer_model,
        surrogate,
        normalization="additive",
        link=torch.nn.Softmax(dim=-1),
    )

    if explainer_output.exists() and not args.overwrite:
        print(f"Loading FastSHAP explainer from {explainer_output}")
        fastshap.explainer.load_state_dict(torch.load(explainer_output, map_location=args.device))
    else:
        print("Training FastSHAP explainer...")
        fastshap.train(
            X_train,
            X_test,
            batch_size=int(args.explainer_batch_size),
            num_samples=int(args.explainer_num_samples),
            max_epochs=int(args.explainer_epochs),
            validation_samples=int(args.explainer_eval_samples),
            lookback=int(args.explainer_lookback),
            lr=float(args.explainer_lr),
            training_seed=int(args.seed),
            validation_seed=int(args.seed),
            verbose=bool(args.verbose),
            bar=bool(args.verbose),
        )
        explainer_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict_to_cpu(fastshap.explainer), explainer_output)

    surrogate.surrogate.eval()
    fastshap.explainer.eval()
    feature_names = feature_names_from_cache(cache, n_features)
    print(f"Exporting FastSHAP explanations to {output}")
    export_shap_values(
        fastshap=fastshap,
        X_test=X_test,
        y_pred=y_pred,
        y_true=y_true,
        groups=groups,
        feature_names=feature_names,
        output=output,
        args=args,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
