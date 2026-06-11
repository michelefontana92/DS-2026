import numpy as np
from sklearn.neighbors import NearestNeighbors


def prepare_similarity_features(X, feature_scaling="standardize"):
    """Return row-normalised features used to define cosine-similarity neighbours."""
    X = np.asarray(X, dtype=np.float32)
    if feature_scaling == "standardize":
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        std = np.where(std > 1e-12, std, 1.0).astype(np.float32)
        X = (X - mean.astype(np.float32)) / std
    elif feature_scaling != "none":
        raise ValueError(f"Unknown feature scaling: {feature_scaling}")

    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, norm, out=np.zeros_like(X, dtype=np.float32), where=norm > 0)


def prepare_explanations(a_batch, normalise=False):
    a_batch = np.asarray(a_batch, dtype=np.float32)
    if not normalise:
        return a_batch
    norm = np.linalg.norm(a_batch, axis=1, keepdims=True)
    return np.divide(a_batch, norm, out=np.zeros_like(a_batch, dtype=np.float32), where=norm > 0)


def top_feature_order(values, rank_mode="signed"):
    values = np.asarray(values, dtype=np.float32)
    if rank_mode == "signed":
        scores = values
    elif rank_mode == "absolute":
        scores = np.abs(values)
    else:
        raise ValueError(f"Unknown rank mode: {rank_mode}")
    return np.argsort(-scores, axis=-1, kind="stable")


def explanation_pair_scores(a_center, a_neighbours, distance, rank_mode):
    if distance == "rank_agreement":
        center_order = top_feature_order(a_center, rank_mode)
        neighbour_order = top_feature_order(a_neighbours, rank_mode)
        return np.mean(center_order[:, None, :] == neighbour_order, axis=2)

    if distance == "cosine_similarity":
        center_norm = _l2_normalise_rows(a_center)
        neighbour_norm = _l2_normalise_rows(a_neighbours)
        return np.sum(center_norm[:, None, :] * neighbour_norm, axis=2)

    diff = a_center[:, None, :] - a_neighbours
    squared = diff * diff
    if distance == "mse":
        return np.mean(squared, axis=2)
    if distance == "rmse":
        return np.sqrt(np.mean(squared, axis=2))
    if distance == "euclidean":
        return np.sqrt(np.sum(squared, axis=2))
    raise ValueError(f"Unknown explanation distance: {distance}")


def _l2_normalise_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norm, out=np.zeros_like(values, dtype=np.float32), where=norm > 0)


def _clean_neighbour_indices(raw_indices, row_start, max_k):
    cleaned = np.empty((raw_indices.shape[0], max_k), dtype=np.int64)
    for offset, neighbours in enumerate(raw_indices):
        self_index = row_start + offset
        valid = neighbours[neighbours != self_index]
        if valid.size < max_k:
            raise RuntimeError(
                f"Only found {valid.size} non-self neighbours for row {self_index}."
            )
        cleaned[offset] = valid[:max_k]
    return cleaned


def compute_stability_scores(
    X_sim,
    a_batch,
    k_values=(5, 10),
    distance="rank_agreement",
    rank_mode="signed",
    neighbour_batch_size=4096,
    n_jobs=1,
):
    """Compute average explanation agreement/distance over top-k cosine neighbours."""
    k_values = sorted(set(int(k) for k in k_values))
    max_k = max(k_values)
    n_samples = int(X_sim.shape[0])
    if n_samples <= max_k:
        return {k: np.full(n_samples, np.nan, dtype=np.float32) for k in k_values}

    nn = NearestNeighbors(
        n_neighbors=max_k + 1,
        metric="euclidean",
        algorithm="auto",
        n_jobs=n_jobs,
    )
    nn.fit(X_sim)

    scores = {k: np.empty(n_samples, dtype=np.float32) for k in k_values}
    for start in range(0, n_samples, int(neighbour_batch_size)):
        end = min(start + int(neighbour_batch_size), n_samples)
        raw_indices = nn.kneighbors(X_sim[start:end], return_distance=False)
        neighbour_indices = _clean_neighbour_indices(raw_indices, start, max_k)
        pair_scores = explanation_pair_scores(
            a_center=a_batch[start:end],
            a_neighbours=a_batch[neighbour_indices],
            distance=distance,
            rank_mode=rank_mode,
        )
        cumulative = np.cumsum(pair_scores, axis=1)
        for k in k_values:
            scores[k][start:end] = cumulative[:, k - 1] / float(k)
    return scores
