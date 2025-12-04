import numpy as np
from dataclasses import dataclass
from typing import Tuple
from sklearn.metrics.pairwise import cosine_similarity

from data_loading import ModalityData, RANDOM_STATE


@dataclass
class TemplateQuerySplit:
    X_templates: np.ndarray
    y_templates: np.ndarray
    X_queries: np.ndarray
    y_queries: np.ndarray


def build_template_query_sets(
    data: ModalityData,
    random_state: int = RANDOM_STATE,
) -> TemplateQuerySplit:
    """
    Basic template/query split for a single modality:
      - For each user: 1 template + remaining queries.
    Used mainly for unimodal experiments when fusion is not required.
    """
    rng = np.random.default_rng(random_state)
    X, y = data.X, data.y
    unique_users = np.unique(y)

    template_vecs, template_labels = [], []
    query_vecs, query_labels = [], []

    for u in unique_users:
        idx = np.where(y == u)[0]
        if len(idx) < 2:
            continue
        rng.shuffle(idx)
        t_idx = idx[0]
        q_idx = idx[1:]

        template_vecs.append(X[t_idx])
        template_labels.append(u)

        query_vecs.extend(X[q_idx])
        query_labels.extend([u] * len(q_idx))

    if not template_vecs:
        raise ValueError("No users with >=2 samples to build template/query sets.")

    X_templates = np.vstack(template_vecs).astype(np.float32)
    X_queries = np.vstack(query_vecs).astype(np.float32)
    y_templates = np.array(template_labels, dtype=int)
    y_queries = np.array(query_labels, dtype=int)

    return TemplateQuerySplit(
        X_templates=X_templates,
        y_templates=y_templates,
        X_queries=X_queries,
        y_queries=y_queries,
    )


def build_joint_template_query_sets(
    face: ModalityData,
    keys: ModalityData,
    random_state: int = RANDOM_STATE,
) -> Tuple[TemplateQuerySplit, TemplateQuerySplit]:
    """
    Build *aligned* template/query splits for face and keystroke data.

    For each user u that appears in BOTH modalities and has at least
    2 samples in each:
      - choose 1 face sample as face template
      - choose 1 keystroke sample as keystroke template
      - use up to min(#face_remaining, #keys_remaining) queries for that user
        so that both modalities have the SAME queries in the SAME order.

    Returns:
        face_split, keys_split (TemplateQuerySplit, TemplateQuerySplit)
    """
    rng = np.random.default_rng(random_state)

    face_users = np.unique(face.y)
    key_users = np.unique(keys.y)
    common_users = np.intersect1d(face_users, key_users)

    face_templates = []
    key_templates = []
    template_labels = []

    face_queries = []
    key_queries = []
    query_labels = []

    for u in common_users:
        f_idx_all = np.where(face.y == u)[0]
        k_idx_all = np.where(keys.y == u)[0]

        # Need at least 2 samples in each modality
        if len(f_idx_all) < 2 or len(k_idx_all) < 2:
            continue

        rng.shuffle(f_idx_all)
        rng.shuffle(k_idx_all)

        f_temp = f_idx_all[0]
        k_temp = k_idx_all[0]

        f_q_idx = f_idx_all[1:]
        k_q_idx = k_idx_all[1:]

        n_q = min(len(f_q_idx), len(k_q_idx))
        if n_q == 0:
            continue

        # templates
        template_labels.append(u)
        face_templates.append(face.X[f_temp])
        key_templates.append(keys.X[k_temp])

        # queries
        for fi, ki in zip(f_q_idx[:n_q], k_q_idx[:n_q]):
            face_queries.append(face.X[fi])
            key_queries.append(keys.X[ki])
            query_labels.append(u)

    if not template_labels:
        raise ValueError(
            "No users with at least 2 samples in both modalities for joint template/query sets."
        )

    y_templates = np.array(template_labels, dtype=int)
    y_queries = np.array(query_labels, dtype=int)

    face_split = TemplateQuerySplit(
        X_templates=np.vstack(face_templates).astype(np.float32),
        y_templates=y_templates,
        X_queries=np.vstack(face_queries).astype(np.float32),
        y_queries=y_queries,
    )

    keys_split = TemplateQuerySplit(
        X_templates=np.vstack(key_templates).astype(np.float32),
        y_templates=y_templates,
        X_queries=np.vstack(key_queries).astype(np.float32),
        y_queries=y_queries,
    )

    return face_split, keys_split


def compute_similarity_scores(
    split: TemplateQuerySplit,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cosine similarity between every query and every template.

    Returns:
        genuine_scores, impostor_scores, S (similarity matrix)
    """
    S = cosine_similarity(split.X_queries, split.X_templates)

    genuine_scores = []
    impostor_scores = []

    label_to_col = {u: i for i, u in enumerate(split.y_templates)}

    for q_idx, q_label in enumerate(split.y_queries):
        for t_label, t_col in label_to_col.items():
            score = S[q_idx, t_col]
            if q_label == t_label:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    return (
        np.array(genuine_scores, dtype=np.float32),
        np.array(impostor_scores, dtype=np.float32),
        S,
    )


def fuse_score_matrices(
    S_face: np.ndarray,
    S_keys: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Score-level fusion:
        S_fused = alpha * S_face + (1-alpha) * S_keys
    """
    if S_face.shape != S_keys.shape:
        raise ValueError(f"Shape mismatch for fusion: {S_face.shape} vs {S_keys.shape}")
    return alpha * S_face + (1.0 - alpha) * S_keys


def scores_from_similarity_matrix(
    S: np.ndarray,
    y_templates: np.ndarray,
    y_queries: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert similarity matrix to genuine/impostor arrays."""
    genuine_scores = []
    impostor_scores = []

    label_to_col = {u: i for i, u in enumerate(y_templates)}

    for q_idx, q_label in enumerate(y_queries):
        for t_label, t_col in label_to_col.items():
            score = S[q_idx, t_col]
            if q_label == t_label:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    return (
        np.array(genuine_scores, dtype=np.float32),
        np.array(impostor_scores, dtype=np.float32),
    )
