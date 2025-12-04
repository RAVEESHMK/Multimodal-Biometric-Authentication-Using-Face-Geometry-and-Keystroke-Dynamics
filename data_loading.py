import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

RANDOM_STATE = 42


FACE_FEATURE_PATH = "face_features.npy"   # 2D array: (N_samples, D_face)
FACE_LABEL_PATH = "face_labels.npy"       # 1D array: (N_samples,)

# Keystroke dataset
KEYSTROKE_CSV_PATH = "keystroke_features.csv"
KEYSTROKE_USER_COL = "subject"            
KEYSTROKE_FEATURE_COLUMNS: Optional[list[str]] = None  # None = all numeric cols except subject


@dataclass
class ModalityData:
    X: np.ndarray  # (N_samples, D)
    y: np.ndarray  # (N_samples,)

    def assert_valid(self, name: str) -> None:
        assert self.X.ndim == 2, f"{name}: X must be 2D"
        assert self.y.ndim == 1, f"{name}: y must be 1D"
        assert self.X.shape[0] == self.y.shape[0], f"{name}: X,y sample mismatch"


def load_face_data(
    feature_path: str = FACE_FEATURE_PATH,
    label_path: str = FACE_LABEL_PATH,
) -> ModalityData:
    """
    Load face features and labels from .npy files.

    Supports:
      - X shape (N, D)      -> used as-is
      - X shape (N, 5, 2)   -> auto-flatten to (N, 10)
    """
    from sklearn.preprocessing import StandardScaler

    X = np.load(feature_path)
    y = np.load(label_path)

    # Auto-flatten landmark array if 3D: (N, 5, 2) -> (N, 10)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)

    X = X.astype(np.float32)
    y = y.astype(int)

    # Optional: standardize to make cosine similarity more meaningful
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    data = ModalityData(X=X, y=y)
    data.assert_valid("Face")
    return data


def load_keystroke_raw(
    csv_path: str = KEYSTROKE_CSV_PATH,
    user_col: str = KEYSTROKE_USER_COL,
    feature_cols: Optional[list[str]] = KEYSTROKE_FEATURE_COLUMNS,
) -> ModalityData:
    """
    Load keystroke features and labels from CSV.

    - user_col: subject ID (e.g., 'subject' with values like 's001')
    - feature_cols: numeric feature columns; if None, use all numeric cols except user_col.
    """
    df = pd.read_csv(csv_path)

    if user_col not in df.columns:
        raise ValueError(
            f"User column '{user_col}' not found. Available: {list(df.columns)}"
        )

    # Select numeric feature columns
    if feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if user_col in numeric_cols:
            numeric_cols.remove(user_col)
        feature_cols = numeric_cols

    if not feature_cols:
        raise ValueError(
            "No keystroke feature columns selected. "
            "Set KEYSTROKE_FEATURE_COLUMNS to a non-empty list."
        )

    X = df[feature_cols].to_numpy(dtype=np.float32)

    # Convert subject labels like 's001' -> 1, 's15' -> 15, etc.
    y_raw = df[user_col].astype(str)
    y_clean = y_raw.str.extract(r"(\d+)").astype(int)[0].to_numpy()
    y = y_clean

    data = ModalityData(X=X, y=y)
    data.assert_valid("Keystroke")
    return data


def align_modalities(
    face: ModalityData,
    keys: ModalityData,
    random_state: int = RANDOM_STATE,
) -> Tuple[ModalityData, ModalityData]:
    """
    Construct a synthetic multimodal population.

    Steps:
      - Let N = min(#face_users, #key_users)
      - Randomly pick N users from each modality
      - Map them to new IDs 0..N-1
    """
    rng = np.random.default_rng(random_state)

    face_users = np.unique(face.y)
    key_users = np.unique(keys.y)

    N = min(len(face_users), len(key_users))
    if N < 2:
        raise ValueError("Need at least 2 users in each modality.")

    face_sel = rng.choice(face_users, size=N, replace=False)
    key_sel = rng.choice(key_users, size=N, replace=False)
    new_ids = np.arange(N)

    face_map: Dict[int, int] = {old: nid for old, nid in zip(face_sel, new_ids)}
    key_map: Dict[int, int] = {old: nid for old, nid in zip(key_sel, new_ids)}

    # Filter & relabel face
    f_mask = np.isin(face.y, face_sel)
    Xf = face.X[f_mask]
    yf_old = face.y[f_mask]
    yf_new = np.array([face_map[int(u)] for u in yf_old], dtype=int)

    # Filter & relabel keystrokes
    k_mask = np.isin(keys.y, key_sel)
    Xk = keys.X[k_mask]
    yk_old = keys.y[k_mask]
    yk_new = np.array([key_map[int(u)] for u in yk_old], dtype=int)

    face_new = ModalityData(X=Xf, y=yf_new)
    keys_new = ModalityData(X=Xk, y=yk_new)

    face_new.assert_valid("Aligned Face")
    keys_new.assert_valid("Aligned Keys")

    return face_new, keys_new
