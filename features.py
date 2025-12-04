import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_loading import ModalityData, RANDOM_STATE

KEYSTROKE_PCA_COMPONENTS = 20 


def make_keystroke_stats_representation(data: ModalityData) -> ModalityData:
    """
    'Stats' representation:
      - Standardize all raw features
      - Append per-sample mean and std across features
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.X)

    sample_mean = X_scaled.mean(axis=1, keepdims=True)
    sample_std = X_scaled.std(axis=1, keepdims=True)

    X_stats = np.hstack([X_scaled, sample_mean, sample_std])
    return ModalityData(X=X_stats.astype(np.float32), y=data.y.copy())


def make_keystroke_pca_representation(
    data: ModalityData,
    n_components: int = KEYSTROKE_PCA_COMPONENTS,
) -> ModalityData:
    """
    PCA representation:
      - Standardize
      - PCA to n_components
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.X)

    n_components = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    return ModalityData(X=X_pca.astype(np.float32), y=data.y.copy())
