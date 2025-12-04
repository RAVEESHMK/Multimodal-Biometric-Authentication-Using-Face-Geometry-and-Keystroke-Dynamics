from __future__ import annotations

import numpy as np

from main import Evaluator

from data_loading import (
    load_face_data,
    load_keystroke_raw,
    align_modalities,
)
from features import (
    make_keystroke_stats_representation,
    make_keystroke_pca_representation,
)
from matching_and_fusion import (
    build_template_query_sets,
    build_joint_template_query_sets,
    compute_similarity_scores,
    fuse_score_matrices,
    scores_from_similarity_matrix,
)

NUM_THRESHOLDS = 500


def evaluate_system(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    system_name: str,
):
    """
    Wrap HW3 Evaluator:
      - compute rates
      - plot score distribution, DET, ROC
      - print d-prime and EER
    """
    evaluator = Evaluator(
        num_thresholds=NUM_THRESHOLDS,
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        plot_title=system_name,
    )
    FPR, FNR, TPR = evaluator.compute_rates()
    evaluator.plot_score_distribution()
    evaluator.plot_det_curve(FPR, FNR)
    evaluator.plot_roc_curve(FPR, TPR)

    dprime = evaluator.get_dprime()
    eer, eer_thr = evaluator.get_EER(FPR, FNR)

    print(f"[{system_name}] d' = {dprime:.4f}, EER = {eer:.5f}, thr = {eer_thr:.4f}")
    return dprime, eer


# --------------------------------------------------
# RQ1 – unimodal vs fused
# --------------------------------------------------

def run_rq1(alpha: float = 0.5):
    """
    RQ1:
      Does score-level fusion of face and keystroke improve authentication
      performance compared to unimodal systems?
    """
    # 1. Load raw data
    face_raw = load_face_data()
    keys_raw = load_keystroke_raw()

    # 2. Keystroke stats representation
    keys_stats = make_keystroke_stats_representation(keys_raw)

    # 3. Align modalities -> synthetic multimodal users
    face_aligned, keys_aligned = align_modalities(face_raw, keys_stats)

    # 4. Build joint template/query splits
    face_split, key_split = build_joint_template_query_sets(face_aligned, keys_aligned)

    # 5. Compute similarity scores
    g_face, i_face, S_face = compute_similarity_scores(face_split)
    g_keys, i_keys, S_keys = compute_similarity_scores(key_split)

    print("\n=== RQ1: Face-only ===")
    dp_face, eer_face = evaluate_system(g_face, i_face, "RQ1_Face_only")

    print("\n=== RQ1: Keystroke-only (stats) ===")
    dp_keys, eer_keys = evaluate_system(g_keys, i_keys, "RQ1_Keys_stats_only")

    # 6. Score-level fusion
    S_fused = fuse_score_matrices(S_face, S_keys, alpha=alpha)
    g_fused, i_fused = scores_from_similarity_matrix(
        S_fused, face_split.y_templates, face_split.y_queries
    )

    print(f"\n=== RQ1: Fused (alpha={alpha:.2f}) ===")
    dp_fused, eer_fused = evaluate_system(g_fused, i_fused, f"RQ1_Fused_a{alpha:.2f}")

    print("\n=== RQ1 SUMMARY ===")
    print(f"Face-only:        d'={dp_face:.4f}, EER={eer_face:.5f}")
    print(f"Keys-only (stats):d'={dp_keys:.4f}, EER={eer_keys:.5f}")
    print(f"Fused (a={alpha:.2f}):  d'={dp_fused:.4f}, EER={eer_fused:.5f}")


# --------------------------------------------------
# RQ2 – keystroke feature representations
# --------------------------------------------------

def run_rq2(alpha: float = 0.5):
    """
    RQ2:
      How does keystroke feature representation (stats vs PCA)
      affect unimodal keystroke performance and the fused system?
    """
    face_raw = load_face_data()
    keys_raw = load_keystroke_raw()

    # Two different keystroke representations
    keys_stats = make_keystroke_stats_representation(keys_raw)
    keys_pca = make_keystroke_pca_representation(keys_raw)

    # --- Stats branch ---
    face_stats, keys_stats_aligned = align_modalities(face_raw, keys_stats, random_state=0)
    split_face_stats, split_keys_stats = build_joint_template_query_sets(
        face_stats, keys_stats_aligned, random_state=0
    )

    g_kstats, i_kstats, S_kstats = compute_similarity_scores(split_keys_stats)
    print("\n=== RQ2: Keystroke-only (stats) ===")
    dp_kstats, eer_kstats = evaluate_system(g_kstats, i_kstats, "RQ2_Keys_stats_only")

    g_fstats_face, i_fstats_face, S_fstats = compute_similarity_scores(split_face_stats)
    S_fused_stats = fuse_score_matrices(S_fstats, S_kstats, alpha=alpha)
    g_fused_stats, i_fused_stats = scores_from_similarity_matrix(
        S_fused_stats, split_face_stats.y_templates, split_face_stats.y_queries
    )
    print(f"\n=== RQ2: Fused (face + keys_stats, alpha={alpha:.2f}) ===")
    dp_fused_stats, eer_fused_stats = evaluate_system(
        g_fused_stats, i_fused_stats, f"RQ2_Fused_stats_a{alpha:.2f}"
    )

    # --- PCA branch ---
    face_pca, keys_pca_aligned = align_modalities(face_raw, keys_pca, random_state=1)
    split_face_pca, split_keys_pca = build_joint_template_query_sets(
        face_pca, keys_pca_aligned, random_state=1
    )

    g_kpca, i_kpca, S_kpca = compute_similarity_scores(split_keys_pca)
    print("\n=== RQ2: Keystroke-only (PCA) ===")
    dp_kpca, eer_kpca = evaluate_system(g_kpca, i_kpca, "RQ2_Keys_pca_only")

    g_fpca_face, i_fpca_face, S_f_pca = compute_similarity_scores(split_face_pca)
    S_fused_pca = fuse_score_matrices(S_f_pca, S_kpca, alpha=alpha)
    g_fused_pca, i_fused_pca = scores_from_similarity_matrix(
        S_fused_pca, split_face_pca.y_templates, split_face_pca.y_queries
    )
    print(f"\n=== RQ2: Fused (face + keys_pca, alpha={alpha:.2f}) ===")
    dp_fused_pca, eer_fused_pca = evaluate_system(
        g_fused_pca, i_fused_pca, f"RQ2_Fused_pca_a{alpha:.2f}"
    )

    print("\n=== RQ2 SUMMARY ===")
    print(f"Keys-only (stats): d'={dp_kstats:.4f}, EER={eer_kstats:.5f}")
    print(f"Keys-only (PCA):   d'={dp_kpca:.4f}, EER={eer_kpca:.5f}")
    print(f"Fused (stats):     d'={dp_fused_stats:.4f}, EER={eer_fused_stats:.5f}")
    print(f"Fused (PCA):       d'={dp_fused_pca:.4f}, EER={eer_fused_pca:.5f}")


def main():
    print(">>> Running RQ1 (unimodal vs fused)...")
    run_rq1(alpha=0.5)

    print("\n\n>>> Running RQ2 (keystroke feature representations)...")
    run_rq2(alpha=0.5)


if __name__ == "__main__":
    main()
