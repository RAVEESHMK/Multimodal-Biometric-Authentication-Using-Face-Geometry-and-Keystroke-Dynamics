import numpy as np
import matplotlib.pyplot as plt


class Evaluator:
    """
    Evaluator class for biometric system performance.
    Computes FPR, FNR, TPR across thresholds,
    score distributions, ROC, DET, d-prime, EER.
    """

    def __init__(
        self,
        num_thresholds: int,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        plot_title: str,
        epsilon: float = 1e-12,
    ):
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(-0.1, 1.1, num_thresholds)
        self.genuine_scores = np.asarray(genuine_scores, dtype=float)
        self.impostor_scores = np.asarray(impostor_scores, dtype=float)
        self.plot_title = plot_title
        self.epsilon = epsilon

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    def get_dprime(self) -> float:
        """Compute d-prime between genuine and impostor score distributions."""

        g_mean = np.mean(self.genuine_scores)
        i_mean = np.mean(self.impostor_scores)
        g_std = np.std(self.genuine_scores)
        i_std = np.std(self.impostor_scores)

        numerator = g_mean - i_mean
        denominator = np.sqrt(0.5 * (g_std**2 + i_std**2)) + self.epsilon
        return float(numerator / denominator)

    def get_EER(self, FPR: np.ndarray, FNR: np.ndarray):
        """
        Compute Equal Error Rate (EER) using linear interpolation
        at the crossing point of FPR and FNR.
        """

        diff = FPR - FNR
        zero_idx = np.where(np.isclose(diff, 0, atol=1e-6))[0]

        # Exact crossing
        if len(zero_idx) > 0:
            idx = zero_idx[0]
            return float(FPR[idx]), float(self.thresholds[idx])

        # Look for sign change
        sign_change = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
        if len(sign_change) > 0:
            idx = sign_change[0]

            x0, x1 = FPR[idx], FPR[idx + 1]
            y0, y1 = FNR[idx], FNR[idx + 1]
            t0, t1 = self.thresholds[idx], self.thresholds[idx + 1]

            denom = (y1 - y0) - (x1 - x0)
            if abs(denom) < 1e-12:
                eer = 0.5 * (x0 + y0)
                thr = 0.5 * (t0 + t1)
            else:
                lam = (y0 - x0) / denom
                eer = x0 + lam * (x1 - x0)
                thr = t0 + lam * (t1 - t0)

            return float(eer), float(thr)

        # Fallback: minimal absolute difference
        idx = np.argmin(np.abs(diff))
        return float(0.5 * (FPR[idx] + FNR[idx])), float(self.thresholds[idx])

    # ---------------------------------------------------------
    # Rate Computation
    # ---------------------------------------------------------

    def compute_rates(self):
        """Compute FPR, FNR, TPR for all thresholds."""

        FPR = []
        FNR = []
        TPR = []

        total_g = len(self.genuine_scores)
        total_i = len(self.impostor_scores)

        for thr in self.thresholds:

            # Genuine scores
            TP = np.sum(self.genuine_scores >= thr)
            FN = total_g - TP

            # Impostor scores
            FP = np.sum(self.impostor_scores >= thr)
            TN = total_i - FP

            # Rates
            fpr = FP / (FP + TN + self.epsilon)
            fnr = FN / (FN + TP + self.epsilon)
            tpr = TP / (TP + FN + self.epsilon)

            FPR.append(fpr)
            FNR.append(fnr)
            TPR.append(tpr)

        return np.array(FPR), np.array(FNR), np.array(TPR)

    # ---------------------------------------------------------
    # Plotting — AUTOSAVE, NON-BLOCKING VERSION
    # ---------------------------------------------------------

    def plot_score_distribution(self):
        plt.figure(figsize=(10, 6))

        plt.hist(
            self.genuine_scores,
            bins=50,
            histtype="stepfilled",
            alpha=0.5,
            label="Genuine Scores",
            edgecolor="green",
        )
        plt.hist(
            self.impostor_scores,
            bins=50,
            histtype="stepfilled",
            alpha=0.5,
            label="Impostor Scores",
            edgecolor="red",
        )

        plt.xlabel("Similarity Score")
        plt.ylabel("Frequency")
        plt.title(f"Score Distribution - System {self.plot_title}\nd-prime = {self.get_dprime():.3f}")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.savefig(f"score_distribution_system_{self.plot_title}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_det_curve(self, FPR, FNR):
        plt.figure(figsize=(8, 8))

        plt.plot(FPR, FNR, lw=2)
        plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
        eer, _ = self.get_EER(FPR, FNR)
        plt.scatter([eer], [eer], s=70, color="black")

        plt.xlabel("False Positive Rate")
        plt.ylabel("False Negative Rate")
        plt.title(f"DET Curve - System {self.plot_title}")
        plt.grid(alpha=0.3)

        plt.savefig(f"det_curve_system_{self.plot_title}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_roc_curve(self, FPR, TPR):
        plt.figure(figsize=(8, 8))

        plt.plot(FPR, TPR, lw=2)
        plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - System {self.plot_title}")
        plt.grid(alpha=0.3)

        plt.savefig(f"roc_curve_system_{self.plot_title}.png", dpi=300, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------
# Demo code (not used in project_experiments.py)
# ---------------------------------------------------------

def main():
    print("This main() is only for debug testing. Project uses Evaluator via import.")


if __name__ == "__main__":
    main()
