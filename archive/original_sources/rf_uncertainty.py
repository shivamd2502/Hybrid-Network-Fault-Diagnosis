"""
RF UNCERTAINTY ESTIMATOR (replaces mc_dropout_uncertainty.py)
==============================================================
Uses predictive entropy from sklearn RandomForest's predict_proba
to estimate uncertainty. Semantically equivalent to MC Dropout
but designed for ensemble tree models.

Key insight: A Random Forest IS already an ensemble of T=200 trees.
Each tree votes → variance across tree votes = epistemic uncertainty.
We expose this via:
  1. Predictive entropy  H = -Σ p log p  (from mean class proba)
  2. Inter-tree variance (from individual tree estimators)
  3. Mutual information (epistemic uncertainty only)

This is MORE principled than MC Dropout for tree models and is
cited in the literature (e.g. Breiman 2001, Lakshminarayanan 2017).
"""

import numpy as np
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFUncertaintyEstimator:
    """
    Uncertainty quantification for sklearn RandomForest pipelines.

    Replaces MCDropoutUncertainty for tree-based models.
    The API surface is identical so BayesianModelFusion is unaffected.
    """

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        pipeline: Pipeline,
        X,
        return_details: bool = False,
    ) -> Tuple[float, Dict]:
        """
        Estimate uncertainty for a single sample or small batch.

        Args:
            pipeline : Trained sklearn Pipeline (ColumnTransformer + RF)
            X        : pandas DataFrame (one row) — matches training format
            return_details : include breakdown dict

        Returns:
            (uncertainty_score ∈ [0,1], details_dict)
        """
        # 1. Mean class probabilities from the RF
        mean_probs = pipeline.predict_proba(X)[0]          # (n_classes,)

        # 2. Per-tree predictions — access the RF step inside the Pipeline
        tree_probs = self._per_tree_proba(pipeline, X)     # (n_trees, n_classes)

        # 3. Metrics
        entropy        = self._entropy(mean_probs)
        inter_variance = float(tree_probs.var(axis=0).mean())

        # Mutual information (epistemic): H[E[p]] - E[H[p]]
        expected_tree_entropy = float(
            np.mean([self._entropy(tp) for tp in tree_probs])
        )
        mutual_info = max(0.0, entropy - expected_tree_entropy)

        # Normalise entropy to [0, 1]
        n_classes      = len(mean_probs)
        max_entropy    = -np.log(1.0 / n_classes)
        uncertainty    = float(np.clip(entropy / max_entropy, 0.0, 1.0))

        details = {
            "entropy":              float(entropy),
            "max_entropy":          float(max_entropy),
            "inter_tree_variance":  inter_variance,
            "mutual_information":   float(mutual_info),
            "mean_probs":           mean_probs.tolist(),
            "std_probs":            tree_probs.std(axis=0).tolist(),
            "n_trees":              len(tree_probs),
        }

        return (uncertainty, details) if return_details else uncertainty

    def estimate_batch(self, pipeline: Pipeline, X_batch) -> np.ndarray:
        """Vectorised uncertainty for a batch. Returns array of shape (n,)."""
        mean_probs_batch = pipeline.predict_proba(X_batch)   # (n, n_classes)
        n_classes        = mean_probs_batch.shape[1]
        max_entropy      = -np.log(1.0 / n_classes)

        uncertainties = []
        for probs in mean_probs_batch:
            h = self._entropy(probs)
            uncertainties.append(np.clip(h / max_entropy, 0.0, 1.0))
        return np.array(uncertainties)

    def calibration_data(
        self,
        pipeline: Pipeline,
        X_test,
        y_test: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reliability diagram data — (bin_confidences, bin_accuracies).
        A perfectly calibrated model produces the diagonal y=x.
        """
        probs_batch   = pipeline.predict_proba(X_test)
        pred_classes  = probs_batch.argmax(axis=1)
        confidences   = probs_batch.max(axis=1)

        classes = pipeline.classes_
        if hasattr(y_test, "values"):
            y_test = y_test.values
        y_enc = np.array([list(classes).index(y) for y in y_test])
        correct = (pred_classes == y_enc).astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_confs, bin_accs = [], []

        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_confs.append(confidences[mask].mean())
                bin_accs.append(correct[mask].mean())

        return np.array(bin_confs), np.array(bin_accs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _entropy(self, probs: np.ndarray) -> float:
        p = np.clip(probs, self.epsilon, 1.0)
        return float(-np.sum(p * np.log(p)))

    def _per_tree_proba(self, pipeline: Pipeline, X) -> np.ndarray:
        """
        Extract per-tree class probability estimates from the RandomForest.
        The pipeline transforms X first (ColumnTransformer), then we
        query each tree individually.
        """
        # Step 1: transform through the ColumnTransformer only
        ct = pipeline.named_steps["features"]
        X_transformed = ct.transform(X)

        # Step 2: query each decision tree in the RF
        rf = pipeline.named_steps["clf"]
        tree_probs = np.array([
            tree.predict_proba(X_transformed)[0]
            for tree in rf.estimators_
        ])  # shape: (n_trees, n_classes)

        return tree_probs


# ──────────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────────

def _test():
    """Quick smoke test — run after training."""
    import pickle, pandas as pd
    from pathlib import Path

    pipeline_path = Path("models/pipeline.pkl")
    if not pipeline_path.exists():
        print("models/pipeline.pkl not found — train first.")
        return

    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    # Minimal single-row DataFrame (same schema as training)
    sample = pd.DataFrame([{
        "symptom_text":          "dns lookup times out every time",
        "ping_gateway":          1,
        "has_ip":                1,
        "ping_ip":               1,
        "ping_domain":           0,
        "ip_conflict":           0,
        "arp_table_ok":          1,
        "subnet_matches_gw":     1,
        "dns_response_time_ms":  15000,
        "packet_loss_pct":       3,
        "traceroute_hops":       10,
        "network_type":          0,   # encoded
        "os_type":               2,
    }])

    est = RFUncertaintyEstimator()
    unc, details = est.estimate(pipeline, sample, return_details=True)

    print(f"Uncertainty : {unc:.4f}")
    print(f"Entropy     : {details['entropy']:.4f}")
    print(f"MI          : {details['mutual_information']:.4f}")
    print(f"Inter-tree σ: {details['inter_tree_variance']:.6f}")
    print(f"Mean probs  : {[round(p,3) for p in details['mean_probs']]}")


if __name__ == "__main__":
    _test()