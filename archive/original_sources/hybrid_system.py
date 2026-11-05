"""
HYBRID UNCERTAINTY-AWARE NETWORK DIAGNOSTIC SYSTEM  (v2 — sklearn RF backend)
==============================================================================
Combines:
  1. Sklearn RF pipeline  (TF-IDF + RandomForest)  — pattern learning
  2. RFC rule-based engine                          — symbolic reasoning
  3. RF entropy uncertainty estimator               — uncertainty quantification
  4. Bayesian model fusion                          — conflict resolution

This file REPLACES hybrid_diagnostic_system.py.
It is compatible with the existing training pipeline that produces:
  models/pipeline.pkl   (sklearn Pipeline)
  models/encoders.pkl   (LabelEncoders for network_type / os_type)

Usage (standalone):
    python hybrid_system.py

Integration with Streamlit:
    See network_app_final.py — import HybridDiagnosticSystem and call
    hybrid_system.diagnose(symptom_text, features).
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

# Internal modules (must be in same directory)
from rule_based_engine import NetworkDiagnosticRules
from rf_uncertainty    import RFUncertaintyEstimator
from bayesian_fusion   import BayesianModelFusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Feature spec — must match training pipeline exactly
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "ping_gateway", "has_ip", "ping_ip", "ping_domain",
    "ip_conflict", "arp_table_ok", "subnet_matches_gw",
    "dns_response_time_ms", "packet_loss_pct", "traceroute_hops",
]
CATEGORICAL_FEATURES = ["network_type", "os_type"]
TEXT_FEATURE          = "symptom_text"


class HybridDiagnosticSystem:
    """
    Hybrid system: RF pipeline + RFC rules + uncertainty + Bayesian fusion.

    Architecture
    ────────────
    symptom_text + structured features
           │
    ┌──────┴──────────────────┐
    │  RF pipeline (sklearn)  │   TF-IDF on text + RF on all features
    │  → predict_proba[]      │
    └──────┬──────────────────┘
           │ proba[]
    RF Entropy Uncertainty     RFC Rule Engine
    H = −Σ p log p             8 deterministic rules
           │                          │
           └──────────┬───────────────┘
                      ▼
             Bayesian Model Fusion
             (consensus / override / weighted avg)
                      │
                      ▼
          Final diagnosis + confidence
          + uncertainty + fusion strategy
          + rule reasons (explainability)
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)

        self.diagnoses = [
            "Router Issue", "DNS Issue", "DNS Timeout", "IP Conflict",
            "DHCP Failure", "Gateway Unreachable", "Network Adapter Issue",
            "Subnet Mismatch",
        ]

        # Load sklearn pipeline
        self.pipeline = None
        self.encoders = None
        self._load_models()

        # Rule engine
        self.rule_engine = NetworkDiagnosticRules()

        # Uncertainty estimator (entropy-based, works with RF)
        self.uncertainty_estimator = RFUncertaintyEstimator()

        # Bayesian fusion — update reliability from your actual eval metrics
        self.fusion = BayesianModelFusion(self.diagnoses)
        self.fusion.update_model_reliability(
            lstm_accuracy=0.996,   # RF test accuracy
            rule_accuracy=0.880,   # rule engine precision on val set
        )

        logger.info("HybridDiagnosticSystem ready")

    # ─────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────

    def _load_models(self):
        pipeline_path = self.models_dir / "pipeline.pkl"
        encoders_path = self.models_dir / "encoders.pkl"

        if pipeline_path.exists():
            with open(pipeline_path, "rb") as f:
                self.pipeline = pickle.load(f)
            logger.info(f"Loaded RF pipeline from {pipeline_path}")
        else:
            logger.warning(f"pipeline.pkl not found — RF arm disabled")

        if encoders_path.exists():
            with open(encoders_path, "rb") as f:
                self.encoders = pickle.load(f)
            logger.info("Loaded label encoders")

    # ─────────────────────────────────────────────────────────────────────
    # Feature builder (raw dict → single-row DataFrame)
    # ─────────────────────────────────────────────────────────────────────

    def _build_df(self, symptom_text: str, features: Dict) -> pd.DataFrame:
        """
        Build the exact DataFrame schema expected by the ColumnTransformer.
        network_type and os_type must already be label-encoded integers.
        """
        row = {TEXT_FEATURE: symptom_text}
        for col in NUMERIC_FEATURES:
            row[col] = float(features.get(col, 0))
        for col in CATEGORICAL_FEATURES:
            row[col] = features.get(col, 0)   # integer (already encoded)
        return pd.DataFrame([row])

    # ─────────────────────────────────────────────────────────────────────
    # Main diagnosis function
    # ─────────────────────────────────────────────────────────────────────

    def diagnose(
        self,
        symptom_text: str,
        features: Dict,
        return_explanation: bool = True,
    ) -> Dict:
        """
        Run the full hybrid pipeline.

        Args
        ────
        symptom_text : free-text description (used by TF-IDF arm)
        features     : dict with keys matching NUMERIC_FEATURES +
                       CATEGORICAL_FEATURES (network_type / os_type
                       should be label-encoded integers as returned by
                       encode_categoricals() in the Streamlit app)
        return_explanation : include human-readable explanation string

        Returns
        ───────
        {
          diagnosis          : str
          confidence         : float  [0, 1]
          uncertainty        : float  [0, 1]   (0 = certain)
          rf_prediction      : dict   (diagnosis, confidence, proba)
          rule_prediction    : dict   (diagnosis, confidence, reasons)
          fusion_strategy    : str    (consensus / rf_override / rule_override /
                                       bayesian_fusion / rule_only)
          all_probabilities  : dict   {diagnosis: prob}
          explanation        : str    (if return_explanation=True)
        }
        """
        result: Dict = {
            "diagnosis":          None,
            "confidence":         0.0,
            "uncertainty":        1.0,
            "rf_prediction":      None,
            "rule_prediction":    None,
            "fusion_strategy":    None,
            "all_probabilities":  {},
            "explanation":        None,
        }

        # ── 1. Rule engine (always runs) ──────────────────────────────────
        rule_diag, rule_conf, rule_reasons = self.rule_engine.diagnose(features)
        rule_probs = self.rule_engine.get_all_probabilities(features)

        result["rule_prediction"] = {
            "diagnosis":  rule_diag,
            "confidence": rule_conf,
            "reasons":    rule_reasons,
        }

        # ── 2. RF pipeline ────────────────────────────────────────────────
        if self.pipeline is not None:
            try:
                X_df = self._build_df(symptom_text, features)

                # Class probabilities
                proba_arr = self.pipeline.predict_proba(X_df)[0]
                classes   = self.pipeline.classes_
                rf_probs  = {c: float(p) for c, p in zip(classes, proba_arr)}

                rf_diag   = classes[int(np.argmax(proba_arr))]
                rf_conf   = float(proba_arr.max())

                # Uncertainty (entropy-based)
                rf_unc, unc_details = self.uncertainty_estimator.estimate(
                    self.pipeline, X_df, return_details=True
                )

                result["rf_prediction"] = {
                    "diagnosis":         rf_diag,
                    "confidence":        rf_conf,
                    "uncertainty":       rf_unc,
                    "probabilities":     rf_probs,
                    "uncertainty_details": unc_details,
                }

                # ── 3. Bayesian fusion ────────────────────────────────────
                final_diag, final_conf, fusion_details = self.fusion.combine(
                    lstm_probs      = rf_probs,    # BayesianFusion expects "lstm"
                    rule_probs      = rule_probs,
                    lstm_uncertainty= rf_unc,
                )

                result["diagnosis"]         = final_diag
                result["confidence"]        = final_conf
                result["uncertainty"]       = rf_unc
                result["fusion_strategy"]   = fusion_details.get("strategy")
                result["all_probabilities"] = (
                    fusion_details.get("fused_probs") or rf_probs
                )

            except Exception as exc:
                logger.error(f"RF arm failed: {exc}", exc_info=True)
                # Graceful fallback to rules-only
                result["diagnosis"]         = rule_diag
                result["confidence"]        = rule_conf
                result["uncertainty"]       = 1.0 - rule_conf
                result["fusion_strategy"]   = "rule_only_fallback"
                result["all_probabilities"] = rule_probs
        else:
            # No RF model — use rules only
            result["diagnosis"]         = rule_diag
            result["confidence"]        = rule_conf
            result["uncertainty"]       = 1.0 - rule_conf
            result["fusion_strategy"]   = "rule_only"
            result["all_probabilities"] = rule_probs

        # ── 4. Explanation ────────────────────────────────────────────────
        if return_explanation:
            result["explanation"] = self._explain(result)

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Batch inference
    # ─────────────────────────────────────────────────────────────────────

    def batch_diagnose(self, samples: List[Dict]) -> List[Dict]:
        """
        samples: list of {"symptom_text": str, "features": dict}
        """
        return [
            self.diagnose(s["symptom_text"], s["features"], return_explanation=False)
            for s in samples
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Human-readable explanation
    # ─────────────────────────────────────────────────────────────────────

    def _explain(self, result: Dict) -> str:
        lines = []
        diag     = result["diagnosis"]
        conf     = result["confidence"]
        unc      = result["uncertainty"]
        strategy = result.get("fusion_strategy", "unknown")

        lines.append(f"DIAGNOSIS : {diag}")
        lines.append(f"Confidence: {conf:.1%}   Uncertainty: {unc:.1%}")
        lines.append("")

        strategy_msg = {
            "consensus":          "✓ RF pipeline and rule engine agreed.",
            "lstm_override":      "✓ RF pipeline was highly confident; rules overridden.",
            "rule_override":      "✓ Rule engine was highly confident; RF overridden.",
            "bayesian_fusion":    "⚠ Models disagreed — Bayesian fusion applied.",
            "rule_only":          "ℹ RF model unavailable — using rules only.",
            "rule_only_fallback": "⚠ RF arm failed — fell back to rules.",
        }
        lines.append(strategy_msg.get(strategy, f"Strategy: {strategy}"))

        if strategy == "bayesian_fusion":
            rf_pred   = (result.get("rf_prediction")   or {}).get("diagnosis", "N/A")
            rule_pred = (result.get("rule_prediction")  or {}).get("diagnosis", "N/A")
            lines.append(f"  RF predicted   : {rf_pred}")
            lines.append(f"  Rules predicted: {rule_pred}")

        lines.append("")
        rule_pred = result.get("rule_prediction") or {}
        if rule_pred.get("reasons"):
            lines.append("Rule-based reasoning:")
            for r in rule_pred["reasons"][:3]:
                lines.append(f"  • {r}")

        # Uncertainty details
        unc_details = (result.get("rf_prediction") or {}).get("uncertainty_details")
        if unc_details:
            lines.append("")
            lines.append("Uncertainty breakdown:")
            lines.append(f"  Entropy          : {unc_details['entropy']:.4f}")
            lines.append(f"  Mutual information: {unc_details['mutual_information']:.4f}")
            lines.append(f"  Inter-tree σ      : {unc_details['inter_tree_variance']:.6f}")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────
    # Calibration helper (useful for paper's experiments)
    # ─────────────────────────────────────────────────────────────────────

    def calibration_data(self, X_test, y_test) -> Tuple:
        """Reliability diagram data for the RF arm."""
        if self.pipeline is None:
            raise RuntimeError("RF pipeline not loaded")
        return self.uncertainty_estimator.calibration_data(
            self.pipeline, X_test, y_test
        )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _test():
    system = HybridDiagnosticSystem()

    test_cases = [
        {
            "name": "DNS Timeout",
            "symptom_text": "nslookup keeps timing out, very slow dns resolution",
            "features": {
                "ping_gateway": 1, "has_ip": 1, "ping_ip": 1, "ping_domain": 0,
                "ip_conflict": 0, "arp_table_ok": 1, "subnet_matches_gw": 1,
                "dns_response_time_ms": 15000, "packet_loss_pct": 2,
                "traceroute_hops": 10, "network_type": 1, "os_type": 2,
            },
        },
        {
            "name": "IP Conflict",
            "symptom_text": "duplicate ip address warning, network keeps dropping",
            "features": {
                "ping_gateway": 0, "has_ip": 1, "ping_ip": 0, "ping_domain": 0,
                "ip_conflict": 1, "arp_table_ok": 0, "subnet_matches_gw": 1,
                "dns_response_time_ms": 800, "packet_loss_pct": 50,
                "traceroute_hops": 1, "network_type": 0, "os_type": 0,
            },
        },
        {
            "name": "DHCP Failure",
            "symptom_text": "no ip address assigned showing 169.254 apipa address",
            "features": {
                "ping_gateway": 0, "has_ip": 0, "ping_ip": 0, "ping_domain": 0,
                "ip_conflict": 0, "arp_table_ok": 1, "subnet_matches_gw": 0,
                "dns_response_time_ms": 500, "packet_loss_pct": 95,
                "traceroute_hops": 0, "network_type": 1, "os_type": 0,
            },
        },
    ]

    print("\n" + "=" * 70)
    print("HYBRID SYSTEM SMOKE TEST")
    print("=" * 70)

    for tc in test_cases:
        print(f"\n{'─'*70}")
        print(f"Test: {tc['name']}")
        result = system.diagnose(tc["symptom_text"], tc["features"])
        print(result["explanation"])
        top3 = sorted(result["all_probabilities"].items(), key=lambda x: -x[1])[:3]
        print("\nTop-3 probabilities:")
        for d, p in top3:
            print(f"  {d:30s}: {p:.3f}")


if __name__ == "__main__":
    _test()