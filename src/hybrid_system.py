"""
hybrid_system.py - Hybrid Uncertainty-Aware Network Diagnostic System (HUANDS).

Combines
  1. TF-IDF + Random Forest pipeline   (models/pipeline.pkl)   - pattern learning
  2. RFC-grounded rule engine          (rule_engine.py)        - symbolic reasoning
  3. RF entropy uncertainty estimator  (rf_uncertainty.py)     - uncertainty
  4. Bayesian model fusion             (bayesian_fusion.py)    - conflict resolution

    symptom_text + structured features
               |
     +---------+----------+
     |                    |
  RF pipeline        RFC rule engine
  P_rf(c), u_rf      P_rule(c), u_rule = 1 - max P_rule
     |                    |
     +---------+----------+
               v
     Bayesian model fusion  (consensus / override / weighted posterior)
               v
     diagnosis + confidence + uncertainty + strategy + rule trace

Usage:
    python src/hybrid_system.py          # smoke test on three hand-written cases

    from hybrid_system import HybridDiagnosticSystem
    system = HybridDiagnosticSystem()
    result = system.diagnose("nslookup keeps timing out", features_dict)
"""
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bayesian_fusion import BayesianModelFusion  # noqa: E402
from config import (  # noqa: E402
    CATEGORICAL_FEATURES, DIAGNOSES, MODELS_DIR, NUMERIC_FEATURES, TEXT_FEATURE,
    silence_joblib_warning,
)
from rf_uncertainty import RFUncertaintyEstimator  # noqa: E402
from rule_engine import NetworkDiagnosticRules  # noqa: E402

silence_joblib_warning()
logger = logging.getLogger(__name__)

# Global reliabilities used by the bayesian_fusion regime. These are the values
# used in the paper (RF test accuracy and an assumed rule-engine precision).
# The measured rule-engine accuracy is reported in results/metrics.json.
RF_RELIABILITY = 0.996
RULE_RELIABILITY = 0.880


class HybridDiagnosticSystem:
    def __init__(self, models_dir: Optional[Path] = None,
                 rf_reliability: float = RF_RELIABILITY,
                 rule_reliability: float = RULE_RELIABILITY):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.diagnoses = list(DIAGNOSES)
        self.pipeline = None
        self.encoders = None
        self._load_models()

        self.rule_engine = NetworkDiagnosticRules()
        self.uncertainty_estimator = RFUncertaintyEstimator()
        self.fusion = BayesianModelFusion(self.diagnoses)
        self.fusion.update_model_reliability(rf_reliability, rule_reliability)

    # ------------------------------------------------------------------
    def _load_models(self):
        pipeline_path = self.models_dir / "pipeline.pkl"
        encoders_path = self.models_dir / "encoders.pkl"
        if pipeline_path.exists():
            with open(pipeline_path, "rb") as f:
                self.pipeline = pickle.load(f)
        else:
            logger.warning(f"{pipeline_path} not found - RF arm disabled (rules only)")
        if encoders_path.exists():
            with open(encoders_path, "rb") as f:
                self.encoders = pickle.load(f)

    def encode_categoricals(self, network_type: str, os_type: str) -> Tuple[int, int]:
        """Label-encode 'WiFi'/'Ethernet' and 'Windows'/'macOS'/'Linux' like the training data."""
        enc = self.encoders or {}
        out = []
        for col, val in (("network_type", network_type), ("os_type", os_type)):
            le = enc.get(col)
            out.append(int(le.transform([val])[0]) if le is not None and val in le.classes_ else 0)
        return out[0], out[1]

    @staticmethod
    def _build_df(symptom_text: str, features: Dict) -> pd.DataFrame:
        """Single-row DataFrame with the exact schema expected by the pipeline."""
        row = {TEXT_FEATURE: symptom_text}
        for col in NUMERIC_FEATURES:
            row[col] = float(features.get(col, 0))
        for col in CATEGORICAL_FEATURES:
            row[col] = features.get(col, 0)   # already label-encoded integers
        return pd.DataFrame([row])

    # ------------------------------------------------------------------
    def diagnose(self, symptom_text: str, features: Dict, return_explanation: bool = True) -> Dict:
        """Run the full hybrid pipeline.

        features: keys of NUMERIC_FEATURES + CATEGORICAL_FEATURES
                  (network_type / os_type as label-encoded integers, see encode_categoricals)
        """
        result: Dict = {
            "diagnosis": None, "confidence": 0.0, "uncertainty": 1.0,
            "rf_prediction": None, "rule_prediction": None,
            "fusion_strategy": None, "all_probabilities": {}, "explanation": None,
        }

        # 1. Rule engine (always runs)
        rule_diag, rule_conf, rule_reasons = self.rule_engine.diagnose(features)
        rule_probs = self.rule_engine.get_all_probabilities(features)
        result["rule_prediction"] = {"diagnosis": rule_diag, "confidence": rule_conf, "reasons": rule_reasons}

        # 2. RF pipeline + uncertainty + 3. fusion
        if self.pipeline is not None:
            try:
                X_df = self._build_df(symptom_text, features)
                proba = self.pipeline.predict_proba(X_df)[0]
                classes = self.pipeline.classes_
                rf_probs = {c: float(p) for c, p in zip(classes, proba)}
                rf_unc, unc_details = self.uncertainty_estimator.estimate(self.pipeline, X_df, return_details=True)
                result["rf_prediction"] = {
                    "diagnosis": classes[int(np.argmax(proba))],
                    "confidence": float(proba.max()),
                    "uncertainty": rf_unc,
                    "probabilities": rf_probs,
                    "uncertainty_details": unc_details,
                }

                final_diag, final_conf, details = self.fusion.combine(rf_probs, rule_probs, rf_unc)
                result.update(
                    diagnosis=final_diag, confidence=final_conf, uncertainty=rf_unc,
                    fusion_strategy=details.get("strategy"),
                    all_probabilities=details.get("fused_probs") or rf_probs,
                )
            except Exception as exc:  # graceful fallback to rules only
                logger.error(f"RF arm failed: {exc}", exc_info=True)
                result.update(diagnosis=rule_diag, confidence=rule_conf, uncertainty=1.0 - rule_conf,
                              fusion_strategy="rule_only_fallback", all_probabilities=rule_probs)
        else:
            result.update(diagnosis=rule_diag, confidence=rule_conf, uncertainty=1.0 - rule_conf,
                          fusion_strategy="rule_only", all_probabilities=rule_probs)

        if return_explanation:
            result["explanation"] = self._explain(result)
        return result

    def batch_diagnose(self, samples: List[Dict]) -> List[Dict]:
        """samples: list of {"symptom_text": str, "features": dict}"""
        return [self.diagnose(s["symptom_text"], s["features"], return_explanation=False) for s in samples]

    # ------------------------------------------------------------------
    @staticmethod
    def _explain(result: Dict) -> str:
        strategy = result.get("fusion_strategy", "unknown")
        lines = [
            f"DIAGNOSIS : {result['diagnosis']}",
            f"Confidence: {result['confidence']:.1%}   Uncertainty: {result['uncertainty']:.1%}",
            "",
        ]
        messages = {
            "consensus": "RF pipeline and rule engine agreed.",
            "rf_override": "Models disagreed; the RF was confident, so its prediction was used.",
            "rule_override": "Models disagreed; the rule engine was confident, so its prediction was used.",
            "bayesian_fusion": "Models disagreed; resolved by Bayesian fusion.",
            "rule_only": "RF model unavailable - using rules only.",
            "rule_only_fallback": "RF arm failed - fell back to rules.",
        }
        lines.append(messages.get(strategy, f"Strategy: {strategy}"))
        rf_pred = (result.get("rf_prediction") or {}).get("diagnosis")
        rule = result.get("rule_prediction") or {}
        if strategy != "consensus" and rf_pred:
            lines.append(f"  RF predicted   : {rf_pred}")
            lines.append(f"  Rules predicted: {rule.get('diagnosis')}")
        if rule.get("reasons"):
            lines += ["", "Rule-based reasoning:"] + [f"  - {r}" for r in rule["reasons"][:3]]
        unc = (result.get("rf_prediction") or {}).get("uncertainty_details")
        if unc:
            lines += ["", "Uncertainty breakdown:",
                      f"  Entropy           : {unc['entropy']:.4f} nats",
                      f"  Mutual information: {unc['mutual_information']:.4f}",
                      f"  Inter-tree var    : {unc['inter_tree_variance']:.6f}"]
        return "\n".join(lines)

    def calibration_data(self, X_test, y_test):
        if self.pipeline is None:
            raise RuntimeError("RF pipeline not loaded")
        return self.uncertainty_estimator.calibration_data(self.pipeline, X_test, y_test)


# ---------------------------------------------------------------------------
def _test():
    system = HybridDiagnosticSystem()
    cases = [
        ("DNS Timeout", "nslookup keeps timing out, very slow dns resolution",
         dict(ping_gateway=1, has_ip=1, ping_ip=1, ping_domain=0, ip_conflict=0, arp_table_ok=1,
              subnet_matches_gw=1, dns_response_time_ms=15000, packet_loss_pct=2, traceroute_hops=10,
              network_type=1, os_type=2)),
        ("IP Conflict", "duplicate ip address warning, network keeps dropping",
         dict(ping_gateway=0, has_ip=1, ping_ip=0, ping_domain=0, ip_conflict=1, arp_table_ok=0,
              subnet_matches_gw=1, dns_response_time_ms=800, packet_loss_pct=50, traceroute_hops=1,
              network_type=0, os_type=0)),
        ("DHCP Failure", "no ip address assigned showing 169.254 apipa address",
         dict(ping_gateway=0, has_ip=0, ping_ip=0, ping_domain=0, ip_conflict=0, arp_table_ok=1,
              subnet_matches_gw=0, dns_response_time_ms=500, packet_loss_pct=95, traceroute_hops=0,
              network_type=1, os_type=0)),
    ]
    for name, text, feats in cases:
        print("-" * 70)
        print(f"Test: {name}")
        res = system.diagnose(text, feats)
        print(res["explanation"])
        top3 = sorted(res["all_probabilities"].items(), key=lambda x: -x[1])[:3]
        print("Top-3:", ", ".join(f"{d} {p:.3f}" for d, p in top3))


if __name__ == "__main__":
    _test()
