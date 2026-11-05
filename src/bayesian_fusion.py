"""
bayesian_fusion.py - uncertainty-weighted fusion of the Random Forest and the rule engine.

Three regimes:
  1. consensus        RF and rules predict the same class
                      -> certainty-weighted average of the two probabilities x 1.05 (capped at 1)
  2. override         they disagree and exactly one is confident (uncertainty < 0.30)
                      -> take the confident model's prediction ("rf_override" / "rule_override")
  3. bayesian_fusion  they disagree and both / neither are confident
                      -> posterior  P(c) ~ prior(c) * [w_rf * P_rf(c) + w_rule * P_rule(c)]
                         with w_m ~ (1 - u_m) * reliability_m

Note: the module was originally written for an LSTM arm; the arguments are now
named after the Random Forest that replaced it. The maths is unchanged.

Run `python src/bayesian_fusion.py` for a small self-test.
"""
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class BayesianModelFusion:
    """Fuses RF and rule-engine class probabilities using their uncertainties."""

    def __init__(self, diagnoses: List[str]):
        self.diagnoses = diagnoses
        self.n_classes = len(diagnoses)

        # Uniform prior; can be replaced with real incident frequencies (update_priors)
        self.priors = {d: 1.0 / self.n_classes for d in diagnoses}

        # Global reliability of each model, used only in the bayesian_fusion regime.
        # These are configured values (see hybrid_system.py), not learned parameters.
        self.model_reliability = {"rf": 0.95, "rules": 0.88}

        self.uncertainty_threshold = 0.30   # u < threshold -> model counts as confident
        self.agreement_boost = 1.05

    # ------------------------------------------------------------------
    def combine(
        self,
        rf_probs: Dict[str, float],
        rule_probs: Dict[str, float],
        rf_uncertainty: float,
        rule_uncertainty: float = None,
    ) -> Tuple[str, float, Dict]:
        """Return (final_diagnosis, final_confidence, details)."""
        if rule_uncertainty is None:
            rule_uncertainty = 1.0 - max(rule_probs.values())

        rf_pred = max(rf_probs, key=rf_probs.get)
        rule_pred = max(rule_probs, key=rule_probs.get)

        # 1. Consensus
        if rf_pred == rule_pred:
            combined = self._combine_agreement(rf_probs[rf_pred], rule_probs[rule_pred],
                                               rf_uncertainty, rule_uncertainty)
            return rf_pred, combined, {
                "strategy": "consensus",
                "rf_prob": rf_probs[rf_pred],
                "rule_prob": rule_probs[rule_pred],
                "rf_uncertainty": rf_uncertainty,
                "rule_uncertainty": rule_uncertainty,
            }

        # 2. Uncertainty-gated override
        rf_confident = rf_uncertainty < self.uncertainty_threshold
        rule_confident = rule_uncertainty < self.uncertainty_threshold

        if rf_confident and not rule_confident:
            return rf_pred, rf_probs[rf_pred], {
                "strategy": "rf_override",
                "reason": "RF confident, rules uncertain",
                "rf_pred": rf_pred, "rule_pred": rule_pred,
                "rf_uncertainty": rf_uncertainty,
                "rule_uncertainty": rule_uncertainty,
            }
        if rule_confident and not rf_confident:
            return rule_pred, rule_probs[rule_pred], {
                "strategy": "rule_override",
                "reason": "Rules confident, RF uncertain",
                "rf_pred": rf_pred, "rule_pred": rule_pred,
                "rf_uncertainty": rf_uncertainty,
                "rule_uncertainty": rule_uncertainty,
            }

        # 3. Bayesian model averaging
        fused = self._bayesian_fusion(rf_probs, rule_probs, rf_uncertainty, rule_uncertainty)
        final_pred = max(fused, key=fused.get)
        return final_pred, fused[final_pred], {
            "strategy": "bayesian_fusion",
            "rf_pred": rf_pred, "rule_pred": rule_pred,
            "fused_probs": fused,
            "rf_uncertainty": rf_uncertainty,
            "rule_uncertainty": rule_uncertainty,
        }

    # ------------------------------------------------------------------
    def _combine_agreement(self, p1: float, p2: float, u1: float, u2: float) -> float:
        """Certainty-weighted average of two agreeing probabilities, with a small boost."""
        c1, c2 = 1.0 - u1, 1.0 - u2
        total = c1 + c2
        w1, w2 = (c1 / total, c2 / total) if total > 0 else (0.5, 0.5)
        return min(1.0, (w1 * p1 + w2 * p2) * self.agreement_boost)

    def _bayesian_fusion(self, rf_probs, rule_probs, rf_unc, rule_unc) -> Dict[str, float]:
        """P(c) ~ prior(c) * [w_rf * P_rf(c) + w_rule * P_rule(c)], normalised."""
        w_rf = (1.0 - rf_unc) * self.model_reliability["rf"]
        w_rule = (1.0 - rule_unc) * self.model_reliability["rules"]
        total = w_rf + w_rule
        if total > 0:
            w_rf, w_rule = w_rf / total, w_rule / total
        else:
            w_rf = w_rule = 0.5

        fused = {}
        for d in self.diagnoses:
            likelihood = w_rf * rf_probs.get(d, 1e-10) + w_rule * rule_probs.get(d, 1e-10)
            fused[d] = self.priors.get(d, 1.0 / self.n_classes) * likelihood
        total = sum(fused.values())
        return {k: v / total for k, v in fused.items()} if total > 0 else fused

    # ------------------------------------------------------------------
    def explain_fusion(self, details: Dict) -> str:
        strategy = details.get("strategy", "unknown")
        if strategy == "consensus":
            return (f"RF and rule engine agreed. Combined confidence from "
                    f"RF={details['rf_prob']:.2f} and Rules={details['rule_prob']:.2f}.")
        if strategy == "rf_override":
            return (f"Models disagreed; the RF was confident (u={details['rf_uncertainty']:.2f}) "
                    f"and the rules were not (u={details['rule_uncertainty']:.2f}). Using the RF.")
        if strategy == "rule_override":
            return (f"Models disagreed; the rules were confident (u={details['rule_uncertainty']:.2f}) "
                    f"and the RF was not (u={details['rf_uncertainty']:.2f}). Using the rules.")
        if strategy == "bayesian_fusion":
            return (f"Models disagreed: RF predicted '{details['rf_pred']}', rules predicted "
                    f"'{details['rule_pred']}'. Resolved by uncertainty-weighted Bayesian fusion.")
        return "Unknown fusion strategy"

    def update_model_reliability(self, rf_accuracy: float, rule_accuracy: float):
        self.model_reliability["rf"] = rf_accuracy
        self.model_reliability["rules"] = rule_accuracy
        logger.info(f"Model reliability: RF={rf_accuracy:.3f}, Rules={rule_accuracy:.3f}")

    def update_priors(self, diagnosis_frequencies: Dict[str, float]):
        total = sum(diagnosis_frequencies.values())
        if total > 0:
            self.priors = {k: v / total for k, v in diagnosis_frequencies.items()}


# ---------------------------------------------------------------------------
def test_fusion():
    diagnoses = ["Router Issue", "DNS Issue", "DNS Timeout", "IP Conflict",
                 "DHCP Failure", "Gateway Unreachable", "Network Adapter Issue", "Subnet Mismatch"]
    fusion = BayesianModelFusion(diagnoses)

    def probs(top, p, rest):
        d = {k: rest for k in diagnoses}
        if top:
            d[top] = p
        return d

    cases = [
        ("Both agree on DNS Timeout", probs("DNS Timeout", 0.92, 0.01), probs("DNS Timeout", 0.85, 0.01), 0.08, 0.15),
        ("RF confident, rules uncertain", probs("IP Conflict", 0.94, 0.01), probs(None, 0, 0.125), 0.06, 0.75),
        ("Disagree, both confident", probs("Router Issue", 0.84, 0.02), probs("Gateway Unreachable", 0.82, 0.02), 0.16, 0.18),
    ]
    for name, rf_p, rule_p, u_rf, u_rule in cases:
        pred, conf, det = fusion.combine(rf_p, rule_p, u_rf, u_rule)
        print(f"[{name}] -> {pred} ({conf:.3f}) via {det['strategy']}")
        print(f"    {fusion.explain_fusion(det)}")


if __name__ == "__main__":
    test_fusion()
