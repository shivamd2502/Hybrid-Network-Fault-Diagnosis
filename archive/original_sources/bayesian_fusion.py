"""
BAYESIAN MODEL FUSION
=====================
Combines predictions from multiple models (LSTM + Rules) using Bayesian inference
with uncertainty weighting.

Key novelty: Weights models based on their uncertainty estimates, not just accuracy.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianModelFusion:
    """
    Fuses predictions from LSTM (deep learning) and rule-based engine
    using uncertainty-weighted Bayesian combination.
    
    Core idea:
      - High-confidence predictions get higher weight
      - When models disagree, uncertainty determines which to trust
      - When both uncertain, fall back to weighted average
    """
    
    def __init__(self, diagnoses: List[str]):
        """
        Args:
            diagnoses: List of all possible diagnosis classes
        """
        self.diagnoses = diagnoses
        self.n_classes = len(diagnoses)
        
        # Prior probabilities (from dataset statistics or uniform)
        # Can be updated with real-world incident frequencies
        self.priors = self._init_priors()
        
        # Model reliability estimates (learned from validation set)
        # These would be tuned during training
        self.model_reliability = {
            "lstm": 0.95,      # LSTM accuracy on val set
            "rules": 0.88,     # Rule engine precision on val set
        }
        
        # Fusion strategy parameters
        self.uncertainty_threshold = 0.30  # Below this, consider uncertain
        self.conflict_resolution = "weighted_average"  # or "max_confidence"
    
    def _init_priors(self) -> Dict[str, float]:
        """
        Initialize prior probabilities for each diagnosis.
        In production, these should come from historical incident data.
        """
        # Uniform prior (no domain knowledge)
        uniform = 1.0 / self.n_classes
        return {diag: uniform for diag in self.diagnoses}
    
    def combine(
        self,
        lstm_probs: Dict[str, float],
        rule_probs: Dict[str, float],
        lstm_uncertainty: float,
        rule_uncertainty: float = None
    ) -> Tuple[str, float, Dict]:
        """
        Fuse LSTM and rule-based predictions using Bayesian combination.
        
        Args:
            lstm_probs: Dict mapping diagnosis -> probability (from LSTM)
            rule_probs: Dict mapping diagnosis -> probability (from rules)
            lstm_uncertainty: Uncertainty estimate for LSTM (0=certain, 1=very uncertain)
            rule_uncertainty: Uncertainty for rules (if None, inferred from confidence)
        
        Returns:
            (final_diagnosis, final_confidence, fusion_details)
        """
        
        # Infer rule uncertainty if not provided
        if rule_uncertainty is None:
            rule_top_conf = max(rule_probs.values())
            rule_uncertainty = 1.0 - rule_top_conf
        
        # ──────────────────────────────────────────────────────────────────
        # CASE 1: Both models agree (consensus)
        # ──────────────────────────────────────────────────────────────────
        lstm_pred = max(lstm_probs, key=lstm_probs.get)
        rule_pred = max(rule_probs, key=rule_probs.get)
        
        if lstm_pred == rule_pred:
            # Agreement → combine probabilities
            combined_prob = self._combine_probabilities_agreement(
                lstm_probs[lstm_pred],
                rule_probs[rule_pred],
                lstm_uncertainty,
                rule_uncertainty
            )
            
            return lstm_pred, combined_prob, {
                "strategy": "consensus",
                "lstm_prob": lstm_probs[lstm_pred],
                "rule_prob": rule_probs[rule_pred],
                "lstm_uncertainty": lstm_uncertainty,
                "rule_uncertainty": rule_uncertainty,
            }
        
        # ──────────────────────────────────────────────────────────────────
        # CASE 2: Models disagree → resolve using uncertainty
        # ──────────────────────────────────────────────────────────────────
        
        # Check if either model is highly certain
        lstm_confident = lstm_uncertainty < self.uncertainty_threshold
        rule_confident = rule_uncertainty < self.uncertainty_threshold
        
        if lstm_confident and not rule_confident:
            # Trust LSTM
            return lstm_pred, lstm_probs[lstm_pred], {
                "strategy": "lstm_override",
                "reason": "LSTM confident, rules uncertain",
                "lstm_uncertainty": lstm_uncertainty,
                "rule_uncertainty": rule_uncertainty,
            }
        
        if rule_confident and not lstm_confident:
            # Trust rules
            return rule_pred, rule_probs[rule_pred], {
                "strategy": "rule_override",
                "reason": "Rules confident, LSTM uncertain",
                "lstm_uncertainty": lstm_uncertainty,
                "rule_uncertainty": rule_uncertainty,
            }
        
        # ──────────────────────────────────────────────────────────────────
        # CASE 3: Both uncertain OR both confident but disagree
        # → Use Bayesian weighted combination
        # ──────────────────────────────────────────────────────────────────
        fused_probs = self._bayesian_fusion(
            lstm_probs,
            rule_probs,
            lstm_uncertainty,
            rule_uncertainty
        )
        
        final_pred = max(fused_probs, key=fused_probs.get)
        final_conf = fused_probs[final_pred]
        
        return final_pred, final_conf, {
            "strategy": "bayesian_fusion",
            "lstm_pred": lstm_pred,
            "rule_pred": rule_pred,
            "fused_probs": fused_probs,
            "lstm_uncertainty": lstm_uncertainty,
            "rule_uncertainty": rule_uncertainty,
        }
    
    def _combine_probabilities_agreement(
        self,
        prob1: float,
        prob2: float,
        unc1: float,
        unc2: float
    ) -> float:
        """
        When models agree, combine their probabilities.
        Higher weight to more certain model.
        """
        # Certainty = 1 - uncertainty
        cert1 = 1.0 - unc1
        cert2 = 1.0 - unc2
        
        # Weighted average by certainty
        weight1 = cert1 / (cert1 + cert2)
        weight2 = cert2 / (cert1 + cert2)
        
        combined = weight1 * prob1 + weight2 * prob2
        
        # Boost: agreement increases confidence
        agreement_boost = 1.05
        combined = min(1.0, combined * agreement_boost)
        
        return combined
    
    def _bayesian_fusion(
        self,
        lstm_probs: Dict[str, float],
        rule_probs: Dict[str, float],
        lstm_unc: float,
        rule_unc: float
    ) -> Dict[str, float]:
        """
        Bayesian model averaging with uncertainty weighting.
        
        P(diagnosis | symptoms) ∝ 
            P(diagnosis) × 
            [w_lstm × P(diagnosis | LSTM) + w_rule × P(diagnosis | rules)]
        
        where w_lstm and w_rule are uncertainty-based weights.
        """
        
        # Compute model weights based on uncertainty and reliability
        cert_lstm = 1.0 - lstm_unc
        cert_rule = 1.0 - rule_unc
        
        # Weight by certainty AND model reliability
        w_lstm = cert_lstm * self.model_reliability["lstm"]
        w_rule = cert_rule * self.model_reliability["rules"]
        
        # Normalize weights
        total_weight = w_lstm + w_rule
        if total_weight > 0:
            w_lstm /= total_weight
            w_rule /= total_weight
        else:
            # Fallback to equal weights if both completely uncertain
            w_lstm = w_rule = 0.5
        
        # Bayesian combination for each diagnosis
        fused = {}
        for diag in self.diagnoses:
            # Likelihood from each model
            p_lstm = lstm_probs.get(diag, 1e-10)  # avoid zero
            p_rule = rule_probs.get(diag, 1e-10)
            
            # Prior
            prior = self.priors.get(diag, 1.0 / self.n_classes)
            
            # Weighted combination of likelihoods
            likelihood = w_lstm * p_lstm + w_rule * p_rule
            
            # Posterior (unnormalized)
            fused[diag] = prior * likelihood
        
        # Normalize to sum to 1
        total = sum(fused.values())
        if total > 0:
            fused = {k: v/total for k, v in fused.items()}
        
        return fused
    
    def explain_fusion(self, fusion_details: Dict) -> str:
        """
        Generate human-readable explanation of fusion decision.
        """
        strategy = fusion_details.get("strategy", "unknown")
        
        if strategy == "consensus":
            return (
                f"Both LSTM and rule-based models agreed on the diagnosis. "
                f"Combined confidence: LSTM={fusion_details['lstm_prob']:.2f}, "
                f"Rules={fusion_details['rule_prob']:.2f}"
            )
        
        elif strategy == "lstm_override":
            return (
                f"LSTM model was highly confident (uncertainty={fusion_details['lstm_uncertainty']:.2f}), "
                f"while rule-based model was uncertain (uncertainty={fusion_details['rule_uncertainty']:.2f}). "
                f"Trusting LSTM prediction."
            )
        
        elif strategy == "rule_override":
            return (
                f"Rule-based model was highly confident (uncertainty={fusion_details['rule_uncertainty']:.2f}), "
                f"while LSTM was uncertain (uncertainty={fusion_details['lstm_uncertainty']:.2f}). "
                f"Trusting rule-based prediction."
            )
        
        elif strategy == "bayesian_fusion":
            return (
                f"Models disagreed: LSTM predicted '{fusion_details['lstm_pred']}', "
                f"Rules predicted '{fusion_details['rule_pred']}'. "
                f"Used Bayesian fusion with uncertainty weighting to resolve conflict."
            )
        
        return "Unknown fusion strategy"
    
    def update_model_reliability(self, lstm_accuracy: float, rule_accuracy: float):
        """
        Update model reliability estimates based on validation performance.
        Should be called after training/validation.
        """
        self.model_reliability["lstm"] = lstm_accuracy
        self.model_reliability["rules"] = rule_accuracy
        logger.info(f"Updated model reliability: LSTM={lstm_accuracy:.3f}, Rules={rule_accuracy:.3f}")
    
    def update_priors(self, diagnosis_frequencies: Dict[str, float]):
        """
        Update prior probabilities based on real-world incident data.
        
        Args:
            diagnosis_frequencies: Dict mapping diagnosis -> observed frequency
        """
        total = sum(diagnosis_frequencies.values())
        if total > 0:
            self.priors = {k: v/total for k, v in diagnosis_frequencies.items()}
            logger.info("Updated priors from real-world data")


# ═══════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════

def test_fusion():
    """Test Bayesian fusion with synthetic scenarios."""
    
    diagnoses = [
        "Router Issue", "DNS Issue", "DNS Timeout", "IP Conflict",
        "DHCP Failure", "Gateway Unreachable", "Network Adapter Issue", "Subnet Mismatch"
    ]
    
    fusion = BayesianModelFusion(diagnoses)
    
    print("\n" + "="*70)
    print("TESTING BAYESIAN MODEL FUSION")
    print("="*70)
    
    # ────────────────────────────────────────────────────────────────────────
    # Test 1: Both models agree
    # ────────────────────────────────────────────────────────────────────────
    print("\n[Test 1] Both models agree on DNS Timeout")
    
    lstm_probs = {d: 0.01 for d in diagnoses}
    lstm_probs["DNS Timeout"] = 0.92
    
    rule_probs = {d: 0.01 for d in diagnoses}
    rule_probs["DNS Timeout"] = 0.85
    
    pred, conf, details = fusion.combine(
        lstm_probs, rule_probs,
        lstm_uncertainty=0.08,  # Low uncertainty
        rule_uncertainty=0.15
    )
    
    print(f"  Final prediction: {pred} (confidence: {conf:.3f})")
    print(f"  Strategy: {details['strategy']}")
    print(f"  Explanation: {fusion.explain_fusion(details)}")
    
    # ────────────────────────────────────────────────────────────────────────
    # Test 2: LSTM confident, Rules uncertain → Trust LSTM
    # ────────────────────────────────────────────────────────────────────────
    print("\n[Test 2] LSTM confident, Rules uncertain")
    
    lstm_probs = {d: 0.01 for d in diagnoses}
    lstm_probs["IP Conflict"] = 0.94
    
    rule_probs = {d: 0.125 for d in diagnoses}  # Uniform (uncertain)
    
    pred, conf, details = fusion.combine(
        lstm_probs, rule_probs,
        lstm_uncertainty=0.06,
        rule_uncertainty=0.75
    )
    
    print(f"  Final prediction: {pred} (confidence: {conf:.3f})")
    print(f"  Strategy: {details['strategy']}")
    print(f"  Explanation: {fusion.explain_fusion(details)}")
    
    # ────────────────────────────────────────────────────────────────────────
    # Test 3: Models disagree, both moderately confident → Bayesian fusion
    # ────────────────────────────────────────────────────────────────────────
    print("\n[Test 3] Models disagree (LSTM: Router, Rules: Gateway)")
    
    lstm_probs = {d: 0.02 for d in diagnoses}
    lstm_probs["Router Issue"] = 0.84
    
    rule_probs = {d: 0.02 for d in diagnoses}
    rule_probs["Gateway Unreachable"] = 0.82
    
    pred, conf, details = fusion.combine(
        lstm_probs, rule_probs,
        lstm_uncertainty=0.16,
        rule_uncertainty=0.18
    )
    
    print(f"  Final prediction: {pred} (confidence: {conf:.3f})")
    print(f"  Strategy: {details['strategy']}")
    print(f"  LSTM prediction: {details.get('lstm_pred', 'N/A')}")
    print(f"  Rule prediction: {details.get('rule_pred', 'N/A')}")
    print(f"  Explanation: {fusion.explain_fusion(details)}")


if __name__ == "__main__":
    test_fusion()