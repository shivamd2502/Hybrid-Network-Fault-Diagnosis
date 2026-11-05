"""The three Bayesian fusion regimes behave as documented."""
from bayesian_fusion import BayesianModelFusion
from config import DIAGNOSES


def probs(top=None, p=0.0, rest=0.01):
    d = {k: rest for k in DIAGNOSES}
    if top:
        d[top] = p
    return d


def test_consensus_boosts_and_caps_confidence():
    f = BayesianModelFusion(DIAGNOSES)
    pred, conf, det = f.combine(probs("DNS Timeout", 0.92), probs("DNS Timeout", 0.85), 0.08, 0.15)
    assert det["strategy"] == "consensus" and pred == "DNS Timeout"
    assert 0.85 < conf <= 1.0


def test_rf_override_when_only_rf_is_confident():
    f = BayesianModelFusion(DIAGNOSES)
    pred, _, det = f.combine(probs("IP Conflict", 0.94), probs(rest=0.125), 0.06, 0.75)
    assert det["strategy"] == "rf_override" and pred == "IP Conflict"


def test_rule_override_when_only_rules_are_confident():
    f = BayesianModelFusion(DIAGNOSES)
    pred, _, det = f.combine(probs("Router Issue", 0.5), probs("Gateway Unreachable", 0.85), 0.45, 0.15)
    assert det["strategy"] == "rule_override" and pred == "Gateway Unreachable"


def test_bayesian_fusion_returns_normalised_posterior():
    f = BayesianModelFusion(DIAGNOSES)
    _, _, det = f.combine(probs("Router Issue", 0.84, 0.02), probs("Gateway Unreachable", 0.82, 0.02), 0.16, 0.18)
    assert det["strategy"] == "bayesian_fusion"
    assert abs(sum(det["fused_probs"].values()) - 1.0) < 1e-9
