# Methodology

A formal, equation-level specification is in
[Hybrid_Diagnostic_System_Math_Architecture.pdf](Hybrid_Diagnostic_System_Math_Architecture.pdf).
This page summarises it and links each part to the code.

## 1. Synthetic dataset ([`src/dataset.py`](../src/dataset.py))

Each fault class has a **fingerprint**:

| Fault class | ping_gw | has_ip | ping_ip | ping_dom | ip_conf | arp_ok | subnet_gw | DNS ms | loss % | hops |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|---|---|
| Router Issue | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 800–5000 | 60–100 | 1–2 |
| DNS Issue | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 50–400 | 0–5 | 8–20 |
| DNS Timeout | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 8000–30000 | 0–8 | 8–20 |
| IP Conflict | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 500–3000 | 20–70 | 1–3 |
| DHCP Failure | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 500–3000 | 80–100 | 0–1 |
| Gateway Unreachable | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 800–4000 | 90–100 | 1 |
| Network Adapter | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0–100 | 100 | 0 |
| Subnet Mismatch | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 500–3000 | 95–100 | 0–1 |

- Continuous metrics are drawn uniformly from the class range.
- Six binary flags are flipped independently with probability 0.02 (`ip_conflict` is never flipped).
- `symptom_text` is drawn uniformly from a bank of 20 sentences per class.
- `network_type` (WiFi/Ethernet) and `os_type` (Windows/macOS/Linux) are uniform and
  independent of the label.
- 150 samples per class (1,200 total), shuffled, seed 42.

Router Issue and Gateway Unreachable share the same flags and overlapping metric ranges
(loss 90–100 %, 1 hop), so the symptom text is often the only thing that separates them.

## 2. Random Forest pipeline ([`src/train.py`](../src/train.py))

```
ColumnTransformer
  ├─ TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=2)  -> symptom_text
  └─ passthrough                                                      -> 7 flags + 3 metrics + 2 encoded categoricals
RandomForestClassifier(n_estimators=200, min_samples_split=5, min_samples_leaf=2,
                       class_weight="balanced", random_state=42)
```

This gives a 112-dimensional feature vector. The split is a stratified 80/20 with seed 42,
and the reported CV is a 5-fold stratified CV on the training set.

## 3. RFC rule engine ([`src/rule_engine.py`](../src/rule_engine.py))

The rules form a decision list: the first rule that matches wins.

| # | Condition | Diagnosis | Confidence | Standard |
|---|---|---|---:|---|
| 1 | has_ip = 0 ∧ arp_ok = 1 | DHCP Failure | 0.95 | RFC 2131 |
| 2 | has_ip = 0 ∧ arp_ok = 0 ∧ loss ≥ 95 | Network Adapter Issue | 0.98 | RFC 1122 |
| 3 | ip_conflict = 1 | IP Conflict | 0.90 | RFC 826 |
| 4 | has_ip = 1 ∧ subnet_ok = 0 ∧ ping_gw = 0 | Subnet Mismatch | 0.88 | RFC 1122 |
| 5 | ping_gw = 0 ∧ has_ip = 1 ∧ subnet_ok = 1 ∧ loss ≥ 90 ∧ hops ≤ 1 | Gateway Unreachable | 0.85 | RFC 1122 |
| 6 | ping_ip = 1 ∧ ping_domain = 0 ∧ dns > 8000 ms | DNS Timeout | 0.82 | RFC 1035 |
| 7 | ping_ip = 1 ∧ ping_domain = 0 ∧ dns < 8000 ms | DNS Issue | 0.80 | RFC 1035 |
| 8 | ping_gw = 0 ∧ ping_ip = 0 ∧ has_ip = 1 ∧ loss ≥ 60 | Router Issue | 0.78 | RFC 1122 |
| – | none matched | heuristic fallback | 0.40–0.60 | – |

To turn the single rule output into a full distribution, the winning class gets its
confidence. The remaining `1 − confidence` is shared among the other classes in proportion
to hand-set partial-match scores, and the result is normalised.

## 4. Uncertainty ([`src/rf_uncertainty.py`](../src/rf_uncertainty.py))

For the forest's mean class probabilities p̄:

- predictive entropy `H = −Σ p̄(c) ln p̄(c)`, normalised to `u = H / ln 8 ∈ [0, 1]`
- inter-tree variance, averaged over classes
- mutual information `MI = H[p̄] − mean_t H[p_t]` (epistemic part, BALD)

Only `u` is used by the fusion. The other two are shown for explanation.

## 5. Bayesian fusion ([`src/bayesian_fusion.py`](../src/bayesian_fusion.py))

The rule uncertainty is `u_rule = 1 − max P_rule`. The confidence threshold is τ = 0.30.

| Regime | Trigger | Output |
|---|---|---|
| consensus | RF and rules predict the same class | `min(1, 1.05 · (w_rf P_rf + w_rule P_rule))`, w ∝ 1 − u |
| rf_override / rule_override | disagree, exactly one has u < τ | the confident model's prediction |
| bayesian_fusion | disagree, both or neither confident | `P(c) ∝ prior(c) · [w_rf P_rf(c) + w_rule P_rule(c)]`, w ∝ (1 − u) · reliability |

The prior is uniform (1/8). Reliabilities are set in [`src/hybrid_system.py`](../src/hybrid_system.py):
RF 0.996, rules 0.880. These values were not fitted on a validation set; see
[RESULTS.md](RESULTS.md#4-limitations).

## 6. Web application ([`src/app.py`](../src/app.py))

1. **Describe the problem** in free text. A keyword scan pre-fills likely answers.
2. **Answer the questionnaire:** 7 yes/no checks, 3 sliders, connection type and OS.
3. **See the result:** diagnosis, confidence, severity, uncertainty, fusion strategy, rule
   trace, a probability chart for all 8 classes, 8 remediation steps and a JSON report download.
   A History tab keeps the diagnoses from the current session.
