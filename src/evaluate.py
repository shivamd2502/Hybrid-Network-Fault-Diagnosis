"""
evaluate.py - reproducible evaluation of the RF pipeline, the rule engine and the hybrid system.

Usage (from the project root, with .venv activated):
    python src/evaluate.py            # uses models/pipeline.pkl (run train.py first)

Writes
    results/metrics.json                 all numbers (also read by the Streamlit app)
    results/evaluation_report.md         human-readable tables
    results/figures/confusion_rf.png, confusion_hybrid.png, confusion_rules.png

What is measured
    1. Test-set accuracy / macro-F1 of RF, rules only and the full hybrid
       (same stratified 80/20 split, seed 42, as train.py)
    2. 5-fold stratified CV of the RF pipeline on the training set
       (+ the original script's CV-on-test-set number, for reference)
    3. Ablations: Decision Tree, RF structured-only, TF-IDF text-only
    4. Text-template diagnostics: how many test sentences also occur in training,
       and RF accuracy when the symptom text is shuffled between rows
    5. Uncertainty: mean uncertainty for correct vs wrong predictions, ECE
    6. 15 hand-written scenarios (tests/scenarios.json) with unseen wording
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CATEGORICAL_FEATURES, DATASET_CSV, FIGURES_DIR, METRICS_JSON, NUMERIC_FEATURES,
    RANDOM_STATE, RESULTS_DIR, SCENARIOS_JSON, TEXT_FEATURE, silence_joblib_warning,
)
from hybrid_system import HybridDiagnosticSystem  # noqa: E402
from train import build_pipeline, load_dataset, preprocess, split  # noqa: E402

silence_joblib_warning()
STRUCT = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def scores(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "errors": int((np.asarray(y_true) != np.asarray(y_pred)).sum()),
            "n": int(len(y_true))}


def plot_cm(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.imshow(cm, cmap="Blues")
    short = [l.replace("Network Adapter Issue", "Adapter").replace("Gateway Unreachable", "Gateway")
             .replace("Subnet Mismatch", "Subnet").replace(" Issue", "").replace(" Failure", "") for l in labels]
    ax.set_xticks(range(len(labels)), short, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), short)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def expected_calibration_error(conf, correct, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def rf_factory():
    return RandomForestClassifier(n_estimators=200, min_samples_split=5, min_samples_leaf=2,
                                  class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATASET_CSV)
    X, y, _ = preprocess(df)
    X_train, X_test, y_train, y_test = split(X, y)

    system = HybridDiagnosticSystem()
    if system.pipeline is None:
        sys.exit("models/pipeline.pkl not found - run python src/train.py first.")
    pipe = system.pipeline
    labels = list(pipe.classes_)
    m = {"dataset": {"n": len(df), "n_train": len(X_train), "n_test": len(X_test),
                     "per_class": {k: int(v) for k, v in y.value_counts().items()}}}

    # ---- 1. Test set: RF, rules, hybrid -----------------------------------
    print("[1/6] Test-set evaluation ...")
    rf_pred = pipe.predict(X_test)
    rows = X_test.to_dict("records")
    rule_pred, hyb_pred, strategies, hyb_conf, rf_unc = [], [], [], [], []
    for r in rows:
        feats = {k: v for k, v in r.items() if k != TEXT_FEATURE}
        res = system.diagnose(r[TEXT_FEATURE], feats, return_explanation=False)
        rule_pred.append(res["rule_prediction"]["diagnosis"])
        hyb_pred.append(res["diagnosis"])
        strategies.append(res["fusion_strategy"])
        hyb_conf.append(res["confidence"])
        rf_unc.append(res["rf_prediction"]["uncertainty"])
    rule_pred, hyb_pred, rf_unc = np.array(rule_pred), np.array(hyb_pred), np.array(rf_unc)
    yt = y_test.values

    changed = hyb_pred != rf_pred
    m["test_set"] = {
        "rf": scores(yt, rf_pred),
        "rules": scores(yt, rule_pred),
        "hybrid": scores(yt, hyb_pred),
        "hybrid_vs_rf": {
            "changed": int(changed.sum()),
            "fixed": int((changed & (hyb_pred == yt)).sum()),
            "broken": int((changed & (rf_pred == yt) & (hyb_pred != yt)).sum()),
        },
        "fusion_strategies": {k: int(v) for k, v in pd.Series(strategies).value_counts().items()},
        "rf_errors": [{"true": t, "pred": p} for t, p in zip(yt[rf_pred != yt], rf_pred[rf_pred != yt])],
    }
    by_strategy = {}
    for s in sorted(set(strategies)):
        mask = np.array(strategies) == s
        by_strategy[s] = {"n": int(mask.sum()), "hybrid_accuracy": float((hyb_pred[mask] == yt[mask]).mean())}
    m["test_set"]["accuracy_by_strategy"] = by_strategy
    m["rules_full_dataset_accuracy"] = float(np.mean(
        [system.rule_engine.diagnose(r)[0] == t for r, t in zip(X.to_dict("records"), y)]))

    plot_cm(yt, rf_pred, labels, f"RF pipeline - test accuracy {m['test_set']['rf']['accuracy']:.2%}",
            FIGURES_DIR / "confusion_rf.png")
    plot_cm(yt, hyb_pred, labels, f"Hybrid system - test accuracy {m['test_set']['hybrid']['accuracy']:.2%}",
            FIGURES_DIR / "confusion_hybrid.png")
    plot_cm(yt, rule_pred, labels, f"Rule engine only - test accuracy {m['test_set']['rules']['accuracy']:.2%}",
            FIGURES_DIR / "confusion_rules.png")

    # ---- 2. Cross-validation ----------------------------------------------
    print("[2/6] Cross-validation ...")
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    cv_train = cross_val_score(build_pipeline(), X_train, y_train, cv=skf)
    cv_test_only = cross_val_score(build_pipeline(), X_test, y_test, cv=5)
    m["cross_validation"] = {
        "rf_train_5fold": {"mean": float(cv_train.mean()), "std": float(cv_train.std()),
                           "folds": [float(v) for v in cv_train]},
        "original_method_cv_on_test_set_only": {
            "mean": float(cv_test_only.mean()), "std": float(cv_test_only.std()),
            "folds": [float(v) for v in cv_test_only],
            "note": "what the original training script reported; each fold trains on 192 test samples"},
    }

    # ---- 3. Ablations ------------------------------------------------------
    print("[3/6] Ablations ...")
    dt = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2,
                                random_state=RANDOM_STATE).fit(X_train[STRUCT], y_train)
    rf_struct = rf_factory().fit(X_train[STRUCT], y_train)
    text_only = Pipeline([("tfidf", TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=2)),
                          ("clf", rf_factory())]).fit(X_train[TEXT_FEATURE], y_train)
    m["ablation_test_accuracy"] = {
        "decision_tree_structured_depth5": float(accuracy_score(yt, dt.predict(X_test[STRUCT]))),
        "rf_structured_only": float(accuracy_score(yt, rf_struct.predict(X_test[STRUCT]))),
        "tfidf_text_only": float(accuracy_score(yt, text_only.predict(X_test[TEXT_FEATURE]))),
        "rule_engine_only": m["test_set"]["rules"]["accuracy"],
        "rf_tfidf_plus_structured": m["test_set"]["rf"]["accuracy"],
        "hybrid_full": m["test_set"]["hybrid"]["accuracy"],
    }

    # ---- 4. Text-template diagnostics --------------------------------------
    print("[4/6] Text-template diagnostics ...")
    shuffled = X_test.copy()
    shuffled[TEXT_FEATURE] = np.random.default_rng(RANDOM_STATE).permutation(shuffled[TEXT_FEATURE].values)
    shared = df.groupby(TEXT_FEATURE)["diagnosis"].nunique()
    m["text_diagnostics"] = {
        "unique_sentences_per_class": {k: int(v) for k, v in df.groupby("diagnosis")[TEXT_FEATURE].nunique().items()},
        "test_sentences_seen_in_training": float(X_test[TEXT_FEATURE].isin(set(X_train[TEXT_FEATURE])).mean()),
        "sentences_shared_by_two_classes": list(shared[shared > 1].index),
        "rf_accuracy_with_shuffled_text": float(accuracy_score(yt, pipe.predict(shuffled))),
    }

    # ---- 5. Uncertainty ----------------------------------------------------
    print("[5/6] Uncertainty ...")
    rf_conf = pipe.predict_proba(X_test).max(axis=1)
    rf_ok = (rf_pred == yt).astype(float)
    hyb_ok = (hyb_pred == yt).astype(float)
    m["uncertainty"] = {
        "rf_uncertainty_mean_correct": float(rf_unc[rf_ok == 1].mean()),
        "rf_uncertainty_mean_wrong": float(rf_unc[rf_ok == 0].mean()) if (rf_ok == 0).any() else None,
        "share_flagged_u_gt_0.30": float((rf_unc > 0.30).mean()),
        "rf_ece": expected_calibration_error(rf_conf, rf_ok),
        "hybrid_ece": expected_calibration_error(np.array(hyb_conf), hyb_ok),
    }

    # ---- 6. Hand-written scenarios -----------------------------------------
    print("[6/6] Hand-written scenarios ...")
    spec = json.loads(Path(SCENARIOS_JSON).read_text(encoding="utf-8"))
    nt, os_ = system.encode_categoricals(spec["defaults"]["network_type"], spec["defaults"]["os_type"])
    sc_rows = []
    for s in spec["scenarios"]:
        feats = dict(s["features"], network_type=nt, os_type=os_)
        res = system.diagnose(s["text"], feats, return_explanation=False)
        sc_rows.append({
            "id": s["id"], "name": s["name"], "ambiguous": s["ambiguous"], "expected": s["expected"],
            "rf": res["rf_prediction"]["diagnosis"], "rules": res["rule_prediction"]["diagnosis"],
            "hybrid": res["diagnosis"], "strategy": res["fusion_strategy"],
            "rf_uncertainty": round(float(res["rf_prediction"]["uncertainty"]), 3),
            "hybrid_confidence": round(float(res["confidence"]), 3),
        })
    lab = [r for r in sc_rows if not r["ambiguous"]]
    m["scenarios"] = {
        "n_labelled": len(lab),
        "rf_correct": sum(r["rf"] in r["expected"] for r in lab),
        "rules_correct": sum(r["rules"] in r["expected"] for r in lab),
        "hybrid_correct": sum(r["hybrid"] in r["expected"] for r in lab),
        "n_ambiguous": len(sc_rows) - len(lab),
        "hybrid_plausible_on_ambiguous": sum(r["hybrid"] in r["expected"] for r in sc_rows if r["ambiguous"]),
        "rows": sc_rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(m, indent=2), encoding="utf-8")
    (RESULTS_DIR / "evaluation_report.md").write_text(render_report(m), encoding="utf-8")
    print_summary(m)
    print(f"\nSaved {METRICS_JSON}, results/evaluation_report.md and figures in {FIGURES_DIR}")


def pct(v):
    return f"{100 * v:.2f}%"


def render_report(m):
    t, cv, ab, td, u, sc = (m["test_set"], m["cross_validation"], m["ablation_test_accuracy"],
                            m["text_diagnostics"], m["uncertainty"], m["scenarios"])
    lines = [
        "# Evaluation report (auto-generated by `python src/evaluate.py`)", "",
        f"Dataset: {m['dataset']['n']} synthetic samples, train {m['dataset']['n_train']} / test {m['dataset']['n_test']} "
        "(stratified, seed 42).", "",
        "## Test set", "",
        "| System | Accuracy | Macro-F1 | Errors |", "|---|---:|---:|---:|",
        *[f"| {name} | {pct(t[k]['accuracy'])} | {t[k]['macro_f1']:.4f} | {t[k]['errors']} / {t[k]['n']} |"
          for name, k in (("RF pipeline (TF-IDF + structured)", "rf"), ("Rule engine only", "rules"),
                          ("Hybrid (RF + rules + fusion)", "hybrid"))],
        "",
        f"Hybrid vs RF: {t['hybrid_vs_rf']['changed']} predictions changed, "
        f"{t['hybrid_vs_rf']['fixed']} fixed, {t['hybrid_vs_rf']['broken']} broken.", "",
        "| Fusion strategy | Samples | Hybrid accuracy |", "|---|---:|---:|",
        *[f"| {k} | {v['n']} | {pct(v['hybrid_accuracy'])} |" for k, v in t["accuracy_by_strategy"].items()],
        "",
        "## Cross-validation (RF pipeline)", "",
        f"- 5-fold stratified CV on the training set: **{pct(cv['rf_train_5fold']['mean'])} +/- "
        f"{pct(cv['rf_train_5fold']['std'])}**",
        f"- Original script (CV on the 240-sample test set only): {pct(cv['original_method_cv_on_test_set_only']['mean'])}"
        f" +/- {pct(cv['original_method_cv_on_test_set_only']['std'])}", "",
        "## Ablation (test accuracy)", "", "| Variant | Accuracy |", "|---|---:|",
        *[f"| {k} | {pct(v)} |" for k, v in ab.items()], "",
        "## Text-template diagnostics", "",
        f"- Unique symptom sentences per class: {sorted(set(td['unique_sentences_per_class'].values()))}",
        f"- Test sentences that also appear in training: {pct(td['test_sentences_seen_in_training'])}",
        f"- Sentences used by two classes: {td['sentences_shared_by_two_classes']}",
        f"- RF accuracy when symptom_text is shuffled between test rows: {pct(td['rf_accuracy_with_shuffled_text'])}", "",
        "## Uncertainty", "",
        f"- Mean RF uncertainty, correct predictions: {u['rf_uncertainty_mean_correct']:.3f}",
        f"- Mean RF uncertainty, wrong predictions: {u['rf_uncertainty_mean_wrong']}",
        f"- Test samples flagged (u > 0.30): {pct(u['share_flagged_u_gt_0.30'])}",
        f"- Expected calibration error: RF {u['rf_ece']:.4f}, hybrid {u['hybrid_ece']:.4f}", "",
        "## Hand-written scenarios (unseen wording)", "",
        f"Labelled scenarios correct: RF {sc['rf_correct']}/{sc['n_labelled']}, rules {sc['rules_correct']}/{sc['n_labelled']}, "
        f"hybrid **{sc['hybrid_correct']}/{sc['n_labelled']}**. Ambiguous scenarios with a plausible hybrid answer: "
        f"{sc['hybrid_plausible_on_ambiguous']}/{sc['n_ambiguous']}.", "",
        "| # | Scenario | Expected | RF | Rules | Hybrid | Strategy | RF u |", "|---:|---|---|---|---|---|---|---:|",
        *[f"| {r['id']} | {r['name']}{' (ambiguous)' if r['ambiguous'] else ''} | {' / '.join(r['expected'])} | "
          f"{r['rf']} | {r['rules']} | {r['hybrid']} | {r['strategy']} | {r['rf_uncertainty']:.2f} |" for r in sc["rows"]],
        "",
    ]
    return "\n".join(lines)


def print_summary(m):
    t = m["test_set"]
    print("\n=== SUMMARY ===")
    for k in ("rf", "rules", "hybrid"):
        print(f"  {k:7s} test accuracy {pct(t[k]['accuracy'])}  ({t[k]['errors']} errors)")
    print(f"  hybrid vs rf: {t['hybrid_vs_rf']}")
    print(f"  strategies  : {t['fusion_strategies']}")
    cv = m["cross_validation"]
    print(f"  RF CV train : {pct(cv['rf_train_5fold']['mean'])} +/- {pct(cv['rf_train_5fold']['std'])}")
    print(f"  CV (orig)   : {pct(cv['original_method_cv_on_test_set_only']['mean'])} +/- "
          f"{pct(cv['original_method_cv_on_test_set_only']['std'])}")
    print("  ablation    :", {k: pct(v) for k, v in m["ablation_test_accuracy"].items()})
    print("  text diag   :", {k: v for k, v in m["text_diagnostics"].items() if k != "unique_sentences_per_class"})
    print("  uncertainty :", m["uncertainty"])
    sc = m["scenarios"]
    print(f"  scenarios   : RF {sc['rf_correct']}/{sc['n_labelled']} | rules {sc['rules_correct']}/{sc['n_labelled']} | "
          f"hybrid {sc['hybrid_correct']}/{sc['n_labelled']} | ambiguous plausible {sc['hybrid_plausible_on_ambiguous']}/{sc['n_ambiguous']}")
    for r in sc["rows"]:
        mark = "ok " if r["hybrid"] in r["expected"] else "BAD"
        print(f"    [{mark}] #{r['id']:2d} {r['name'][:34]:34s} exp={'/'.join(r['expected'])[:30]:30s} "
              f"rf={r['rf'][:18]:18s} rules={r['rules'][:18]:18s} hyb={r['hybrid'][:18]:18s} {r['strategy']}")


if __name__ == "__main__":
    main()
