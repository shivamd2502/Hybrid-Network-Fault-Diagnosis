"""End-to-end checks: dataset reproducibility, model loading and hybrid diagnosis."""
import hashlib

import pytest

from config import DATASET_CSV, DIAGNOSES, PIPELINE_PKL
from dataset import create_dataset


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generator_reproduces_committed_dataset(tmp_path):
    out = tmp_path / "regen.csv"
    create_dataset(1200, out)
    assert sha256(out) == sha256(DATASET_CSV)


@pytest.fixture(scope="module")
def system():
    if not PIPELINE_PKL.exists():
        pytest.skip("models/pipeline.pkl missing - run python src/train.py")
    from hybrid_system import HybridDiagnosticSystem
    return HybridDiagnosticSystem()


def test_hybrid_diagnoses_dns_timeout(system):
    nt, os_ = system.encode_categoricals("WiFi", "Windows")
    res = system.diagnose(
        "dns query is timing out every time",
        dict(ping_gateway=1, has_ip=1, ping_ip=1, ping_domain=0, ip_conflict=0, arp_table_ok=1,
             subnet_matches_gw=1, dns_response_time_ms=15000, packet_loss_pct=2, traceroute_hops=10,
             network_type=nt, os_type=os_),
    )
    assert res["diagnosis"] == "DNS Timeout"
    assert res["fusion_strategy"] == "consensus"
    assert 0.0 <= res["uncertainty"] <= 1.0
    assert set(res["all_probabilities"]) <= set(DIAGNOSES)


def test_rules_only_fallback_without_model(tmp_path):
    from hybrid_system import HybridDiagnosticSystem
    res = HybridDiagnosticSystem(models_dir=tmp_path).diagnose(
        "duplicate ip address detected on network",
        dict(ping_gateway=0, has_ip=1, ip_conflict=1, arp_table_ok=0, packet_loss_pct=40),
    )
    assert res["fusion_strategy"] == "rule_only"
    assert res["diagnosis"] == "IP Conflict"
