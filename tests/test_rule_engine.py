"""Each of the 8 rules fires on the noise-free fingerprint of its own fault class."""
import pytest

from rule_engine import NetworkDiagnosticRules

FINGERPRINTS = {
    "DHCP Failure": dict(ping_gateway=0, has_ip=0, ping_ip=0, ping_domain=0, ip_conflict=0,
                         arp_table_ok=1, subnet_matches_gw=0, dns_response_time_ms=1000,
                         packet_loss_pct=90, traceroute_hops=0),
    "Network Adapter Issue": dict(ping_gateway=0, has_ip=0, ping_ip=0, ping_domain=0, ip_conflict=0,
                                  arp_table_ok=0, subnet_matches_gw=0, dns_response_time_ms=50,
                                  packet_loss_pct=100, traceroute_hops=0),
    "IP Conflict": dict(ping_gateway=0, has_ip=1, ping_ip=0, ping_domain=0, ip_conflict=1,
                        arp_table_ok=0, subnet_matches_gw=1, dns_response_time_ms=1500,
                        packet_loss_pct=40, traceroute_hops=2),
    "Subnet Mismatch": dict(ping_gateway=0, has_ip=1, ping_ip=0, ping_domain=0, ip_conflict=0,
                            arp_table_ok=1, subnet_matches_gw=0, dns_response_time_ms=1500,
                            packet_loss_pct=97, traceroute_hops=0),
    "Gateway Unreachable": dict(ping_gateway=0, has_ip=1, ping_ip=0, ping_domain=0, ip_conflict=0,
                                arp_table_ok=1, subnet_matches_gw=1, dns_response_time_ms=2000,
                                packet_loss_pct=95, traceroute_hops=1),
    "DNS Timeout": dict(ping_gateway=1, has_ip=1, ping_ip=1, ping_domain=0, ip_conflict=0,
                        arp_table_ok=1, subnet_matches_gw=1, dns_response_time_ms=15000,
                        packet_loss_pct=3, traceroute_hops=12),
    "DNS Issue": dict(ping_gateway=1, has_ip=1, ping_ip=1, ping_domain=0, ip_conflict=0,
                      arp_table_ok=1, subnet_matches_gw=1, dns_response_time_ms=200,
                      packet_loss_pct=2, traceroute_hops=12),
    "Router Issue": dict(ping_gateway=0, has_ip=1, ping_ip=0, ping_domain=0, ip_conflict=0,
                         arp_table_ok=1, subnet_matches_gw=1, dns_response_time_ms=2000,
                         packet_loss_pct=70, traceroute_hops=2),
}


@pytest.mark.parametrize("expected,features", FINGERPRINTS.items())
def test_rule_fires_for_fingerprint(expected, features):
    diag, conf, reasons = NetworkDiagnosticRules().diagnose(features)
    assert diag == expected
    assert 0.7 <= conf <= 1.0
    assert reasons


@pytest.mark.parametrize("features", FINGERPRINTS.values())
def test_probabilities_sum_to_one(features):
    probs = NetworkDiagnosticRules().get_all_probabilities(features)
    assert len(probs) == 8
    assert abs(sum(probs.values()) - 1.0) < 1e-9
