"""
RFC-BASED NETWORK DIAGNOSTIC RULE ENGINE
=========================================
Implements expert system rules derived from networking RFCs and best practices:
  - RFC 1122 (Host Requirements)
  - RFC 2131 (DHCP)
  - RFC 1035 (DNS)
  - RFC 792 (ICMP)
  - RFC 826 (ARP)

This provides symbolic reasoning to complement deep learning predictions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkDiagnosticRules:
    """
    Rule-based expert system for network troubleshooting.
    
    Each rule returns:
      - diagnosis: str (issue type)
      - confidence: float (0-1)
      - matched_rules: List[str] (which RFC rules fired)
    """
    
    def __init__(self):
        self.diagnoses = [
            "Router Issue",
            "DNS Issue", 
            "DNS Timeout",
            "IP Conflict",
            "DHCP Failure",
            "Gateway Unreachable",
            "Network Adapter Issue",
            "Subnet Mismatch"
        ]
        
        # Rule confidence weights (tuned from domain knowledge)
        self.rule_weights = {
            "dhcp_no_ip": 0.95,
            "adapter_complete_failure": 0.98,
            "ip_conflict_flag": 0.90,
            "subnet_mismatch_flag": 0.88,
            "gateway_unreachable_specific": 0.85,
            "dns_timeout_high_latency": 0.82,
            "dns_resolution_failure": 0.80,
            "router_wan_failure": 0.78,
        }
    
    def diagnose(self, features: Dict) -> Tuple[str, float, List[str]]:
        """
        Apply diagnostic rules in priority order.
        
        Args:
            features: Dict with keys matching NUMERIC_FEATURES + CATEGORICAL_FEATURES
        
        Returns:
            (diagnosis, confidence, matched_rules)
        """
        
        # Extract features
        ping_gw = features.get("ping_gateway", 0)
        has_ip = features.get("has_ip", 1)
        ping_ip = features.get("ping_ip", 0)
        ping_domain = features.get("ping_domain", 0)
        ip_conflict = features.get("ip_conflict", 0)
        arp_ok = features.get("arp_table_ok", 1)
        subnet_ok = features.get("subnet_matches_gw", 1)
        dns_rt = features.get("dns_response_time_ms", 0)
        pkt_loss = features.get("packet_loss_pct", 0)
        hops = features.get("traceroute_hops", 0)
        
        # ═══════════════════════════════════════════════════════════════════
        # RULE 1: DHCP Failure (RFC 2131)
        # ═══════════════════════════════════════════════════════════════════
        if has_ip == 0 and arp_ok == 1:
            # No IP assigned, but adapter is functional
            return "DHCP Failure", self.rule_weights["dhcp_no_ip"], [
                "RFC 2131: No valid IP address obtained from DHCP server",
                "Link-local 169.254.x.x indicates DHCP discovery failure"
            ]
        
        # ═══════════════════════════════════════════════════════════════════
        # RULE 2: Network Adapter Issue (Physical Layer)
        # ═══════════════════════════════════════════════════════════════════
        if has_ip == 0 and arp_ok == 0 and pkt_loss >= 95:
            # No IP, no ARP, complete packet loss = hardware failure
            return "Network Adapter Issue", self.rule_weights["adapter_complete_failure"], [
                "Physical layer failure: No ARP resolution possible",
                "Complete packet loss indicates disabled or faulty NIC",
                "RFC 1122: Link layer must be operational for IP stack"
            ]
        
        # ═══════════════════════════════════════════════════════════════════
        # RULE 3: IP Conflict (RFC 826 - ARP)
        # ═══════════════════════════════════════════════════════════════════
        if ip_conflict == 1:
            # OS detected duplicate IP via ARP
            return "IP Conflict", self.rule_weights["ip_conflict_flag"], [
                "RFC 826: Duplicate IP address detected via ARP",
                "ARP table shows conflicting MAC address for same IP",
                "OS generated IP conflict warning"
            ]
        
        # ═══════════════════════════════════════════════════════════════════
        # RULE 4: Subnet Mismatch (RFC 1122 - Addressing)
        # ═══════════════════════════════════════════════════════════════════
        if has_ip == 1 and subnet_ok == 0 and ping_gw == 0:
            # Has IP but wrong subnet → can't reach gateway
            return "Subnet Mismatch", self.rule_weights["subnet_mismatch_flag"], [
                "RFC 1122: Device subnet does not contain default gateway",
                "Layer 3 addressing error prevents routing",
                "Subnet mask misconfiguration detected"
            ]
        
        # ═══════════════════════════════════════════════════════════════════
        # RULE 5: Gateway Unreachable (RFC 1122 - Routing)
        # ═══════════════════════════════════════════════════════════════════
        if ping_gw == 0 and has_ip == 1 and subnet_ok == 1 and pkt_loss >= 90 and hops <= 1:
            # Can't ping gateway despite correct subnet
            return "Gateway Unreachable", self.rule_weights["gateway_unreachable_specific"], [
                "RFC 1122: Default gateway not responding to ICMP echo",
                "ARP resolution failing for gateway MAC address",
                "Traceroute dies at hop 1 (first hop = gateway)",
                "Gateway device may be offline or misconfigured"
            ]
        
        # ═══════════════════════════════════════════════════════════════════
        # RULE 6: DNS Timeout (RFC 1035 - DNS)
        # ═══════════════════════════════════════════════════════════════════
        if ping_ip == 1 and ping_domain == 0 and dns_rt > 8000:
            # Can reach IPs but DNS is timing out
            return "DNS Timeout", self.rule_weights["dns_timeout_high_latency"], [
                "RFC 1035: DNS query timeout (>8s response time)",
                "UDP port 53 reachable but resolver not responding",
                "Network layer functional (ping 8.8.8.8 works)",
                "DNS server overloaded or ISP throttling queries"
            ]
        
        # ═══════════════════════════════════════════════════════════════════
        # RULE 7: DNS Issue (RFC 1035 - DNS Resolution)
        # ═══════════════════════════════════════════════════════════════════
        if ping_ip == 1 and ping_domain == 0 and dns_rt < 8000:
            # Can reach IPs but DNS fails (not timeout, just wrong/no answer)
            return "DNS Issue", self.rule_weights["dns_resolution_failure"], [
                "RFC 1035: DNS resolution failure (NXDOMAIN or SERVFAIL)",
                "Can ping external IPs but domain names don't resolve",
                "DNS server misconfigured or returning incorrect results",
                "Nameserver not responding with valid A records"
            ]
        
        # ═══════════════════════════════════════════════════════════════════
        # RULE 8: Router Issue (WAN failure)
        # ═══════════════════════════════════════════════════════════════════
        if ping_gw == 0 and ping_ip == 0 and has_ip == 1 and pkt_loss >= 60:
            # Has local IP but can't reach gateway or internet
            return "Router Issue", self.rule_weights["router_wan_failure"], [
                "RFC 1122: WAN interface failure on router",
                "Local network operational (DHCP worked) but no internet",
                "Router not routing packets to ISP",
                "High packet loss indicates upstream link failure"
            ]
        
        # ═══════════════════════════════════════════════════════════════════
        # FALLBACK: If no strong rule matches, use heuristics
        # ═══════════════════════════════════════════════════════════════════
        return self._heuristic_fallback(features)
    
    def _heuristic_fallback(self, features: Dict) -> Tuple[str, float, List[str]]:
        """
        Weak heuristics when no strong rule fires.
        Returns lower confidence to signal uncertainty.
        """
        ping_gw = features.get("ping_gateway", 0)
        has_ip = features.get("has_ip", 1)
        ping_ip = features.get("ping_ip", 0)
        pkt_loss = features.get("packet_loss_pct", 0)
        
        # Guess based on symptom patterns
        if pkt_loss > 50:
            if ping_gw == 0:
                return "Router Issue", 0.55, ["Heuristic: High packet loss + no gateway"]
            else:
                return "Gateway Unreachable", 0.50, ["Heuristic: High packet loss"]
        
        if has_ip == 0:
            return "DHCP Failure", 0.60, ["Heuristic: No IP assigned"]
        
        if ping_ip == 0 and ping_gw == 1:
            return "Router Issue", 0.58, ["Heuristic: Gateway ok but no internet"]
        
        # Ultimate fallback
        return "Router Issue", 0.40, ["No strong rule matched - defaulting to most common issue"]
    
    def explain_rules(self, diagnosis: str) -> List[str]:
        """
        Return all possible rules that could lead to this diagnosis.
        Useful for debugging and transparency.
        """
        rule_explanations = {
            "DHCP Failure": [
                "Device has no valid IP address (not 169.254.x.x)",
                "DHCP discovery packets not receiving offers from server",
                "DHCP scope exhausted or server disabled on router"
            ],
            "Network Adapter Issue": [
                "No IP address AND no ARP table entries",
                "Complete packet loss (100%)",
                "Physical layer failure - cable unplugged or NIC disabled"
            ],
            "IP Conflict": [
                "Operating system detected duplicate IP address",
                "ARP cache shows multiple MACs for same IP",
                "Another device is using your IP address"
            ],
            "Subnet Mismatch": [
                "Device has IP but subnet doesn't contain gateway",
                "Subnet mask misconfigured (e.g., /16 instead of /24)",
                "Gateway IP outside of device's calculated subnet"
            ],
            "Gateway Unreachable": [
                "Cannot ping default gateway despite correct subnet",
                "Traceroute fails at first hop",
                "Gateway device offline or routing table missing default route"
            ],
            "DNS Timeout": [
                "Can ping 8.8.8.8 but domain names don't resolve",
                "DNS response time > 8 seconds",
                "UDP port 53 queries timing out"
            ],
            "DNS Issue": [
                "Can ping external IPs but nslookup fails",
                "DNS returns NXDOMAIN or SERVFAIL",
                "Wrong DNS server configured or DNS server misconfigured"
            ],
            "Router Issue": [
                "WAN link down on router",
                "Can get IP locally but no internet connectivity",
                "ISP connection failed or router not routing packets"
            ]
        }
        
        return rule_explanations.get(diagnosis, ["No explanation available"])
    
    def get_all_probabilities(self, features: Dict) -> Dict[str, float]:
        """
        Score all diagnoses instead of just returning the top match.
        Used for model fusion with LSTM.
        
        Returns:
            Dict mapping diagnosis -> confidence score
        """
        scores = {}
        
        # Run primary diagnosis
        primary_diag, primary_conf, _ = self.diagnose(features)
        scores[primary_diag] = primary_conf
        
        # Assign small probabilities to other diagnoses based on feature overlap
        remaining_prob = 1.0 - primary_conf
        other_diagnoses = [d for d in self.diagnoses if d != primary_diag]
        
        # Distribute remaining probability based on partial rule matches
        partial_scores = self._compute_partial_scores(features, other_diagnoses)
        total_partial = sum(partial_scores.values())
        
        if total_partial > 0:
            for diag, partial_score in partial_scores.items():
                scores[diag] = remaining_prob * (partial_score / total_partial)
        else:
            # Uniform distribution if no partial matches
            for diag in other_diagnoses:
                scores[diag] = remaining_prob / len(other_diagnoses)
        
        # Normalize to ensure sum = 1.0
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def _compute_partial_scores(self, features: Dict, diagnoses: List[str]) -> Dict[str, float]:
        """
        Assign partial scores based on how many rule conditions are met.
        """
        scores = {diag: 0.0 for diag in diagnoses}
        
        ping_gw = features.get("ping_gateway", 0)
        has_ip = features.get("has_ip", 1)
        ping_ip = features.get("ping_ip", 0)
        ping_domain = features.get("ping_domain", 0)
        ip_conflict = features.get("ip_conflict", 0)
        arp_ok = features.get("arp_table_ok", 1)
        subnet_ok = features.get("subnet_matches_gw", 1)
        dns_rt = features.get("dns_response_time_ms", 0)
        pkt_loss = features.get("packet_loss_pct", 0)
        hops = features.get("traceroute_hops", 0)
        
        # Partial match scoring (0-1 for each diagnosis)
        if "DHCP Failure" in diagnoses:
            if has_ip == 0:
                scores["DHCP Failure"] += 0.7
            if subnet_ok == 0:
                scores["DHCP Failure"] += 0.3
        
        if "Network Adapter Issue" in diagnoses:
            if pkt_loss >= 80:
                scores["Network Adapter Issue"] += 0.5
            if arp_ok == 0:
                scores["Network Adapter Issue"] += 0.5
        
        if "IP Conflict" in diagnoses:
            if arp_ok == 0:
                scores["IP Conflict"] += 0.4
            if ping_gw == 0 and has_ip == 1:
                scores["IP Conflict"] += 0.3
        
        if "Subnet Mismatch" in diagnoses:
            if subnet_ok == 0:
                scores["Subnet Mismatch"] += 0.6
            if ping_gw == 0 and has_ip == 1:
                scores["Subnet Mismatch"] += 0.4
        
        if "Gateway Unreachable" in diagnoses:
            if ping_gw == 0:
                scores["Gateway Unreachable"] += 0.5
            if hops <= 2:
                scores["Gateway Unreachable"] += 0.3
            if pkt_loss >= 70:
                scores["Gateway Unreachable"] += 0.2
        
        if "DNS Timeout" in diagnoses:
            if dns_rt > 5000:
                scores["DNS Timeout"] += 0.6
            if ping_ip == 1 and ping_domain == 0:
                scores["DNS Timeout"] += 0.4
        
        if "DNS Issue" in diagnoses:
            if ping_ip == 1 and ping_domain == 0:
                scores["DNS Issue"] += 0.7
            if dns_rt < 5000:
                scores["DNS Issue"] += 0.3
        
        if "Router Issue" in diagnoses:
            if ping_ip == 0 and ping_gw == 0 and has_ip == 1:
                scores["Router Issue"] += 0.6
            if pkt_loss >= 50:
                scores["Router Issue"] += 0.4
        
        return scores


# ═══════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════

def test_rule_engine():
    """Test the rule engine with known scenarios."""
    engine = NetworkDiagnosticRules()
    
    test_cases = [
        {
            "name": "DHCP Failure",
            "features": {
                "ping_gateway": 0, "has_ip": 0, "ping_ip": 0, "ping_domain": 0,
                "ip_conflict": 0, "arp_table_ok": 1, "subnet_matches_gw": 0,
                "dns_response_time_ms": 500, "packet_loss_pct": 90, "traceroute_hops": 0
            },
            "expected": "DHCP Failure"
        },
        {
            "name": "DNS Timeout",
            "features": {
                "ping_gateway": 1, "has_ip": 1, "ping_ip": 1, "ping_domain": 0,
                "ip_conflict": 0, "arp_table_ok": 1, "subnet_matches_gw": 1,
                "dns_response_time_ms": 12000, "packet_loss_pct": 2, "traceroute_hops": 10
            },
            "expected": "DNS Timeout"
        },
        {
            "name": "IP Conflict",
            "features": {
                "ping_gateway": 0, "has_ip": 1, "ping_ip": 0, "ping_domain": 0,
                "ip_conflict": 1, "arp_table_ok": 0, "subnet_matches_gw": 1,
                "dns_response_time_ms": 800, "packet_loss_pct": 50, "traceroute_hops": 1
            },
            "expected": "IP Conflict"
        }
    ]
    
    print("\n" + "="*70)
    print("TESTING RULE-BASED ENGINE")
    print("="*70)
    
    for case in test_cases:
        diag, conf, rules = engine.diagnose(case["features"])
        
        print(f"\nTest: {case['name']}")
        print(f"  Predicted: {diag} (confidence: {conf:.2f})")
        print(f"  Expected:  {case['expected']}")
        print(f"  Match: {'✓' if diag == case['expected'] else '✗'}")
        print(f"  Rules fired:")
        for rule in rules:
            print(f"    - {rule}")


if __name__ == "__main__":
    test_rule_engine()