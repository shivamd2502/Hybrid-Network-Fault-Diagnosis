"""
dataset.py - synthetic network-fault dataset generator (v3).

Each of the 8 fault classes has a "fingerprint": fixed values for 7 binary
diagnostic flags, class-specific ranges for 3 continuous metrics, and a bank of
20 free-text symptom sentences. Samples are drawn class by class (150 each for
1,200 samples), binary flags are flipped with 2 % probability, and the result
is shuffled.

Usage (from the project root, with .venv activated):
    python src/dataset.py                      # writes data/network_dataset_v3.csv
    python src/dataset.py --samples 2400 --out data/my_dataset.csv

The default call reproduces data/network_dataset_v3.csv exactly (seed 42).
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET_CSV, RANDOM_STATE, SEVERITY, SOLUTIONS  # noqa: E402

# ============================================================================
# SYMPTOM VARIATIONS AND FINGERPRINTS (one entry per fault class)
# ============================================================================
SYMPTOM_VARIATIONS = {

    "Router Issue": {
        "keywords": [
            "no internet connection at all",
            "wifi connected but no internet access",
            "router is offline or unresponsive",
            "internet keeps dropping every few minutes",
            "cannot ping the default gateway address",
            "reset router but still no internet",
            "router lights show no WAN connection",
            "modem not responding to any requests",
            "gateway timeout on all devices",
            "lost internet after power outage",
            "ISP line seems down or disconnected",
            "router admin page not loading",
            "all devices lost internet simultaneously",
            "broadband light on router is red",
            "cannot reach any external server",
            "rebooting router does not restore internet",
            "no packets getting through to ISP",
            "WAN port on router showing no link",
            "router stuck in boot loop",
            "internet was working then suddenly dropped"
        ],
        # Diagnostic fingerprint
        "symptoms": {
            "ping_gateway": 0,   # can't reach gateway
            "has_ip":       1,   # device has an IP (DHCP worked locally)
            "ping_ip":      0,   # no external IP reachable
            "ping_domain":  0,
            "ip_conflict":  0,
            "arp_table_ok": 1,
            "subnet_matches_gw": 1,
        },
        "dns_response_time_ms": (800, 5000),   # high — no route out
        "packet_loss_pct":      (60, 100),
        "traceroute_hops":      (1, 2),         # dies at hop 1
    },

    "DNS Issue": {
        "keywords": [
            "cannot open any websites in browser",
            "dns resolution failed error message",
            "site not found in all browsers",
            "can ping IP address but websites do not load",
            "domain names not resolving correctly",
            "nslookup returns no answer",
            "browser shows server not found",
            "all websites give DNS error",
            "can reach server by IP but not by hostname",
            "dns lookup failure for every domain",
            "changing DNS server fixes the problem",
            "flushing DNS cache resolves temporarily",
            "local network works but no internet browsing",
            "ping google.com fails but ping 8.8.8.8 works",
            "nameserver not responding to queries",
            "internet connected but no browsing possible",
            "google.com does not resolve to any IP",
            "DNS server is not reachable",
            "host command returns NXDOMAIN",
            "resolver returning empty responses"
        ],
        "symptoms": {
            "ping_gateway": 1,
            "has_ip":       1,
            "ping_ip":      1,   # can reach IPs — DNS is the failure point
            "ping_domain":  0,
            "ip_conflict":  0,
            "arp_table_ok": 1,
            "subnet_matches_gw": 1,
        },
        "dns_response_time_ms": (50, 400),      # DNS responds but wrongly
        "packet_loss_pct":      (0, 5),
        "traceroute_hops":      (8, 20),
    },

    "DNS Timeout": {
        "keywords": [
            "dns query is timing out every time",
            "nslookup request timed out message",
            "dns server not responding within timeout",
            "very slow dns resolution taking many seconds",
            "dig command shows timeout waiting for server",
            "websites load extremely slowly due to DNS",
            "dns response takes over 10 seconds",
            "intermittent dns timeouts throughout the day",
            "recursive dns query timing out",
            "dns server unreachable but gateway is fine",
            "dns lookup hangs indefinitely",
            "name resolution delay causing page load failure",
            "ISP dns server responding too slowly",
            "alternate DNS like 8.8.8.8 works but default does not",
            "timeout waiting for nameserver response",
            "dns resolution succeeds only sometimes",
            "high latency on all dns queries",
            "dns cache keeps expiring with slow refresh",
            "UDP port 53 queries dropping frequently",
            "ping works but browser hangs on dns step"
        ],
        "symptoms": {
            "ping_gateway": 1,
            "has_ip":       1,
            "ping_ip":      1,
            "ping_domain":  0,   # domain fails — but due to TIMEOUT not NXDOMAIN
            "ip_conflict":  0,
            "arp_table_ok": 1,
            "subnet_matches_gw": 1,
        },
        # KEY discriminator vs DNS Issue: response time is very high
        "dns_response_time_ms": (8000, 30000),
        "packet_loss_pct":      (0, 8),
        "traceroute_hops":      (8, 20),
    },

    "IP Conflict": {
        "keywords": [
            "duplicate ip address detected on network",
            "ip conflict error shown by operating system",
            "another device already has my ip address",
            "network conflict notification from Windows",
            "ip address already in use on this network",
            "arp conflict detected for assigned address",
            "two devices sharing the same ip address",
            "static ip clashes with dhcp assigned address",
            "cannot communicate due to duplicate ip",
            "arpa table shows duplicate entry",
            "ip collision detected on LAN segment",
            "cannot use this ip address it is taken",
            "network shows ip conflict in event log",
            "intermittent connectivity due to ip conflict",
            "device keeps losing connectivity every minute",
            "ping replies coming from wrong MAC address",
            "ARP cache poisoned by duplicate IP",
            "network adapter reporting address conflict",
            "ip address unavailable conflict with another host",
            "connectivity drops when second device joins network"
        ],
        "symptoms": {
            "ping_gateway": 0,   # conflict breaks routing
            "has_ip":       1,
            "ping_ip":      0,
            "ping_domain":  0,
            "ip_conflict":  1,   # KEY flag
            "arp_table_ok": 0,   # ARP table corrupt
            "subnet_matches_gw": 1,
        },
        "dns_response_time_ms": (500, 3000),
        "packet_loss_pct":      (20, 70),
        "traceroute_hops":      (1, 3),
    },

    "DHCP Failure": {
        "keywords": [
            "no ip address assigned by dhcp server",
            "169.254 self assigned apipa address instead of dhcp",
            "dhcp server not responding to discovery",
            "cannot obtain dhcp lease on this network",
            "dhcp discovery packets timing out",
            "dhcp offer never received from server",
            "dhcp lease expired and cannot renew",
            "dhcp server unavailable or disabled",
            "automatic private ip assigned instead of network ip",
            "ipconfig shows 169.254 address indicating no dhcp",
            "dhcp scope may be exhausted on router",
            "dhcp request timeout after multiple attempts",
            "router dhcp disabled causing no ip assignment",
            "cannot get valid ip from access point",
            "dhcp server unreachable from this vlan",
            "link local address only no real ip",
            "ip configuration failed dhcp not responding",
            "network adapter shows limited connectivity",
            "dhcp nak received instead of ack",
            "device stuck with zero configuration ip"
        ],
        "symptoms": {
            "ping_gateway": 0,
            "has_ip":       0,   # KEY: no valid IP at all
            "ping_ip":      0,
            "ping_domain":  0,
            "ip_conflict":  0,
            "arp_table_ok": 1,
            "subnet_matches_gw": 0,
        },
        "dns_response_time_ms": (500, 3000),
        "packet_loss_pct":      (80, 100),
        "traceroute_hops":      (0, 1),
    },

    "Gateway Unreachable": {
        "keywords": [
            "default gateway is not reachable by ping",
            "no route to host error message returned",
            "routing problem preventing gateway access",
            "cannot ping the default gateway address",
            "gateway device appears to be offline",
            "routing table missing default route entry",
            "network unreachable error from ping command",
            "host unreachable icmp message returned",
            "gateway configuration incorrect or missing",
            "invalid or missing default route to gateway",
            "traceroute fails at first hop only",
            "gateway ip not responding to arp requests",
            "wrong gateway ip configured on device",
            "route print shows no default gateway",
            "gateway device powered off or faulty",
            "netstat shows no default route",
            "first hop in traceroute is unreachable",
            "ping returns destination host unreachable",
            "gateway mac address missing from arp cache",
            "default route deleted from routing table"
        ],
        "symptoms": {
            "ping_gateway": 0,   # gateway unreachable
            "has_ip":       1,   # device HAS an IP (unlike DHCP failure)
            "ping_ip":      0,
            "ping_domain":  0,
            "ip_conflict":  0,
            "arp_table_ok": 1,
            "subnet_matches_gw": 1,  # subnet ok, routing is the issue
        },
        # KEY discriminator vs Router Issue: fewer packet loss, hops die at 1
        "dns_response_time_ms": (800, 4000),
        "packet_loss_pct":      (90, 100),
        "traceroute_hops":      (1, 1),
    },

    "Network Adapter Issue": {
        "keywords": [
            "network adapter is disabled in device manager",
            "ethernet port not working at all",
            "network interface showing as down",
            "wifi adapter not found by operating system",
            "no network hardware detected on this computer",
            "network card driver missing or corrupted",
            "ethernet cable shows unplugged in system tray",
            "network adapter error in device manager",
            "wifi not working adapter unresponsive",
            "network device unavailable or missing",
            "adapter not responding to enable command",
            "network interface offline after restart",
            "ethernet unplugged notification even with cable",
            "adapter driver problem yellow exclamation mark",
            "network hardware failure suspected",
            "wireless adapter completely dead no signal",
            "NIC not detected in bios or operating system",
            "device manager shows network adapter with error",
            "uninstalling driver fixes adapter temporarily",
            "physical network interface card faulty"
        ],
        "symptoms": {
            "ping_gateway": 0,
            "has_ip":       0,   # no IP because adapter is down
            "ping_ip":      0,
            "ping_domain":  0,
            "ip_conflict":  0,
            "arp_table_ok": 0,   # no ARP possible
            "subnet_matches_gw": 0,
        },
        "dns_response_time_ms": (0, 100),      # never even sends a query
        "packet_loss_pct":      (100, 100),
        "traceroute_hops":      (0, 0),
    },

    "Subnet Mismatch": {
        "keywords": [
            "wrong subnet mask configured on this device",
            "device not on same subnet as default gateway",
            "subnet mismatch causing connectivity failure",
            "incorrect subnet configuration for this network",
            "subnet does not match gateway subnet",
            "cannot reach gateway due to subnet error",
            "ip address outside valid subnet range",
            "netmask configuration is incorrect",
            "subnet calculation shows device out of range",
            "gateway on different subnet than device",
            "invalid subnet assigned by dhcp or static",
            "cidr prefix length configured incorrectly",
            "subnetting error causing routing failure",
            "255.255.0.0 instead of 255.255.255.0 configured",
            "wrong network mask on interface",
            "device showing on different logical network",
            "layer 3 addressing mismatch detected",
            "cannot ARP for gateway across subnet boundary",
            "broadcast address conflicts with gateway",
            "host bits set in subnet mask incorrectly"
        ],
        "symptoms": {
            "ping_gateway": 0,   # can't reach gateway (wrong subnet)
            "has_ip":       1,   # has an IP but on wrong subnet
            "ping_ip":      0,
            "ping_domain":  0,
            "ip_conflict":  0,
            "arp_table_ok": 1,
            "subnet_matches_gw": 0,  # KEY discriminator
        },
        "dns_response_time_ms": (500, 3000),
        "packet_loss_pct":      (95, 100),
        "traceroute_hops":      (0, 1),
    },
}

# ============================================================================
# DATASET GENERATOR
# ============================================================================

BINARY_NOISE = 0.02  # probability of flipping each binary flag


def generate_dataset(num_samples: int = 1200, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate the synthetic dataset with one fingerprint per fault class."""
    np.random.seed(random_state)
    random.seed(random_state)

    issue_types = list(SYMPTOM_VARIATIONS.keys())
    samples_per_issue = num_samples // len(issue_types)

    rows = []
    for issue_type in issue_types:
        cfg = SYMPTOM_VARIATIONS[issue_type]
        syms = cfg["symptoms"]
        rt_lo, rt_hi = cfg["dns_response_time_ms"]
        pl_lo, pl_hi = cfg["packet_loss_pct"]
        hop_lo, hop_hi = cfg["traceroute_hops"]

        for _ in range(samples_per_issue):
            row = {
                # Text feature (for TF-IDF)
                "symptom_text": random.choice(cfg["keywords"]),
                # Binary diagnostic flags
                "ping_gateway": syms.get("ping_gateway", random.randint(0, 1)),
                "has_ip": syms.get("has_ip", random.randint(0, 1)),
                "ping_ip": syms.get("ping_ip", random.randint(0, 1)),
                "ping_domain": syms.get("ping_domain", random.randint(0, 1)),
                "ip_conflict": syms.get("ip_conflict", 0),
                "arp_table_ok": syms.get("arp_table_ok", random.randint(0, 1)),
                "subnet_matches_gw": syms.get("subnet_matches_gw", 1),
                # Continuous discriminating features
                "dns_response_time_ms": random.randint(rt_lo, rt_hi),
                "packet_loss_pct": random.randint(pl_lo, pl_hi),
                "traceroute_hops": random.randint(hop_lo, max(hop_lo, hop_hi)),
                # Categorical context (independent of the label)
                "network_type": random.choice(["WiFi", "Ethernet"]),
                "os_type": random.choice(["Windows", "macOS", "Linux"]),
                # Target
                "diagnosis": issue_type,
            }

            # 2 % chance of flipping each binary flag (measurement noise)
            for feat in ["ping_gateway", "has_ip", "ping_ip", "ping_domain",
                         "arp_table_ok", "subnet_matches_gw"]:
                if random.random() < BINARY_NOISE:
                    row[feat] = 1 - row[feat]

            rows.append(row)

    return pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)


def create_dataset(num_samples: int = 1200, out_path: Path = DATASET_CSV) -> pd.DataFrame:
    """Generate the dataset, add solutions/severity/metadata columns and save it as CSV."""
    print(f"[1/3] Generating {num_samples} samples (seed {RANDOM_STATE})...")
    df = generate_dataset(num_samples)

    print("[2/3] Adding solutions and metadata...")
    df["solutions"] = df["diagnosis"].map(lambda x: " | ".join(SOLUTIONS.get(x, ["No solution available"])))
    df["severity"] = df["diagnosis"].map(SEVERITY)
    df["dataset_version"] = "3.0"
    df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="h")

    col_order = [
        "symptom_text",
        "ping_gateway", "has_ip", "ping_ip", "ping_domain",
        "ip_conflict", "arp_table_ok", "subnet_matches_gw",
        "dns_response_time_ms", "packet_loss_pct", "traceroute_hops",
        "network_type", "os_type",
        "diagnosis", "severity", "solutions", "timestamp", "dataset_version",
    ]
    df = df[col_order]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # lineterminator="\n" keeps the file byte-identical on Windows and Linux
    df.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"[3/3] Saved {len(df)} rows x {len(df.columns)} columns to {out_path}")
    for diag in sorted(df["diagnosis"].unique()):
        n = int((df["diagnosis"] == diag).sum())
        print(f"    {diag:25s}: {n:4d} ({n / len(df) * 100:.1f}%)")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate the synthetic network-fault dataset.")
    p.add_argument("--samples", type=int, default=1200)
    p.add_argument("--out", type=Path, default=DATASET_CSV)
    args = p.parse_args()
    create_dataset(args.samples, args.out)