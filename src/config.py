"""
config.py - shared constants: paths, feature schema, fault classes and remediation steps.

All paths are resolved relative to the project root with pathlib, so every script
works on Windows and Linux no matter which directory it is launched from.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TESTS_DIR = PROJECT_ROOT / "tests"

DATASET_CSV = DATA_DIR / "network_dataset_v3.csv"
PIPELINE_PKL = MODELS_DIR / "pipeline.pkl"
ENCODERS_PKL = MODELS_DIR / "encoders.pkl"
METRICS_JSON = RESULTS_DIR / "metrics.json"
SCENARIOS_JSON = TESTS_DIR / "scenarios.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# --------------------------------------------------------------------------
# Feature schema (must match the trained pipeline)
# --------------------------------------------------------------------------
TEXT_FEATURE = "symptom_text"
BINARY_FEATURES = [
    "ping_gateway", "has_ip", "ping_ip", "ping_domain",
    "ip_conflict", "arp_table_ok", "subnet_matches_gw",
]
CONTINUOUS_FEATURES = ["dns_response_time_ms", "packet_loss_pct", "traceroute_hops"]
NUMERIC_FEATURES = BINARY_FEATURES + CONTINUOUS_FEATURES
CATEGORICAL_FEATURES = ["network_type", "os_type"]
TARGET = "diagnosis"
DROP_COLS = ["solutions", "timestamp", "dataset_version", "severity"]

# --------------------------------------------------------------------------
# Fault classes
# --------------------------------------------------------------------------
DIAGNOSES = [
    "Router Issue", "DNS Issue", "DNS Timeout", "IP Conflict",
    "DHCP Failure", "Gateway Unreachable", "Network Adapter Issue", "Subnet Mismatch",
]

SEVERITY = {
    "Network Adapter Issue": "High",
    "DHCP Failure": "High",
    "IP Conflict": "High",
    "Router Issue": "Medium",
    "Gateway Unreachable": "Medium",
    "Subnet Mismatch": "Medium",
    "DNS Issue": "Low",
    "DNS Timeout": "Low",
}

SOLUTIONS = {
    "Router Issue": [
        "Check if router is powered on — verify all indicator lights",
        "Power cycle: unplug router for 30s then reconnect",
        "Check WAN/Internet cable between router and modem",
        "Verify ISP service is active — call ISP or check status page",
        "Check router WAN IP in admin panel (usually 192.168.1.1)",
        "Update router firmware from manufacturer website",
        "Review router logs for PPPoE or DHCP errors from ISP",
        "Factory reset router if all else fails",
    ],
    "DNS Issue": [
        "Change DNS to Google (8.8.8.8, 8.8.4.4) or Cloudflare (1.1.1.1)",
        "Flush DNS cache — Windows: ipconfig /flushdns",
        "Flush DNS cache — macOS: sudo dscacheutil -flushcache",
        "Test with nslookup google.com to confirm DNS failure",
        "Restart network adapter to clear DNS state",
        "Check router DNS settings in admin panel",
        "Try alternate DNS: Quad9 (9.9.9.9) or OpenDNS (208.67.222.222)",
        "Disable DNS-over-HTTPS temporarily to test",
    ],
    "DNS Timeout": [
        "Test DNS timeout: nslookup -timeout=5 google.com",
        "Switch to a faster DNS server (1.1.1.1 is fastest globally)",
        "Flush DNS cache — Windows: ipconfig /flushdns",
        "Check if UDP port 53 is blocked by local firewall",
        "Ping 1.1.1.1 — if that works but DNS times out, ISP is throttling DNS",
        "Contact ISP — their DNS resolver may be degraded",
        "Enable DNS over HTTPS/TLS to bypass ISP interference",
        "Restart router to reset DNS cache and forwarder state",
    ],
    "IP Conflict": [
        "Identify conflicting device: arp -a | look for duplicate MACs",
        "Release IP: ipconfig /release (Windows) or sudo dhclient -r",
        "Renew IP: ipconfig /renew or sudo dhclient",
        "If using static IP, move it outside DHCP pool range",
        "Set DHCP reservations in router to prevent future conflicts",
        "Check DHCP pool size — expand if needed",
        "Restart router to clear stale DHCP lease table",
        "Reserve MAC-to-IP bindings for all static devices in router",
    ],
    "DHCP Failure": [
        "Verify DHCP is enabled in router admin panel",
        "Release and renew: ipconfig /release && ipconfig /renew",
        "Power cycle router — 30s off clears DHCP server state",
        "Check DHCP pool is not exhausted (add more addresses)",
        "Temporarily assign a static IP to confirm it is a DHCP issue",
        "Check if device is in a VLAN that can reach the DHCP server",
        "Increase DHCP lease time in router settings",
        "Reset router DHCP scope to defaults",
    ],
    "Gateway Unreachable": [
        "Verify gateway IP: route print (Windows) or ip route show (Linux)",
        "Ping gateway: ping <gateway_ip> — confirm it is unreachable",
        "Re-enter correct default gateway in network settings",
        "Check if gateway device is powered on and its port is active",
        "Restart your network adapter",
        "Check routing table for stale or conflicting routes",
        "Reset TCP/IP stack: netsh int ip reset (Windows)",
        "Verify no firewall rule is blocking ICMP to gateway",
    ],
    "Network Adapter Issue": [
        "Open Device Manager — check for yellow ! on network adapter",
        "Right-click adapter → Enable if disabled",
        "Check physical cable connection — try a different cable",
        "Update driver: Device Manager → Update driver",
        "Uninstall adapter in Device Manager then reboot to reinstall",
        "Try a different USB or PCIe NIC to isolate hardware failure",
        "Run Windows Network Diagnostics or Linux ethtool -t eth0",
        "Replace NIC if driver reinstall does not resolve it",
    ],
    "Subnet Mismatch": [
        "Check device IP and mask: ipconfig /all or ip addr",
        "Identify gateway IP and its subnet",
        "Verify device subnet contains the gateway IP (use subnet calculator)",
        "Example correct config: device 192.168.1.50/24, gateway 192.168.1.1",
        "Set correct subnet mask: 255.255.255.0 for /24 networks",
        "If using static IP, reconfigure with matching subnet",
        "Check DHCP server subnet option — it may be misconfigured",
        "Restart network after fixing subnet to apply changes",
    ],
}


def silence_joblib_warning():
    """Hide a harmless sklearn/joblib UserWarning that is printed once per worker."""
    import warnings
    warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
