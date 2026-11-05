# """
# NETWORK TROUBLESHOOTING DATASET GENERATOR
# =========================================
# Generates synthetic dataset of network issues with symptoms and diagnoses.
# Dataset structure: Features + Labels for ML training

# Author: AI Assistant
# Date: 2026
# """

# import pandas as pd
# import numpy as np
# from datetime import datetime

# # ============================================================================
# # STEP 1: DEFINE NETWORK ISSUES & RULES
# # ============================================================================

# NETWORK_ISSUES = {
#     "Router Issue": {
#         "description": "Device cannot reach gateway/router",
#         "keywords": ["no internet", "cannot ping gateway", "router offline", "wifi connected no internet"],
#         "symptoms": {"ping_gateway": 0, "has_ip": 1, "ping_ip": 0}
#     },
#     "DNS Issue": {
#         "description": "Device can reach network but cannot resolve domain names",
#         "keywords": ["cannot access websites", "dns failed", "site not found", "ping ip works but google fails"],
#         "symptoms": {"ping_gateway": 1, "has_ip": 1, "ping_ip": 1, "ping_domain": 0}
#     },
#     "IP Conflict": {
#         "description": "Multiple devices have same IP address on network",
#         "keywords": ["duplicate ip", "ip conflict", "network conflict", "cannot connect"],
#         "symptoms": {"has_ip": 1, "ip_conflict": 1, "ping_gateway": 0}
#     },
#     "DHCP Failure": {
#         "description": "Device unable to obtain IP from DHCP server",
#         "keywords": ["no ip address", "169.254 address", "self-assigned ip", "dhcp not responding"],
#         "symptoms": {"has_ip": 0, "ip_conflict": 0}
#     },
#     "Gateway Unreachable": {
#         "description": "Device has IP but gateway is offline/misconfigured",
#         "keywords": ["gateway unreachable", "no default gateway", "routing issue", "cannot reach network"],
#         "symptoms": {"has_ip": 1, "ping_gateway": 0, "ip_conflict": 0}
#     },
#     "Network Adapter Issue": {
#         "description": "Physical network interface is disabled or faulty",
#         "keywords": ["ethernet disabled", "network adapter down", "no network connection", "ethernet unplugged"],
#         "symptoms": {"has_ip": 0, "ping_gateway": 0}
#     },
#     "Subnet Mismatch": {
#         "description": "Device IP not on same subnet as gateway",
#         "keywords": ["wrong subnet", "different network", "subnet issue", "cannot reach gateway"],
#         "symptoms": {"has_ip": 1, "ping_gateway": 0, "ip_conflict": 0}
#     },
#     "DNS Timeout": {
#         "description": "DNS server configured but not responding",
#         "keywords": ["dns timeout", "dns server down", "slow internet", "sites loading slow"],
#         "symptoms": {"ping_gateway": 1, "has_ip": 1, "ping_ip": 1, "ping_domain": 0}
#     }
# }

# # ============================================================================
# # STEP 2: GENERATE SYNTHETIC DATASET
# # ============================================================================

# def generate_dataset(num_samples=300, random_state=42):
#     """
#     Generate synthetic network troubleshooting dataset
    
#     Parameters:
#     -----------
#     num_samples : int
#         Number of samples to generate (default: 300)
#     random_state : int
#         Random seed for reproducibility
    
#     Returns:
#     --------
#     pd.DataFrame : Dataset with features and labels
#     """
    
#     np.random.seed(random_state)
    
#     data = []
#     issue_types = list(NETWORK_ISSUES.keys())
    
#     # Generate samples proportionally from each issue type
#     samples_per_issue = num_samples // len(issue_types)
    
#     for issue_type in issue_types:
#         issue_config = NETWORK_ISSUES[issue_type]
        
#         for _ in range(samples_per_issue):
#             # Base symptoms from issue type
#             symptoms = issue_config["symptoms"].copy()
            
#             # Fill missing features with random or default values
#             row = {
#                 # USER DESCRIPTION (NLP Input)
#                 "symptom_text": np.random.choice(issue_config["keywords"]),
                
#                 # DIAGNOSTIC FEATURES (Binary)
#                 "ping_gateway": symptoms.get("ping_gateway", np.random.randint(0, 2)),
#                 "has_ip": symptoms.get("has_ip", np.random.randint(0, 2)),
#                 "ping_ip": symptoms.get("ping_ip", np.random.randint(0, 2)),
#                 "ping_domain": symptoms.get("ping_domain", np.random.randint(0, 2)),
#                 "ip_conflict": symptoms.get("ip_conflict", np.random.randint(0, 2)),
                
#                 # ENVIRONMENTAL FACTORS
#                 "network_type": np.random.choice(["WiFi", "Ethernet"]),
#                 "os_type": np.random.choice(["Windows", "macOS", "Linux"]),
#                 "recently_updated": np.random.randint(0, 2),
#                 "vpn_enabled": np.random.randint(0, 2),
#                 "firewall_enabled": np.random.randint(0, 2),
                
#                 # LABEL (Diagnosis)
#                 "diagnosis": issue_type,
#             }
            
#             # Add slight noise/variation to symptoms (real-world scenarios)
#             noise_probability = 0.05
#             for feature in ["ping_gateway", "has_ip", "ping_ip", "ping_domain"]:
#                 if np.random.random() < noise_probability:
#                     row[feature] = 1 - row[feature]  # Flip bit with 5% probability
            
#             data.append(row)
    
#     df = pd.DataFrame(data)
    
#     # Shuffle dataset
#     df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
#     return df


# # ============================================================================
# # STEP 3: DEFINE SOLUTIONS FOR EACH DIAGNOSIS
# # ============================================================================

# SOLUTIONS = {
#     "Router Issue": [
#         "Step 1: Check if router is powered on (look for lights on router)",
#         "Step 2: Restart router - unplug for 30 seconds, plug back in",
#         "Step 3: Check Wi-Fi signal strength - move closer to router",
#         "Step 4: Check if other devices can connect (isolate the problem)",
#         "Step 5: Restart your network adapter - disable & re-enable network",
#         "Step 6: Update router firmware from manufacturer website",
#         "Step 7: If still fails - reset router (factory reset) or call ISP"
#     ],
    
#     "DNS Issue": [
#         "Step 1: Change DNS server to Google (8.8.8.8, 8.8.4.4)",
#         "Step 2: Flush DNS cache:",
#         "  - Windows: ipconfig /flushdns",
#         "  - macOS/Linux: sudo dscacheutil -flushcache",
#         "Step 3: Restart network adapter",
#         "Step 4: Test with nslookup google.com",
#         "Step 5: Try alternate DNS (Cloudflare: 1.1.1.1, 1.0.0.1)",
#         "Step 6: Check router DNS settings in admin panel"
#     ],
    
#     "IP Conflict": [
#         "Step 1: Identify device with conflicting IP using arp -a",
#         "Step 2: Release current IP - ipconfig /release (Windows)",
#         "Step 3: Renew DHCP lease - ipconfig /renew (Windows)",
#         "Step 4: Restart network adapter to get new IP",
#         "Step 5: Configure static IP on affected device (if DHCP fails)",
#         "Step 6: Check DHCP pool size in router settings",
#         "Step 7: Restart router to reset DHCP lease table"
#     ],
    
#     "DHCP Failure": [
#         "Step 1: Check DHCP server is enabled in router settings",
#         "Step 2: Release & renew IP address:",
#         "  - Windows: ipconfig /release && ipconfig /renew",
#         "  - macOS: System Preferences > Network > Renew DHCP Lease",
#         "Step 3: Restart router (power cycle)",
#         "Step 4: Check if device is in DHCP scope range",
#         "Step 5: Configure static IP as temporary workaround",
#         "Step 6: Check router DHCP server status in logs"
#     ],
    
#     "Gateway Unreachable": [
#         "Step 1: Verify default gateway - route print (Windows) or netstat -r (macOS/Linux)",
#         "Step 2: Ping default gateway - ping <gateway_ip>",
#         "Step 3: Configure correct default gateway (check network documentation)",
#         "Step 4: Restart TCP/IP stack:",
#         "  - Windows: ipconfig /all then set gateway via Network Settings",
#         "Step 5: Check if gateway device is powered on",
#         "Step 6: Restart your network adapter",
#         "Step 7: Check router is configured with correct WAN IP"
#     ],
    
#     "Network Adapter Issue": [
#         "Step 1: Check if network adapter is disabled in Device Manager",
#         "Step 2: Enable network adapter - right-click > Enable",
#         "Step 3: Check if ethernet cable is connected (physical check)",
#         "Step 4: Restart network adapter - disable then enable",
#         "Step 5: Update network adapter drivers from manufacturer",
#         "Step 6: Uninstall & reinstall network adapter driver",
#         "Step 7: If still fails - adapter may be faulty, replace hardware",
#         "Step 8: Run ipconfig /all to verify adapter is recognized"
#     ],
    
#     "Subnet Mismatch": [
#         "Step 1: Check device IP address - ipconfig (Windows) or ifconfig (Linux)",
#         "Step 2: Check gateway IP and subnet mask",
#         "Step 3: Verify device is on same subnet as gateway",
#         "Step 4: Example: Device 192.168.1.100/24 should have gateway 192.168.1.1",
#         "Step 5: If mismatch, configure correct subnet mask",
#         "Step 6: Use subnet calculator to verify subnet range",
#         "Step 7: Check DHCP server is assigning correct subnet mask",
#         "Step 8: Restart network to apply correct settings"
#     ],
    
#     "DNS Timeout": [
#         "Step 1: Test DNS resolution - nslookup google.com",
#         "Step 2: Change to faster DNS (8.8.8.8 or 1.1.1.1)",
#         "Step 3: Flush DNS cache - ipconfig /flushdns (Windows)",
#         "Step 4: Check if DNS server is reachable - ping 8.8.8.8",
#         "Step 5: Increase DNS timeout in network settings",
#         "Step 6: Check ISP DNS server status (your ISP may have issues)",
#         "Step 7: Restart router to reset DNS cache",
#         "Step 8: Check if firewalls are blocking DNS (port 53)"
#     ]
# }


# # ============================================================================
# # STEP 4: CREATE COMPREHENSIVE DATASET WITH METADATA
# # ============================================================================

# def create_full_dataset(num_samples=300, save_to_csv=True):
#     """
#     Create and save complete dataset with all metadata
    
#     Parameters:
#     -----------
#     num_samples : int
#         Number of samples to generate
#     save_to_csv : bool
#         Whether to save to CSV file
    
#     Returns:
#     --------
#     pd.DataFrame : Complete dataset
#     """
    
#     print("=" * 70)
#     print("NETWORK TROUBLESHOOTING DATASET GENERATOR")
#     print("=" * 70)
    
#     # Generate base dataset
#     print(f"\n[1/4] Generating {num_samples} synthetic samples...")
#     df = generate_dataset(num_samples=num_samples)
    
#     # Add solutions
#     print("[2/4] Adding solution recommendations...")
#     df['solutions'] = df['diagnosis'].map(
#         lambda x: ' | '.join(SOLUTIONS.get(x, ["No solution found"]))
#     )
    
#     # Add metadata
#     print("[3/4] Adding metadata...")
#     df['timestamp'] = datetime.now().isoformat()
#     df['dataset_version'] = "1.0"
#     df['severity'] = df['diagnosis'].apply(
#         lambda x: 'High' if x in ['Network Adapter Issue', 'DHCP Failure'] 
#                   else 'Medium' if x in ['Router Issue', 'Gateway Unreachable']
#                   else 'Low'
#     )
    
#     # Reorder columns
#     column_order = [
#         'symptom_text',
#         'ping_gateway', 'has_ip', 'ping_ip', 'ping_domain', 'ip_conflict',
#         'network_type', 'os_type', 'recently_updated', 'vpn_enabled', 'firewall_enabled',
#         'diagnosis', 'severity', 'solutions', 'timestamp', 'dataset_version'
#     ]
#     df = df[column_order]
    
#     # Save to CSV
#     if save_to_csv:
#         csv_path = 'network_dataset.csv'
#         df.to_csv(csv_path, index=False)
#         print(f"[4/4] Dataset saved to: {csv_path}")
#     else:
#         print("[4/4] Skipping CSV save")
    
#     # Print statistics
#     print("\n" + "=" * 70)
#     print("DATASET STATISTICS")
#     print("=" * 70)
#     print(f"\nTotal samples: {len(df)}")
#     print(f"\nDiagnosis distribution:")
#     print(df['diagnosis'].value_counts())
#     print(f"\nNetwork type distribution:")
#     print(df['network_type'].value_counts())
#     print(f"\nOS distribution:")
#     print(df['os_type'].value_counts())
#     print(f"\nSeverity distribution:")
#     print(df['severity'].value_counts())
    
#     print(f"\nDataset shape: {df.shape}")
#     print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
#     # Print sample
#     print("\n" + "=" * 70)
#     print("SAMPLE RECORDS")
#     print("=" * 70)
#     print("\nFirst 3 records:")
#     print(df.head(3).to_string())
    
#     return df


# # ============================================================================
# # STEP 5: DATA ANALYSIS & VALIDATION
# # ============================================================================

# def analyze_dataset(df):
#     """
#     Analyze dataset for quality and balance
#     """
#     print("\n" + "=" * 70)
#     print("DATASET ANALYSIS & VALIDATION")
#     print("=" * 70)
    
#     print("\n[Analysis 1] Feature Correlation with Diagnosis")
#     print("-" * 70)
    
#     # Convert categorical features to numeric for analysis
#     df_numeric = df.copy()
#     df_numeric['diagnosis_code'] = pd.Categorical(df_numeric['diagnosis']).codes
    
#     feature_cols = ['ping_gateway', 'has_ip', 'ping_ip', 'ping_domain', 'ip_conflict']
    
#     for feature in feature_cols:
#         correlation = df_numeric[feature].corr(df_numeric['diagnosis_code'])
#         print(f"{feature:20s}: {correlation:+.3f}")
    
#     print("\n[Analysis 2] Missing Values")
#     print("-" * 70)
#     missing = df.isnull().sum()
#     if missing.sum() == 0:
#         print("✓ No missing values found")
#     else:
#         print(missing[missing > 0])
    
#     print("\n[Analysis 3] Feature Value Distribution")
#     print("-" * 70)
#     for feature in feature_cols:
#         value_counts = df[feature].value_counts()
#         print(f"{feature:20s}: {dict(value_counts)}")
    
#     print("\n[Analysis 4] Class Balance (Diagnosis)")
#     print("-" * 70)
#     class_dist = df['diagnosis'].value_counts()
#     class_pct = df['diagnosis'].value_counts(normalize=True) * 100
    
#     for diagnosis in class_dist.index:
#         count = class_dist[diagnosis]
#         pct = class_pct[diagnosis]
#         bar = "█" * int(pct / 2)
#         print(f"{diagnosis:25s} | {count:3d} ({pct:5.1f}%) {bar}")
    
#     return df


# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================

# if __name__ == "__main__":
#     # Generate dataset
#     dataset = create_full_dataset(num_samples=300, save_to_csv=True)
    
#     # Analyze dataset
#     analyze_dataset(dataset)
    
#     print("\n" + "=" * 70)
#     print("✓ DATASET GENERATION COMPLETE")
#     print("=" * 70)

"""
NETWORK TROUBLESHOOTING DATASET GENERATOR v3.0
===============================================
Fixes applied vs v2.0:
  1. Added discriminating features: dns_response_time_ms, packet_loss_pct,
     traceroute_hops, arp_table_ok, subnet_matches_gw
  2. Fixed feature overlap between DNS Issue / DNS Timeout,
     Router Issue / Gateway Unreachable, Subnet Mismatch / Gateway Unreachable
  3. Removed noisy uncorrelated features: vpn_enabled, firewall_enabled,
     recently_updated (near-zero importance, pure noise)
  4. Richer symptom_text vocabulary per class (used later by TF-IDF)
  5. Tighter noise model (2% vs 5%) to preserve class signal

Usage:
    python network_dataset_generator.py
Output:
    network_dataset_v3.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ============================================================================
# SYMPTOM VARIATIONS  (richer vocabulary → better TF-IDF later)
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
# SOLUTIONS
# ============================================================================

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

# ============================================================================
# DATASET GENERATOR
# ============================================================================

def generate_dataset(num_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic network troubleshooting dataset with well-separated
    feature fingerprints per issue class.
    """
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
                # ── Text feature (for TF-IDF) ─────────────────────────────
                "symptom_text": random.choice(cfg["keywords"]),

                # ── Binary diagnostic flags ───────────────────────────────
                "ping_gateway":      syms.get("ping_gateway",  random.randint(0, 1)),
                "has_ip":            syms.get("has_ip",         random.randint(0, 1)),
                "ping_ip":           syms.get("ping_ip",        random.randint(0, 1)),
                "ping_domain":       syms.get("ping_domain",    random.randint(0, 1)),
                "ip_conflict":       syms.get("ip_conflict",    0),
                "arp_table_ok":      syms.get("arp_table_ok",   random.randint(0, 1)),
                "subnet_matches_gw": syms.get("subnet_matches_gw", 1),

                # ── Continuous discriminating features ────────────────────
                "dns_response_time_ms": random.randint(rt_lo, rt_hi),
                "packet_loss_pct":      random.randint(pl_lo, pl_hi),
                "traceroute_hops":      random.randint(hop_lo, max(hop_lo, hop_hi)),

                # ── Categorical context ───────────────────────────────────
                "network_type": random.choice(["WiFi", "Ethernet"]),
                "os_type":      random.choice(["Windows", "macOS", "Linux"]),

                # ── Target ────────────────────────────────────────────────
                "diagnosis": issue_type,
            }

            # Low noise: 2% chance of flipping a binary flag
            for feat in ["ping_gateway", "has_ip", "ping_ip", "ping_domain",
                         "arp_table_ok", "subnet_matches_gw"]:
                if random.random() < 0.02:
                    row[feat] = 1 - row[feat]

            rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def create_dataset(num_samples: int = 1200, save: bool = True) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("NETWORK TROUBLESHOOTING DATASET GENERATOR v3.0")
    print("=" * 70)

    print(f"\n[1/3] Generating {num_samples} samples...")
    df = generate_dataset(num_samples)
    print(f"  ✓ {len(df)} samples generated")

    print("[2/3] Adding solutions and metadata...")
    df["solutions"] = df["diagnosis"].map(
        lambda x: " | ".join(SOLUTIONS.get(x, ["No solution available"]))
    )
    df["severity"] = df["diagnosis"].map({
        "Network Adapter Issue": "High",
        "DHCP Failure":          "High",
        "IP Conflict":           "High",
        "Router Issue":          "Medium",
        "Gateway Unreachable":   "Medium",
        "Subnet Mismatch":       "Medium",
        "DNS Issue":             "Low",
        "DNS Timeout":           "Low",
    })
    df["dataset_version"] = "3.0"
    df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="H")

    col_order = [
        "symptom_text",
        "ping_gateway", "has_ip", "ping_ip", "ping_domain",
        "ip_conflict", "arp_table_ok", "subnet_matches_gw",
        "dns_response_time_ms", "packet_loss_pct", "traceroute_hops",
        "network_type", "os_type",
        "diagnosis", "severity", "solutions", "timestamp", "dataset_version",
    ]
    df = df[col_order]

    if save:
        path = "network_dataset_v3.csv"
        df.to_csv(path, index=False)
        print(f"  ✓ Saved to {path}")

    print("[3/3] Statistics:")
    print(f"  Total samples  : {len(df)}")
    print(f"  Features       : {len(df.columns)}")
    print(f"\n  Class distribution:")
    for diag in sorted(df["diagnosis"].unique()):
        n = (df["diagnosis"] == diag).sum()
        print(f"    {diag:25s}: {n:4d} ({n/len(df)*100:.1f}%)")

    return df


if __name__ == "__main__":
    df = create_dataset(num_samples=1200, save=True)
    print("\n✓ Dataset generation complete → network_dataset_v3.csv")