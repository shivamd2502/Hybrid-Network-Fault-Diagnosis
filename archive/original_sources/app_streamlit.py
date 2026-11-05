"""
NETWORK TROUBLESHOOTING ASSISTANT — STREAMLIT APP v3.1
=======================================================
Updated to match the final training pipeline which uses:
  - ColumnTransformer (TF-IDF on symptom_text + passthrough numeric/categorical)
  - RandomForestClassifier (no CalibratedClassifierCV wrapper)
  - Saved as models/pipeline.pkl + models/encoders.pkl

Run:
    streamlit run network_app_final.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from hybrid_system import HybridDiagnosticSystem

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🔧 Network Troubleshooting AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
<style>
.main-header {
    color: #0066cc;
    font-size: 2.3em;
    font-weight: 700;
    margin-bottom: 4px;
}
.sub-header {
    color: #666666;
    font-size: 1.05em;
    margin-bottom: 20px;
}
.diag-card {
    background-color: #e8f4fb;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #0066cc;
    margin: 16px 0;
}
.step-card {
    background-color: #f5f5f5;
    color: #000000;   /* 🔥 ADD THIS LINE */
    padding: 10px 14px;
    border-radius: 6px;
    border-left: 3px solid #0066cc;
    margin: 5px 0;
    font-size: 0.95em;
}
.badge-high {
    background:#c8e6c9; color:#1b5e20;
    padding:4px 14px; border-radius:20px;
    font-weight:700; font-size:0.88em;
}
.badge-med {
    background:#fff9c4; color:#f57f17;
    padding:4px 14px; border-radius:20px;
    font-weight:700; font-size:0.88em;
}
.badge-low {
    background:#ffcccc; color:#b71c1c;
    padding:4px 14px; border-radius:20px;
    font-weight:700; font-size:0.88em;
}
.feature-card {
    background:#f9f9f9;
    border-radius:8px;
    padding:10px;
    margin:4px 0;
    font-size:0.88em;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS — must match training pipeline exactly
# ============================================================================

NUMERIC_FEATURES = [
    "ping_gateway", "has_ip", "ping_ip", "ping_domain",
    "ip_conflict", "arp_table_ok", "subnet_matches_gw",
    "dns_response_time_ms", "packet_loss_pct", "traceroute_hops",
]
CATEGORICAL_FEATURES = ["network_type", "os_type"]
TEXT_FEATURE = "symptom_text"
DROP_COLS = ["solutions", "timestamp", "dataset_version", "severity"]

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
        "Ping 1.1.1.1 — if that works but DNS times out, ISP is throttling",
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

SEVERITY = {
    "Network Adapter Issue": "High",
    "DHCP Failure":          "High",
    "IP Conflict":           "High",
    "Router Issue":          "Medium",
    "Gateway Unreachable":   "Medium",
    "Subnet Mismatch":       "Medium",
    "DNS Issue":             "Low",
    "DNS Timeout":           "Low",
}

# ============================================================================
# LOAD MODEL  (matches new pipeline: models/pipeline.pkl + models/encoders.pkl)
# ============================================================================

@st.cache_resource(show_spinner="Loading Hybrid AI model...")
def load_hybrid_system():
    try:
        system = HybridDiagnosticSystem(models_dir="models")
        return system
    except Exception as e:
        st.error(f"Failed to load hybrid system: {e}")
        return None

# ============================================================================
# INFERENCE
# ============================================================================

def hybrid_predict(system: "HybridDiagnosticSystem", raw_input: dict) -> dict:
    """
    Adapter between Streamlit form data and HybridDiagnosticSystem.diagnose().
    
    raw_input keys:
        symptom_text            : str
        ping_gateway / has_ip / ping_ip / ping_domain
        ip_conflict / arp_table_ok / subnet_matches_gw  : int 0 or 1
        dns_response_time_ms / packet_loss_pct / traceroute_hops : numeric
        network_type / os_type  : int (label-encoded)
    """
    symptom_text = raw_input.pop("symptom_text")
    
    result = system.diagnose(
        symptom_text   = symptom_text,
        features       = raw_input,
        return_explanation = True,
    )
    
    # Reshape to match what the rest of the app expects
    return {
        "diagnosis":  result["diagnosis"],
        "confidence": result["confidence"],
        "all_probs":  result["all_probabilities"],
        # Extra hybrid fields for display
        "uncertainty":        result["uncertainty"],
        "fusion_strategy":    result["fusion_strategy"],
        "rule_reasons":       (result.get("rule_prediction") or {}).get("reasons", []),
        "rf_pred":            (result.get("rf_prediction")   or {}).get("diagnosis"),
        "rule_pred":          (result.get("rule_prediction") or {}).get("diagnosis"),
    }


def encode_categoricals(encoders: dict, network_type: str, os_type: str) -> tuple[int, int]:
    """Encode network_type and os_type using the saved LabelEncoders."""
    nt_le = encoders.get("network_type")
    os_le = encoders.get("os_type")

    nt_enc = int(nt_le.transform([network_type])[0]) if nt_le and network_type in nt_le.classes_ else 0
    os_enc = int(os_le.transform([os_type])[0])      if os_le and os_type      in os_le.classes_ else 0

    return nt_enc, os_enc


def keyword_hints(text: str) -> dict:
    """
    Fast keyword scan to pre-fill the follow-up form.
    The ML model uses full TF-IDF — this only improves UX defaults.
    """
    t = text.lower()
    h = {}
    if any(k in t for k in ["no ip", "169.254", "apipa", "dhcp"]):
        h["has_ip"] = 0
    if any(k in t for k in ["conflict", "duplicate ip", "same ip", "ip conflict"]):
        h["ip_conflict"] = 1
    if any(k in t for k in ["cannot ping gateway", "gateway unreachable", "no route"]):
        h["ping_gateway"] = 0
    if any(k in t for k in ["can ping ip", "ping ip but", "ping 8.8.8.8 works"]):
        h["ping_ip"] = 1
    if any(k in t for k in ["websites", "browser", "nslookup", "dns", "domain"]):
        h["ping_domain"] = 0
    if any(k in t for k in ["timeout", "slow dns", "nslookup timeout", "dns timeout"]):
        h["dns_response_time_ms"] = 15000
    if any(k in t for k in ["adapter", "driver", "ethernet unplugged", "no hardware", "nic"]):
        h["has_ip"] = 0
        h["arp_table_ok"] = 0
    if any(k in t for k in ["subnet", "netmask", "wrong mask", "subnet mismatch"]):
        h["subnet_matches_gw"] = 0
    return h

# ============================================================================
# SESSION STATE
# ============================================================================

def init_state():
    defaults = {
        "step":         1,
        "symptom_text": "",
        "hints":        {},
        "result":       None,
        "history":      [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        mode = st.radio(
            "Select Mode:",
            ["🔍 Diagnose", "📊 History", "ℹ️ About"],
            index=0,
        )

        st.divider()
        st.markdown("### 🤖 Model Info")
        st.markdown("""
        | Property | Value |
        |---|---|
        | Algorithm | Random Forest |
        | Features | TF-IDF + Structured |
        | Test accuracy | **99.6%** |
        | CV score | 98.75% ± 1.67% |
        | Version | 3.1 |
        """)

        st.divider()
        st.markdown("### 📋 Quick Tips")
        st.info("""
**Get the best diagnosis:**
- Describe symptoms in plain English
- Mention what you CAN and CANNOT do
- Include any error messages you see
- Specify WiFi or Ethernet
- Note OS (Windows / macOS / Linux)
        """)
    return mode

# ============================================================================
# DIAGNOSE TAB
# ============================================================================

def render_diagnose(system):
    st.markdown('<div class="main-header">🔧 Network Troubleshooting Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Network Diagnosis — describe your problem in plain English</div>', unsafe_allow_html=True)

    # Progress indicator
    step = st.session_state.step
    labels = ["Describe Problem", "Answer Questions", "View Diagnosis"]
    cols = st.columns(3)
    for i, (col, label) in enumerate(zip(cols, labels), 1):
        if i < step:
            color, weight, sym = "#4caf50", "400", "✓"
        elif i == step:
            color, weight, sym = "#0066cc", "700", "●"
        else:
            color, weight, sym = "#aaa", "400", "○"
        col.markdown(
            f'<div style="text-align:center;color:{color};font-weight:{weight}">'
            f'{sym} Step {i}: {label}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")

    # ── STEP 1: Describe problem ───────────────────────────────────────────────
    if step == 1:
        st.markdown("### Step 1️⃣ — Describe your network problem")

        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.step   = 1
                st.session_state.result = None
                st.session_state.hints  = {}
                st.rerun()

        symptom = st.text_area(
            "What is happening?",
            placeholder=(
                "E.g. 'Websites won't load but I can ping 8.8.8.8. "
                "nslookup times out every single time. "
                "Other devices on the same WiFi are also affected.'"
            ),
            height=130,
        )

        if st.button("▶ Analyse Symptom", type="primary", use_container_width=True):
            if symptom.strip():
                st.session_state.symptom_text = symptom.strip()
                st.session_state.hints = keyword_hints(symptom)
                st.session_state.step  = 2
                st.rerun()
            else:
                st.warning("⚠️ Please describe your problem before continuing.")

    # ── STEP 2: Follow-up questions ───────────────────────────────────────────
    elif step == 2:
        st.markdown("### Step 2️⃣ — Answer a few quick questions")

        symptom_preview = st.session_state.symptom_text
        if len(symptom_preview) > 120:
            symptom_preview = symptom_preview[:120] + "..."
        st.info(f"📝 Your symptom: *{symptom_preview}*")

        hints = st.session_state.hints

        with st.form("followup_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Connectivity checks**")

                ping_gw = st.radio(
                    "Can you ping the default gateway? (e.g. ping 192.168.1.1)",
                    ["Yes", "No"],
                    index=1 if hints.get("ping_gateway") == 0 else 0,
                    horizontal=True,
                )
                has_ip = st.radio(
                    "Does your device have a valid IP? (not 169.254.x.x)",
                    ["Yes", "No"],
                    index=1 if hints.get("has_ip") == 0 else 0,
                    horizontal=True,
                )
                ping_ip = st.radio(
                    "Can you ping an external IP? (e.g. ping 8.8.8.8)",
                    ["Yes", "No"],
                    index=0 if hints.get("ping_ip") == 1 else 1,
                    horizontal=True,
                )
                ping_dom = st.radio(
                    "Can you access websites or resolve domain names?",
                    ["Yes", "No"],
                    index=1 if hints.get("ping_domain") == 0 else 0,
                    horizontal=True,
                )

            with col2:
                st.markdown("**Network diagnostics**")

                ip_conflict = st.radio(
                    "Any IP address conflict warning shown by the OS?",
                    ["No", "Yes"],
                    index=1 if hints.get("ip_conflict") == 1 else 0,
                    horizontal=True,
                )
                arp_ok = st.radio(
                    "Does the ARP table look normal? (arp -a shows expected MACs)",
                    ["Yes", "No"],
                    index=1 if hints.get("arp_table_ok") == 0 else 0,
                    horizontal=True,
                )
                subnet_ok = st.radio(
                    "Does your subnet mask match the gateway subnet?",
                    ["Yes", "No"],
                    index=1 if hints.get("subnet_matches_gw") == 0 else 0,
                    horizontal=True,
                )

                st.markdown("**Measurements** *(estimate if unsure)*")
                dns_rt = st.slider(
                    "DNS response time (ms) — 0 = instant, 999+ = very slow / timed out",
                    min_value=0, max_value=30000,
                    value=hints.get("dns_response_time_ms", 200),
                    step=100,
                )
                pkt_loss = st.slider(
                    "Estimated packet loss (%)",
                    min_value=0, max_value=100,
                    value=hints.get("packet_loss_pct", 5),
                )
                hops = st.slider(
                    "Traceroute — how many hops before failure? (0 = not run / immediate fail)",
                    min_value=0, max_value=30,
                    value=hints.get("traceroute_hops", 5),
                )

            st.markdown("**System info**")
            c1, c2 = st.columns(2)
            net_type = c1.selectbox("Connection type", ["WiFi", "Ethernet"])
            os_type  = c2.selectbox("Operating system", ["Windows", "macOS", "Linux"])

            col_back, col_submit = st.columns([1, 3])
            with col_back:
                back = st.form_submit_button("← Back")
            with col_submit:
                submitted = st.form_submit_button("🔍 Get Diagnosis", type="primary", use_container_width=True)

        if back:
            st.session_state.step = 1
            st.rerun()

        if submitted:
            nt_enc, os_enc = encode_categoricals(
                system.encoders, net_type, os_type
            )
 
            raw = {
                "symptom_text":          st.session_state.symptom_text,
                "ping_gateway":          1 if ping_gw      == "Yes" else 0,
                "has_ip":                1 if has_ip       == "Yes" else 0,
                "ping_ip":               1 if ping_ip      == "Yes" else 0,
                "ping_domain":           1 if ping_dom     == "Yes" else 0,
                "ip_conflict":           1 if ip_conflict  == "Yes" else 0,
                "arp_table_ok":          1 if arp_ok       == "Yes" else 0,
                "subnet_matches_gw":     1 if subnet_ok    == "Yes" else 0,
                "dns_response_time_ms":  dns_rt,
                "packet_loss_pct":       pkt_loss,
                "traceroute_hops":       hops,
                "network_type":          nt_enc,
                "os_type":               os_enc,
            }
 
            h_result = hybrid_predict(system, dict(raw))  # dict() to avoid mutation
 
            result = {
                "diagnosis":        h_result["diagnosis"],
                "confidence":       h_result["confidence"],
                "all_probs":        h_result["all_probs"],
                "symptom":          st.session_state.symptom_text,
                "solutions":        SOLUTIONS.get(h_result["diagnosis"], []),
                "severity":         SEVERITY.get(h_result["diagnosis"], "Medium"),
                "timestamp":        datetime.now().isoformat(),
                "raw_input":        {k: v for k, v in raw.items()
                                     if k != "symptom_text"},
                # Hybrid-specific fields
                "uncertainty":      h_result.get("uncertainty", 0.0),
                "fusion_strategy":  h_result.get("fusion_strategy"),
                "rule_reasons":     h_result.get("rule_reasons", []),
                "rf_pred":          h_result.get("rf_pred"),
                "rule_pred":        h_result.get("rule_pred"),
            }
 
            st.session_state.result  = result
            st.session_state.history.append(result)
            st.session_state.step    = 3
            st.rerun()
    # ── STEP 3: Results ───────────────────────────────────────────────────────
    elif step == 3:
        result = st.session_state.result
        if not result:
            st.session_state.step = 1
            st.rerun()

        diag  = result["diagnosis"]
        conf  = result["confidence"]
        sev   = result["severity"]
        probs = result["all_probs"]

        # Confidence badge
        if conf >= 0.80:
            badge = '<span class="badge-high">🟢 High Confidence</span>'
        elif conf >= 0.55:
            badge = '<span class="badge-med">🟡 Medium Confidence</span>'
        else:
            badge = '<span class="badge-low">🔴 Low Confidence</span>'

        severity_color = {"High": "#c62828", "Medium": "#e65100", "Low": "#2e7d32"}.get(sev, "#555")

        st.markdown("### Step 3️⃣ — Diagnosis Result")
        st.markdown(f"""
        <div class="diag-card">
            <h2>🎯 {diag}</h2>
            <p>
                Confidence: <strong>{conf:.1%}</strong> &nbsp; {badge}
                &nbsp;&nbsp; Severity: <strong style="color:{severity_color}">{sev}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Hybrid model info ──────────────────────────────────────────
        unc      = result.get("uncertainty", 0.0)
        strategy = result.get("fusion_strategy", "—")
        rule_reasons = result.get("rule_reasons", [])
 
        st.markdown("#### 🔀 Hybrid model decision")
        c1, c2 = st.columns(2)
        c1.metric("Model uncertainty", f"{unc:.1%}")
        c2.metric("Fusion strategy",   strategy.replace("_", " ").title())
 
        if result.get("rf_pred") and result.get("rule_pred"):
            if result["rf_pred"] != result["rule_pred"]:
                st.warning(
                    f"RF predicted **{result['rf_pred']}**, "
                    f"rule engine predicted **{result['rule_pred']}** — "
                    f"Bayesian fusion resolved to **{diag}**."
                )
            else:
                st.success(
                    f"Both models agreed: **{diag}** (consensus)."
                )
 
        if rule_reasons:
            with st.expander("RFC rule reasoning"):
                for r in rule_reasons:
                    st.markdown(f"- {r}")

        # ── Probability chart across all 8 classes ────────────────────────────
        st.markdown("#### 📊 Model confidence across all issue types")
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        chart_df = pd.DataFrame(sorted_probs, columns=["Issue", "Probability (%)"])
        chart_df["Probability (%)"] = (chart_df["Probability (%)"] * 100).round(2)
        st.bar_chart(chart_df.set_index("Issue"), height=260)

        # ── Solution steps ────────────────────────────────────────────────────
        st.markdown("#### 🛠️ Recommended Fix Steps")
        for i, step_txt in enumerate(result["solutions"], 1):
            st.markdown(
                f'<div class="step-card"><strong>Step {i}:</strong> {step_txt}</div>',
                unsafe_allow_html=True,
            )

        # ── Feature summary: what drove the diagnosis ─────────────────────────
        st.markdown("#### 🔍 Key diagnostic signals")
        raw = result["raw_input"]

        feature_display = [
            ("ping_gateway",          raw.get("ping_gateway"),          "Gateway reachable"),
            ("has_ip",                raw.get("has_ip"),                "Valid IP assigned"),
            ("ping_ip",               raw.get("ping_ip"),               "External IP pingable"),
            ("ping_domain",           raw.get("ping_domain"),           "Domains resolving"),
            ("ip_conflict",           raw.get("ip_conflict"),           "IP conflict detected"),
            ("arp_table_ok",          raw.get("arp_table_ok"),          "ARP table healthy"),
            ("subnet_matches_gw",     raw.get("subnet_matches_gw"),     "Subnet matches gateway"),
            ("dns_response_time_ms",  raw.get("dns_response_time_ms"),  "DNS response time (ms)"),
            ("packet_loss_pct",       raw.get("packet_loss_pct"),       "Packet loss (%)"),
            ("traceroute_hops",       raw.get("traceroute_hops"),       "Traceroute hops"),
        ]

        binary_keys = {"ping_gateway","has_ip","ping_ip","ping_domain",
                       "arp_table_ok","subnet_matches_gw"}
        # ip_conflict: 1 = bad (red), 0 = fine (green)
        alert_keys = {"ip_conflict"}

        feat_cols = st.columns(5)
        for i, (key, val, label) in enumerate(feature_display):
            with feat_cols[i % 5]:
                if key in binary_keys:
                    good  = val == 1
                    color = "#2e7d32" if good else "#c62828"
                    sym   = "✓" if good else "✗"
                    disp  = "Yes" if good else "No"
                elif key in alert_keys:
                    good  = val == 0
                    color = "#2e7d32" if good else "#c62828"
                    sym   = "✓" if good else "⚠"
                    disp  = "No" if good else "Yes"
                else:
                    color = "#1565c0"
                    sym   = "📊"
                    disp  = str(val)

                st.markdown(f"""
                <div class="feature-card" style="border-left:4px solid {color}">
                    <div style="color:{color};font-weight:700">{sym} {label}</div>
                    <div style="color:#555">{disp}</div>
                </div>""", unsafe_allow_html=True)

        # ── Actions ───────────────────────────────────────────────────────────
        st.markdown("---")
        c1, c2, c3 = st.columns(3)

        with c1:
            report = {
                "diagnosis":  diag,
                "confidence": f"{conf:.1%}",
                "severity":   sev,
                "symptom":    result["symptom"],
                "timestamp":  result["timestamp"],
                "solutions":  result["solutions"],
                "all_probs":  {k: f"{v:.1%}" for k, v in probs.items()},
            }
            st.download_button(
                "💾 Download Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name=f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        with c2:
            if st.button("✏️ Refine Answers", use_container_width=True):
                st.session_state.step = 2
                st.rerun()

        with c3:
            if st.button("🔄 New Diagnosis", use_container_width=True):
                st.session_state.step   = 1
                st.session_state.result = None
                st.session_state.hints  = {}
                st.rerun()

# ============================================================================
# HISTORY TAB
# ============================================================================

def render_history():
    st.markdown('<div class="main-header">📊 Diagnosis History</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">All diagnoses from this session</div>', unsafe_allow_html=True)

    history = st.session_state.history

    if not history:
        st.info("📭 No diagnoses yet — go to 🔍 Diagnose to get started.")
        return

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Diagnoses", len(history))
    avg_conf = np.mean([h["confidence"] for h in history])
    m2.metric("Average Confidence", f"{avg_conf:.1%}")
    most_common = pd.Series([h["diagnosis"] for h in history]).mode()[0]
    m3.metric("Most Common Issue", most_common)
    high_sev = sum(1 for h in history if h.get("severity") == "High")
    m4.metric("High Severity Cases", high_sev)

    st.divider()

    # History table
    rows = []
    for h in history:
        rows.append({
            "Time":       datetime.fromisoformat(h["timestamp"]).strftime("%H:%M:%S"),
            "Symptom":    h["symptom"][:65] + ("..." if len(h["symptom"]) > 65 else ""),
            "Diagnosis":  h["diagnosis"],
            "Confidence": f"{h['confidence']:.1%}",
            "Severity":   h.get("severity", "—"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Frequency chart
    if len(history) > 1:
        st.markdown("#### Issue frequency")
        freq = pd.Series([h["diagnosis"] for h in history]).value_counts()
        st.bar_chart(freq, height=240)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        export = [
            {k: v for k, v in h.items() if k != "raw_input"}
            for h in history
        ]
        st.download_button(
            "💾 Export History as JSON",
            data=json.dumps(export, indent=2),
            file_name=f"diagnosis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ============================================================================
# ABOUT TAB
# ============================================================================

def render_about():
    st.markdown('<div class="main-header">ℹ️ About This Assistant</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
### 🤖 Technology Stack

**Machine Learning:**
- Algorithm: Random Forest (200 trees, balanced class weights)
- Text features: TF-IDF (100 unigram + bigram features on symptom description)
- Structured features: 10 binary/numeric diagnostic flags + 2 categorical
- Feature engineering: `ColumnTransformer` (sklearn) — no data leakage
- Test accuracy: **99.6%** | CV: 98.75% ± 1.67%

**Pipeline architecture:**
```
symptom_text  →  TF-IDF (100 features)  ┐
numeric cols  →  passthrough            ├─ RandomForest → Diagnosis
categorical   →  passthrough (encoded)  ┘
```

**Framework:**
- Streamlit (web UI)
- scikit-learn (ML pipeline)
- Python 3.10+
        """)

    with c2:
        st.markdown("""
### 🎯 Supported Issues

| Issue | Severity |
|---|---|
| Router Issue | Medium |
| DNS Issue | Low |
| DNS Timeout | Low |
| IP Conflict | High |
| DHCP Failure | High |
| Gateway Unreachable | Medium |
| Network Adapter Issue | High |
| Subnet Mismatch | Medium |

### 🔑 Key discriminating features

| Confused pair | Discriminator |
|---|---|
| DNS Issue vs DNS Timeout | `dns_response_time_ms` |
| Router vs Gateway | `traceroute_hops`, `packet_loss_pct` |
| Subnet Mismatch vs Gateway | `subnet_matches_gw` |
| DHCP Failure vs Adapter | `arp_table_ok` |
        """)

    st.divider()

    st.markdown("""
### 📚 Useful Diagnostic Commands

```bash
# Check IP configuration
ipconfig /all             # Windows
ip addr show              # Linux / macOS

# Test connectivity layer by layer
ping 192.168.1.1          # 1. Gateway (layer 3)
ping 8.8.8.8              # 2. External IP (routing)
ping google.com           # 3. DNS + internet (layer 7)

# DNS diagnosis
nslookup google.com                   # basic DNS test
nslookup -timeout=5 google.com        # detect DNS timeout
nslookup google.com 1.1.1.1           # test alternate DNS

# ARP / routing
arp -a                    # check for duplicate MACs (IP conflict)
route print               # Windows — check default gateway
ip route show             # Linux — check routing table

# Traceroute
tracert google.com        # Windows
traceroute 8.8.8.8        # Linux / macOS
```
    """)

# ============================================================================
# MAIN
# ============================================================================

def main():
    init_state()
 
    # Replace:  pipeline, encoders = load_model()
    # With:
    system = load_hybrid_system()
 
    mode = render_sidebar()
 
    if system is None:
        st.error("Model not found — run training scripts first.")
        return
 
    if mode == "🔍 Diagnose":
        render_diagnose(system)        # <-- pass system, not pipeline+encoders
    elif mode == "📊 History":
        render_history()
    elif mode == "ℹ️ About":
        render_about()


if __name__ == "__main__":
    main()